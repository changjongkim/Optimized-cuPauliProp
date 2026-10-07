// Out-of-core parity-owned Pauli propagation on ONE GPU.
//
// Terms are split into buckets by g parity rows that every rotation of the current layer
// leaves unchanged. Buckets therefore never exchange or merge terms during that layer, so
// each bucket can be loaded onto the GPU, taken through the whole layer (127 gates with
// local deduplication and truncation), and written back, one bucket at a time. Between
// layers the output of each bucket is partitioned by the next layer's rows and appended to
// the next set of buckets in host memory. GPU memory has to hold one bucket, not the
// operator; the host bucket set at a layer boundary is a consistent checkpoint.
//
// Uses the Clifford-frame form of the kicked Ising circuit (see ki_prof.cpp --frame).

#include <cupauliprop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include "../driver/ibm_eagle.hpp"

extern "C" {
void pp_owners(const uint64_t* xz, int64_t n, int words, const uint64_t* masks, int g, uint8_t* owners);
void pp_counts_sorted(const uint8_t* owners, int64_t n, int R, int64_t* cnt);
void pp_owners_strided(const uint64_t* xz, int64_t n, int64_t stride, int words, const uint64_t* masks, int g, uint8_t* owners);
int pp_partition(const uint64_t* xz, const double* coef, uint8_t* owners, int64_t n, int words,
                 uint64_t* xzOut, double* coefOut, int32_t* idx);
}

[[noreturn]] static void die(const char* what, int line) {
  fprintf(stderr, "FATAL: %s (line %d)\n", what, line);
  std::exit(1);
}
#define CUPP(x) do { cupaulipropStatus_t s_ = (x); if (s_ != CUPAULIPROP_STATUS_SUCCESS) { \
  fprintf(stderr, "cuPauliProp status %d\n", (int)s_); die(#x, __LINE__); } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e_)); die(#x, __LINE__); } } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Opts {
  int steps = 20;
  double thetaOverPi = 0.25;
  double coef = 1e-4;
  int weight = 0;
  int truncEvery = 10;
  int obsQubit = 62;
  double vramFrac = 0.9;
  double expShare = 0.2;
  long target = 4000000;      // desired terms per bucket at the start of a layer
  int gmax = 8;               // at most 2^gmax buckets
  int cands = 8;
  long sample = 200000;
  int seed = 7;
  int ckptLayer = -1;         // write a checkpoint when this layer has been completed
  int stopAtCkpt = 0;         // 1: exit right after writing the checkpoint
  int rebucket = 0;           // on resume: 1 = merge all checkpointed buckets into one (re-split on demand)
  std::string ckptFile;       // checkpoint to write
  std::string resume;         // checkpoint to resume from
  std::string tag = "ooc";
  std::string outdir = ".";
};

struct Gate { cupaulipropQuantumOperator_t op; bool trunc; };
struct Bucket { std::vector<uint64_t> xz; std::vector<double> cf; };
struct Work { Bucket b; int startQ; };   // a bucket and the next gate of the layer to apply to it

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { if (i + 1 >= argc) die("missing option value", __LINE__); return argv[++i]; };
    if      (a == "--steps")       o.steps = atoi(next());
    else if (a == "--theta")       o.thetaOverPi = atof(next());
    else if (a == "--coef")        o.coef = atof(next());
    else if (a == "--weight")      o.weight = atoi(next());
    else if (a == "--trunc-every") o.truncEvery = atoi(next());
    else if (a == "--obs-qubit")   o.obsQubit = atoi(next());
    else if (a == "--vram-frac")   o.vramFrac = atof(next());
    else if (a == "--exp-share")   o.expShare = atof(next());
    else if (a == "--target")      o.target = atol(next());
    else if (a == "--gmax")        o.gmax = atoi(next());
    else if (a == "--cands")       o.cands = atoi(next());
    else if (a == "--seed")        o.seed = atoi(next());
    else if (a == "--ckpt-layer")  o.ckptLayer = atoi(next());
    else if (a == "--ckpt-file")   o.ckptFile = next();
    else if (a == "--stop-at-ckpt") o.stopAtCkpt = atoi(next());
    else if (a == "--resume")      o.resume = next();
    else if (a == "--rebucket")    o.rebucket = atoi(next());
    else if (a == "--tag")         o.tag = next();
    else if (a == "--outdir")      o.outdir = next();
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }
  if (o.gmax > 8) die("--gmax is at most 8", __LINE__);

  CUDA(cudaSetDevice(0));
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  CUPP(cupaulipropCreate(&handle));

  int32_t nw = 0;
  CUPP(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &nw));
  const int words = 2 * nw;
  const size_t xzPerTerm = words * sizeof(uint64_t), perTerm = xzPerTerm + sizeof(double);
  size_t freeMem = 0, totalMem = 0;
  CUDA(cudaMemGetInfo(&freeMem, &totalMem));
  const size_t usable = (size_t)(freeMem * o.vramFrac);
  const long capTerms = (long)((size_t)(usable * o.expShare) / perTerm);
  const size_t xzMem = (size_t)capTerms * xzPerTerm, cfMem = (size_t)capTerms * sizeof(double);
  const size_t wsMem = usable - 2 * (xzMem + cfMem) - (size_t)capTerms * 11;
  void *xz[2], *cf[2], *dWs;
  for (int b = 0; b < 2; b++) { CUDA(cudaMalloc(&xz[b], xzMem)); CUDA(cudaMalloc(&cf[b], cfMem)); }
  CUDA(cudaMalloc(&dWs, wsMem));
  uint8_t* dOwners; CUDA(cudaMalloc((void**)&dOwners, (size_t)capTerms));
  int32_t* dIdx; CUDA(cudaMalloc((void**)&dIdx, (size_t)capTerms * sizeof(int32_t)));
  int cur = 0;
  cupaulipropPauliExpansion_t ex[2];
  bool haveEx = false;
  auto makeExpansions = [&](long nCur) {
    if (haveEx) { CUPP(cupaulipropDestroyPauliExpansion(ex[0])); CUPP(cupaulipropDestroyPauliExpansion(ex[1])); }
    cur = 0;
    CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, xz[0], xzMem, cf[0], cfMem,
                                         CUDA_R_64F, nCur, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[0]));
    CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, xz[1], xzMem, cf[1], cfMem,
                                         CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[1]));
    haveEx = true;
  };
  cupaulipropWorkspaceDescriptor_t ws;
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));

  cupaulipropCoefficientTruncationParams_t coefParams; coefParams.cutoff = o.coef;
  cupaulipropPauliWeightTruncationParams_t weightParams; weightParams.cutoff = o.weight;
  std::vector<cupaulipropTruncationStrategy_t> trunc;
  trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
  if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});

  // ---- Clifford-frame circuit: layer n holds 127 rotations ----
  const double PI = 3.14159265358979323846;
  const int32_t (*zzLayers[3])[2] = {ZZ_QUBITS_RED, ZZ_QUBITS_BLUE, ZZ_QUBITS_GREEN};
  std::vector<std::vector<int32_t>> nb(NUM_CIRCUIT_QUBITS);
  for (int l = 0; l < 3; l++)
    for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) {
      nb[zzLayers[l][i][0]].push_back(zzLayers[l][i][1]);
      nb[zzLayers[l][i][1]].push_back(zzLayers[l][i][0]);
    }
  std::vector<std::vector<Gate>> layers(o.steps);
  for (int n = 0; n < o.steps; n++) {
    for (int32_t q = 0; q < NUM_CIRCUIT_QUBITS; q++) {
      const int deg = (int)nb[q].size();
      double sign = ((deg % 2) && ((n / 2) % 2)) ? -1.0 : 1.0;
      std::vector<int32_t> qs; std::vector<cupaulipropPauliKind_t> ps;
      qs.push_back(q);
      if (n % 2 == 0) {
        ps.push_back(CUPAULIPROP_PAULI_X);
      } else {
        ps.push_back(deg % 2 ? CUPAULIPROP_PAULI_Y : CUPAULIPROP_PAULI_X);
        for (int32_t b : nb[q]) { qs.push_back(b); ps.push_back(CUPAULIPROP_PAULI_Z); }
        if (deg % 4 == 2 || deg % 4 == 3) sign *= -1.0;
      }
      Gate g;
      CUPP(cupaulipropCreatePauliRotationGateOperator(handle, sign * o.thetaOverPi * PI, (int32_t)qs.size(), qs.data(), ps.data(), &g.op));
      g.trunc = ((271 * n + q) % o.truncEvery == 0) || (q == 0 && n > 0);
      layers[n].push_back(g);
    }
  }

  auto numTerms = [&]() { int64_t n = 0; CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &n)); return (long)n; };
  auto setWs = [&]() {
    int64_t need = 0;
    CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
    if (need > (int64_t)wsMem) { fprintf(stderr, "workspace exceeded: need %ld have %zu\n", (long)need, wsMem); die("workspace", __LINE__); }
    CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
  };

  // Row sets: a row is a random set m of Z bits; odd layers also carry the X bits L*m.
  auto makeRows = [&](std::mt19937_64& rng, int g, bool oddFrame) {
    std::vector<uint64_t> rows((size_t)std::max(g, 1) * words, 0);
    for (int j = 0; j < g; j++) {
      uint64_t* row = &rows[(size_t)j * words];
      for (int w = 0; w < nw; w++) {
        uint64_t v = rng();
        if (w == nw - 1 && NUM_CIRCUIT_QUBITS % 64) v &= (1ULL << (NUM_CIRCUIT_QUBITS % 64)) - 1;
        row[nw + w] = v;
      }
      if (oddFrame)
        for (int a = 0; a < NUM_CIRCUIT_QUBITS; a++) {
          if (!((row[nw + a / 64] >> (a % 64)) & 1ULL)) continue;
          if (nb[a].size() % 2) row[a / 64] ^= (1ULL << (a % 64));
          for (int32_t b : nb[a]) row[b / 64] ^= (1ULL << (b % 64));
        }
    }
    return rows;
  };
  auto ownerCpu = [&](const uint64_t* t, const std::vector<uint64_t>& rows, int g) {
    int ow = 0;
    for (int j = 0; j < g; j++) {
      int par = 0;
      for (int w = 0; w < words; w++) par ^= __builtin_popcountll(t[w] & rows[(size_t)j * words + w]) & 1;
      ow |= par << j;
    }
    return ow;
  };

  std::vector<Bucket> curB(1), nextB;
  curB[0].xz.assign(words, 0);
  curB[0].xz[nw + o.obsQubit / 64] |= (1ULL << (o.obsQubit % 64));
  curB[0].cf.assign(1, 1.0);
  int firstLayer = o.steps - 1;

  // A checkpoint is the bucket set at a layer boundary: every bucket is a union of parity
  // classes of the next layer, so the buckets can be reloaded as they are, merged, or re-split.
  auto writeCheckpoint = [&](int nextLayer) {
    FILE* f = fopen(o.ckptFile.c_str(), "wb");
    if (!f) die("cannot open checkpoint for writing", __LINE__);
    int32_t hdr[3] = {0x50504B43, words, nextLayer};
    int64_t nbk = (int64_t)curB.size(), total = 0;
    fwrite(hdr, sizeof(int32_t), 3, f); fwrite(&nbk, sizeof(int64_t), 1, f);
    for (auto& b : curB) {
      int64_t n = (int64_t)b.cf.size(); total += n;
      fwrite(&n, sizeof(int64_t), 1, f);
      fwrite(b.xz.data(), sizeof(uint64_t), (size_t)n * words, f);
      fwrite(b.cf.data(), sizeof(double), (size_t)n, f);
    }
    fclose(f);
    printf("checkpoint written: next layer %d, %ld buckets, %ld terms, %.3f GiB\n", nextLayer, (long)nbk, (long)total,
           total * (double)perTerm / 1073741824.0);
  };
  if (!o.resume.empty()) {
    FILE* f = fopen(o.resume.c_str(), "rb");
    if (!f) die("cannot open checkpoint for reading", __LINE__);
    int32_t hdr[3]; int64_t nbk = 0;
    if (fread(hdr, sizeof(int32_t), 3, f) != 3 || hdr[0] != 0x50504B43 || hdr[1] != words) die("bad checkpoint", __LINE__);
    if (fread(&nbk, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
    firstLayer = hdr[2];
    curB.assign(o.rebucket ? 1 : (size_t)nbk, Bucket());
    long total = 0;
    for (int64_t i = 0; i < nbk; i++) {
      int64_t n = 0;
      if (fread(&n, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
      Bucket& b = curB[o.rebucket ? 0 : (size_t)i];
      const size_t o1 = b.xz.size(), o2 = b.cf.size();
      b.xz.resize(o1 + (size_t)n * words); b.cf.resize(o2 + (size_t)n);
      if (fread(b.xz.data() + o1, sizeof(uint64_t), (size_t)n * words, f) != (size_t)n * words) die("bad checkpoint", __LINE__);
      if (fread(b.cf.data() + o2, sizeof(double), (size_t)n, f) != (size_t)n) die("bad checkpoint", __LINE__);
      total += n;
    }
    fclose(f);
    printf("resumed from checkpoint: next layer %d, %ld terms in %zu bucket(s)\n", firstLayer, total, curB.size());
  }

  printf("config: steps=%d theta=%.4fpi coef=%g weight=%d truncEvery=%d GPU capacity=%ld terms/buffer workspace=%.1f GiB "
         "bucket target=%ld gmax=%d\n", o.steps, o.thetaOverPi, o.coef, o.weight, o.truncEvery, capTerms, wsMem / 1073741824.0, o.target, o.gmax);
  std::mt19937_64 rng(o.seed);
  double tGates = 0, tPart = 0, tH2D = 0, tD2H = 0, tAppend = 0, tSelect = 0, expec = 0;
  long maxBucketPeak = 0, maxGlobal = 0, hostPeakBytes = 0, bucketRuns = 0, movedTerms = 0, splits = 0;
  double growth = 2.0;
  std::vector<uint64_t> stageXz; std::vector<double> stageCf;
  const double tBegin = now();

  for (int n = firstLayer; n >= 0; n--) {
    long Nin = 0;
    for (auto& b : curB) Nin += (long)b.cf.size();
    maxGlobal = std::max(maxGlobal, Nin);

    // Choose the bucket count and rows for the NEXT layer from a sample of the current terms.
    int gNext = 0;
    std::vector<uint64_t> rowsNext(words, 0);
    if (n > 0) {
      const double est = Nin * growth;
      while (gNext < o.gmax && est / (double)(1L << gNext) > (double)o.target) gNext++;
      const double ts = now();
      const bool odd = ((n - 1) % 2 == 1);
      double bestMax = 1e300;
      const long stride = std::max<long>(1, Nin / std::max<long>(1, o.sample));
      for (int c = 0; c < (gNext ? o.cands : 1); c++) {
        std::vector<uint64_t> rows = makeRows(rng, gNext, odd);
        std::vector<long> h((size_t)1 << gNext, 0);
        for (auto& b : curB)
          for (size_t i = 0; i < b.cf.size(); i += stride) h[ownerCpu(&b.xz[i * words], rows, gNext)]++;
        const double mx = (double)*std::max_element(h.begin(), h.end());
        if (mx < bestMax) { bestMax = mx; rowsNext = rows; }
      }
      tSelect += now() - ts;
      nextB.assign((size_t)1 << gNext, Bucket());
    }

    long Nout = 0;
    std::deque<Work> queue;
    for (auto& b : curB)
      if (!b.cf.empty()) { queue.push_back(Work()); queue.back().b.xz.swap(b.xz); queue.back().b.cf.swap(b.cf); queue.back().startQ = NUM_CIRCUIT_QUBITS - 1; }
    while (!queue.empty()) {
      Work w; w.b.xz.swap(queue.front().b.xz); w.b.cf.swap(queue.front().b.cf); w.startQ = queue.front().startQ;
      queue.pop_front();
      Bucket& b = w.b;
      const size_t bi = (size_t)-1;
      const long nb0 = (long)b.cf.size();
      if (nb0 > capTerms) { fprintf(stderr, "bucket has %ld terms, GPU buffer holds %ld\n", nb0, capTerms); die("bucket too large to load", __LINE__); }
      bucketRuns++;
      double t0 = now();
      CUDA(cudaMemcpy(xz[0], b.xz.data(), (size_t)nb0 * xzPerTerm, cudaMemcpyHostToDevice));
      CUDA(cudaMemcpy(cf[0], b.cf.data(), (size_t)nb0 * sizeof(double), cudaMemcpyHostToDevice));
      std::vector<uint64_t>().swap(b.xz); std::vector<double>().swap(b.cf);
      makeExpansions(nb0);
      tH2D += now() - t0;

      // The rest of the layer for this bucket: rotations with local deduplication and truncation.
      // If the bucket outgrows the GPU, it is split in two by one more row of this layer's family
      // (the halves stay independent for the remaining gates) and both halves are queued.
      t0 = now();
      long peak = nb0;
      bool split = false;
      for (int q = w.startQ; q >= 0; q--) {
        const Gate& g = layers[n][q];
        const long m = numTerms();
        if (m == 0) break;
        peak = std::max(peak, m);
        cupaulipropPauliExpansionView_t v;
        CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
        int64_t rp = 0, rc = 0, need = 0;
        const int32_t nT = g.trunc ? (int32_t)trunc.size() : 0;
        cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorApplication(
          handle, v, g.op, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, g.trunc ? trunc.data() : nullptr, wsMem, &rp, &rc, ws);
        if (sp == CUPAULIPROP_STATUS_SUCCESS)
          CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
        if (sp != CUPAULIPROP_STATUS_SUCCESS || rp > (int64_t)xzMem || rc > (int64_t)cfMem || need > (int64_t)wsMem) {
          CUPP(cupaulipropDestroyPauliExpansionView(v));
          if (m < 1000) die("cannot split further", __LINE__);
          std::vector<uint64_t> row = makeRows(rng, 1, n % 2 == 1);
          int64_t c2[2] = {0, 0};
          pp_owners((const uint64_t*)xz[cur], m, words, row.data(), 1, dOwners);
          if (pp_partition((const uint64_t*)xz[cur], (const double*)cf[cur], dOwners, m, words,
                           (uint64_t*)xz[1 - cur], (double*)cf[1 - cur], dIdx) != 0) die("partition", __LINE__);
          pp_counts_sorted(dOwners, m, 2, c2);
          long off2 = 0;
          for (int h = 0; h < 2; h++) {
            if (c2[h] == 0) continue;
            queue.push_front(Work());
            Work& nwk = queue.front();
            nwk.startQ = q;
            nwk.b.xz.resize((size_t)c2[h] * words); nwk.b.cf.resize((size_t)c2[h]);
            CUDA(cudaMemcpy(nwk.b.xz.data(), (char*)xz[1 - cur] + off2 * xzPerTerm, (size_t)c2[h] * xzPerTerm, cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(nwk.b.cf.data(), (char*)cf[1 - cur] + off2 * sizeof(double), (size_t)c2[h] * sizeof(double), cudaMemcpyDeviceToHost));
            off2 += c2[h];
          }
          splits++; split = true;
          break;
        }
        CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
        CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
          handle, v, ex[1 - cur], g.op, 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, g.trunc ? trunc.data() : nullptr, ws, stream));
        CUDA(cudaStreamSynchronize(stream));
        CUPP(cupaulipropDestroyPauliExpansionView(v));
        cur ^= 1;
      }
      if (split) { tGates += now() - t0; continue; }
      maxBucketPeak = std::max(maxBucketPeak, peak);
      const long m = numTerms();
      Nout += m;
      tGates += now() - t0;
      if (m == 0) continue;

      if (n == 0) {                                   // last layer: trace with |0...0>
        cupaulipropPauliExpansionView_t v;
        CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
        CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, wsMem, ws));
        setWs();
        double sig = 0, e2 = 0;
        CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &e2, ws, stream));
        CUDA(cudaStreamSynchronize(stream));
        expec += sig * std::pow(2.0, e2);
        CUPP(cupaulipropDestroyPauliExpansionView(v));
        continue;
      }

      // Partition this bucket's output by the next layer's rows and append to the next buckets.
      t0 = now();
      std::vector<int64_t> cnt((size_t)1 << gNext, 0);
      const void *srcXz = xz[cur], *srcCf = cf[cur];
      if (gNext > 0) {
        pp_owners((const uint64_t*)xz[cur], m, words, rowsNext.data(), gNext, dOwners);
        if (pp_partition((const uint64_t*)xz[cur], (const double*)cf[cur], dOwners, m, words,
                         (uint64_t*)xz[1 - cur], (double*)cf[1 - cur], dIdx) != 0) die("partition", __LINE__);
        pp_counts_sorted(dOwners, m, 1 << gNext, cnt.data());
        srcXz = xz[1 - cur]; srcCf = cf[1 - cur];
      } else {
        cnt[0] = m;
      }
      CUDA(cudaDeviceSynchronize());
      tPart += now() - t0;
      t0 = now();
      stageXz.resize((size_t)m * words); stageCf.resize((size_t)m);
      CUDA(cudaMemcpy(stageXz.data(), srcXz, (size_t)m * xzPerTerm, cudaMemcpyDeviceToHost));
      CUDA(cudaMemcpy(stageCf.data(), srcCf, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost));
      tD2H += now() - t0;
      t0 = now();
      long off = 0;
      for (size_t d = 0; d < cnt.size(); d++) {
        if (cnt[d] == 0) continue;
        nextB[d].xz.insert(nextB[d].xz.end(), stageXz.begin() + off * words, stageXz.begin() + (off + cnt[d]) * words);
        nextB[d].cf.insert(nextB[d].cf.end(), stageCf.begin() + off, stageCf.begin() + off + cnt[d]);
        off += cnt[d];
        if (d != bi) movedTerms += cnt[d];
      }
      tAppend += now() - t0;
      long hostBytes = 0;
      for (auto& x : queue) hostBytes += (long)(x.b.cf.size() * perTerm);
      for (auto& x : nextB) hostBytes += (long)(x.cf.size() * perTerm);
      hostPeakBytes = std::max(hostPeakBytes, hostBytes);
    }
    if (Nin > 0 && Nout > 0) growth = std::max(1.0, std::min(8.0, (double)Nout / (double)Nin * 1.3));
    if (n > 0) { curB.swap(nextB); nextB.clear(); }
    maxGlobal = std::max(maxGlobal, Nout);
    printf("layer %2d: terms in=%ld out=%ld buckets next=%d elapsed=%.1f s\n", n, Nin, Nout, 1 << gNext, now() - tBegin);
    fflush(stdout);
    if (n == o.ckptLayer && n > 0 && !o.ckptFile.empty()) {
      writeCheckpoint(n - 1);
      if (o.stopAtCkpt) { printf("stopped at checkpoint\n"); return 0; }
    }
  }
  const double wall = now() - tBegin;
  printf("expectation=%.12g\n", expec);
  printf("max_global_terms_at_layer_boundary=%ld largest_bucket_on_gpu=%ld terms (%.2f GiB) host_peak=%.2f GiB bucket_runs=%ld mid_layer_splits=%ld\n",
         maxGlobal, maxBucketPeak, maxBucketPeak * (double)perTerm / 1073741824.0, hostPeakBytes / 1073741824.0, bucketRuns, splits);
  printf("wall=%.3f s: gates=%.3f partition=%.3f host->gpu=%.3f gpu->host=%.3f host append=%.3f row selection=%.3f\n",
         wall, tGates, tPart, tH2D, tD2H, tAppend, tSelect);
  return 0;
}
