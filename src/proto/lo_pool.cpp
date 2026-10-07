// Bucket-task Pauli propagation on the GPUs of one node, with no MPI.
//
// The operator is kept in host memory as buckets defined by g parity rows that every
// rotation of the current layer leaves unchanged; buckets are therefore independent tasks
// for the whole layer. One worker thread per GPU repeatedly takes a bucket from the shared
// pool, runs the whole layer on it (127 rotations with local deduplication and truncation),
// partitions the result by the next layer's rows and appends the pieces to the next pool.
// A bucket that outgrows its GPU is split in two by one more row of the same family and both
// halves go back to the pool. GPUs never exchange data with each other.
//
// Uses the Clifford-frame form of the kicked Ising circuit (see ki_prof.cpp --frame).

#include <cupauliprop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../driver/ibm_eagle.hpp"

extern "C" {
void pp_owners(const uint64_t* xz, int64_t n, int words, const uint64_t* masks, int g, uint8_t* owners);
void pp_counts_sorted(const uint8_t* owners, int64_t n, int R, int64_t* cnt);
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
  int gpus = 1;
  int steps = 20;
  double thetaOverPi = 0.25;
  double coef = 1e-4;
  int weight = 0;
  int truncEvery = 10;
  int obsQubit = 62;
  double vramFrac = 0.9;
  double expShare = 0.2;
  long target = 4000000;      // desired terms per bucket at the start of a layer
  int gmax = 8;
  int cands = 8;
  long sample = 200000;
  int seed = 7;
};

struct Gate { cupaulipropQuantumOperator_t op; bool trunc; };
struct Bucket { std::vector<uint64_t> xz; std::vector<double> cf; };
struct Work { Bucket b; int startQ; };

static int g_nw = 0, g_words = 0;
static std::vector<std::vector<int32_t>> g_nb;

// A row is a random set m of Z bits; odd layers also carry the X bits L*m (L = lattice Laplacian mod 2).
static std::vector<uint64_t> makeRows(std::mt19937_64& rng, int g, bool oddFrame) {
  std::vector<uint64_t> rows((size_t)std::max(g, 1) * g_words, 0);
  for (int j = 0; j < g; j++) {
    uint64_t* row = &rows[(size_t)j * g_words];
    for (int w = 0; w < g_nw; w++) {
      uint64_t v = rng();
      if (w == g_nw - 1 && NUM_CIRCUIT_QUBITS % 64) v &= (1ULL << (NUM_CIRCUIT_QUBITS % 64)) - 1;
      row[g_nw + w] = v;
    }
    if (oddFrame)
      for (int a = 0; a < NUM_CIRCUIT_QUBITS; a++) {
        if (!((row[g_nw + a / 64] >> (a % 64)) & 1ULL)) continue;
        if (g_nb[a].size() % 2) row[a / 64] ^= (1ULL << (a % 64));
        for (int32_t b : g_nb[a]) row[b / 64] ^= (1ULL << (b % 64));
      }
  }
  return rows;
}

struct Shared {
  std::mutex qm;
  std::condition_variable qcv;
  std::deque<Work> queue;
  int active = 0;
  std::vector<Bucket> nextB;
  std::vector<std::mutex> bm{256};
  std::mutex em;
  double expec = 0;
  std::atomic<long> nout{0}, splits{0}, runs{0}, peak{0};
};

struct Worker {
  int dev = 0;
  Opts o;
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  size_t xzPerTerm = 0, xzMem = 0, cfMem = 0, wsMem = 0;
  long capTerms = 0;
  void *xz[2], *cf[2], *dWs;
  uint8_t* dOwners;
  int32_t* dIdx;
  cupaulipropPauliExpansion_t ex[2];
  bool haveEx = false;
  int cur = 0;
  cupaulipropWorkspaceDescriptor_t ws;
  cupaulipropCoefficientTruncationParams_t coefParams;
  cupaulipropPauliWeightTruncationParams_t weightParams;
  std::vector<cupaulipropTruncationStrategy_t> trunc;
  std::vector<std::vector<Gate>> layers;
  std::mt19937_64 rng;
  std::vector<uint64_t> stageXz;
  std::vector<double> stageCf;
  double tGates = 0, tPart = 0, tH2D = 0, tD2H = 0, tAppend = 0, tIdle = 0;

  void init(int device, const Opts& opts) {
    dev = device; o = opts;
    CUDA(cudaSetDevice(dev));
    CUPP(cupaulipropCreate(&handle));
    xzPerTerm = g_words * sizeof(uint64_t);
    const size_t perTerm = xzPerTerm + sizeof(double);
    size_t freeMem = 0, totalMem = 0;
    CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t usable = (size_t)(freeMem * o.vramFrac);
    capTerms = (long)((size_t)(usable * o.expShare) / perTerm);
    xzMem = (size_t)capTerms * xzPerTerm; cfMem = (size_t)capTerms * sizeof(double);
    wsMem = usable - 2 * (xzMem + cfMem) - (size_t)capTerms * 11;
    for (int b = 0; b < 2; b++) { CUDA(cudaMalloc(&xz[b], xzMem)); CUDA(cudaMalloc(&cf[b], cfMem)); }
    CUDA(cudaMalloc(&dWs, wsMem));
    CUDA(cudaMalloc((void**)&dOwners, (size_t)capTerms));
    CUDA(cudaMalloc((void**)&dIdx, (size_t)capTerms * sizeof(int32_t)));
    CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));
    coefParams.cutoff = o.coef; weightParams.cutoff = o.weight;
    trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
    if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});
    rng.seed((uint64_t)o.seed * 7919ULL + dev);

    const double PI = 3.14159265358979323846;
    layers.resize(o.steps);
    for (int n = 0; n < o.steps; n++)
      for (int32_t q = 0; q < NUM_CIRCUIT_QUBITS; q++) {
        const int deg = (int)g_nb[q].size();
        double sign = ((deg % 2) && ((n / 2) % 2)) ? -1.0 : 1.0;
        std::vector<int32_t> qs; std::vector<cupaulipropPauliKind_t> ps;
        qs.push_back(q);
        if (n % 2 == 0) {
          ps.push_back(CUPAULIPROP_PAULI_X);
        } else {
          ps.push_back(deg % 2 ? CUPAULIPROP_PAULI_Y : CUPAULIPROP_PAULI_X);
          for (int32_t b : g_nb[q]) { qs.push_back(b); ps.push_back(CUPAULIPROP_PAULI_Z); }
          if (deg % 4 == 2 || deg % 4 == 3) sign *= -1.0;
        }
        Gate g;
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, sign * o.thetaOverPi * PI, (int32_t)qs.size(), qs.data(), ps.data(), &g.op));
        g.trunc = ((271 * n + q) % o.truncEvery == 0) || (q == 0 && n > 0);
        layers[n].push_back(g);
      }
  }

  void makeExpansions(long nCur) {
    if (haveEx) { CUPP(cupaulipropDestroyPauliExpansion(ex[0])); CUPP(cupaulipropDestroyPauliExpansion(ex[1])); }
    cur = 0;
    CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, xz[0], xzMem, cf[0], cfMem,
                                         CUDA_R_64F, nCur, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[0]));
    CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, xz[1], xzMem, cf[1], cfMem,
                                         CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[1]));
    haveEx = true;
  }
  long numTerms() { int64_t n = 0; CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &n)); return (long)n; }

  // Worker loop for layer n: take buckets from the pool until the pool is empty and nobody is working.
  void run(int n, Shared& sh, int gNext, const std::vector<uint64_t>& rowsNext) {
    CUDA(cudaSetDevice(dev));
    const int words = g_words;
    while (true) {
      Work w;
      {
        const double ti = now();
        std::unique_lock<std::mutex> lk(sh.qm);
        sh.qcv.wait(lk, [&] { return !sh.queue.empty() || sh.active == 0; });
        tIdle += now() - ti;
        if (sh.queue.empty()) { lk.unlock(); sh.qcv.notify_all(); return; }
        w.b.xz.swap(sh.queue.front().b.xz); w.b.cf.swap(sh.queue.front().b.cf); w.startQ = sh.queue.front().startQ;
        sh.queue.pop_front();
        sh.active++;
      }
      process(n, sh, w, gNext, rowsNext, words);
      { std::lock_guard<std::mutex> lk(sh.qm); sh.active--; }
      sh.qcv.notify_all();
    }
  }

  void process(int n, Shared& sh, Work& w, int gNext, const std::vector<uint64_t>& rowsNext, int words) {
    Bucket& b = w.b;
    const long nb0 = (long)b.cf.size();
    if (nb0 > capTerms) { fprintf(stderr, "bucket has %ld terms, GPU buffer holds %ld\n", nb0, capTerms); die("bucket too large to load", __LINE__); }
    sh.runs++;
    double t0 = now();
    CUDA(cudaMemcpy(xz[0], b.xz.data(), (size_t)nb0 * xzPerTerm, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(cf[0], b.cf.data(), (size_t)nb0 * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<uint64_t>().swap(b.xz); std::vector<double>().swap(b.cf);
    makeExpansions(nb0);
    tH2D += now() - t0;

    t0 = now();
    long peak = nb0;
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
        // Outgrew this GPU: split by one more row of this layer's family and requeue both halves.
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
          Work nwk; nwk.startQ = q;
          nwk.b.xz.resize((size_t)c2[h] * words); nwk.b.cf.resize((size_t)c2[h]);
          CUDA(cudaMemcpy(nwk.b.xz.data(), (char*)xz[1 - cur] + off2 * xzPerTerm, (size_t)c2[h] * xzPerTerm, cudaMemcpyDeviceToHost));
          CUDA(cudaMemcpy(nwk.b.cf.data(), (char*)cf[1 - cur] + off2 * sizeof(double), (size_t)c2[h] * sizeof(double), cudaMemcpyDeviceToHost));
          off2 += c2[h];
          { std::lock_guard<std::mutex> lk(sh.qm); sh.queue.push_front(Work()); sh.queue.front().b.xz.swap(nwk.b.xz); sh.queue.front().b.cf.swap(nwk.b.cf); sh.queue.front().startQ = q; }
          sh.qcv.notify_all();
        }
        sh.splits++;
        tGates += now() - t0;
        return;
      }
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, v, ex[1 - cur], g.op, 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, g.trunc ? trunc.data() : nullptr, ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      cur ^= 1;
    }
    long pk = sh.peak.load();
    while (peak > pk && !sh.peak.compare_exchange_weak(pk, peak)) {}
    const long m = numTerms();
    sh.nout += m;
    tGates += now() - t0;
    if (m == 0) return;

    if (n == 0) {                                     // last layer: trace with |0...0>
      cupaulipropPauliExpansionView_t v;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
      CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, wsMem, ws));
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      double sig = 0, e2 = 0;
      CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &e2, ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      std::lock_guard<std::mutex> lk(sh.em);
      sh.expec += sig * std::pow(2.0, e2);
      return;
    }

    // Partition the output by the next layer's rows and append the pieces to the next pool.
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
      std::lock_guard<std::mutex> lk(sh.bm[d]);
      sh.nextB[d].xz.insert(sh.nextB[d].xz.end(), stageXz.begin() + off * words, stageXz.begin() + (off + cnt[d]) * words);
      sh.nextB[d].cf.insert(sh.nextB[d].cf.end(), stageCf.begin() + off, stageCf.begin() + off + cnt[d]);
      off += cnt[d];
    }
    tAppend += now() - t0;
  }
};

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { if (i + 1 >= argc) die("missing option value", __LINE__); return argv[++i]; };
    if      (a == "--gpus")        o.gpus = atoi(next());
    else if (a == "--steps")       o.steps = atoi(next());
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
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }
  if (o.gmax > 8) die("--gmax is at most 8", __LINE__);
  int nd = 0;
  CUDA(cudaGetDeviceCount(&nd));
  if (o.gpus > nd) die("fewer GPUs visible than requested", __LINE__);

  int32_t nw = 0;
  CUPP(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &nw));
  g_nw = nw; g_words = 2 * nw;
  const int words = g_words;
  const int32_t (*zzLayers[3])[2] = {ZZ_QUBITS_RED, ZZ_QUBITS_BLUE, ZZ_QUBITS_GREEN};
  g_nb.assign(NUM_CIRCUIT_QUBITS, {});
  for (int l = 0; l < 3; l++)
    for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) {
      g_nb[zzLayers[l][i][0]].push_back(zzLayers[l][i][1]);
      g_nb[zzLayers[l][i][1]].push_back(zzLayers[l][i][0]);
    }

  std::vector<Worker> workers(o.gpus);
  for (int t = 0; t < o.gpus; t++) workers[t].init(t, o);
  printf("config: gpus=%d steps=%d theta=%.4fpi coef=%g weight=%d truncEvery=%d GPU capacity=%ld terms/buffer bucket target=%ld gmax=%d\n",
         o.gpus, o.steps, o.thetaOverPi, o.coef, o.weight, o.truncEvery, workers[0].capTerms, o.target, o.gmax);

  auto ownerCpu = [&](const uint64_t* t, const std::vector<uint64_t>& rows, int g) {
    int ow = 0;
    for (int j = 0; j < g; j++) {
      int par = 0;
      for (int w = 0; w < words; w++) par ^= __builtin_popcountll(t[w] & rows[(size_t)j * words + w]) & 1;
      ow |= par << j;
    }
    return ow;
  };

  std::vector<Bucket> curB(1);
  curB[0].xz.assign(words, 0);
  curB[0].xz[nw + o.obsQubit / 64] |= (1ULL << (o.obsQubit % 64));
  curB[0].cf.assign(1, 1.0);

  Shared sh;
  std::mt19937_64 rng(o.seed);
  double tSelect = 0, growth = 2.0;
  long maxGlobal = 0;
  const double tBegin = now();

  for (int n = o.steps - 1; n >= 0; n--) {
    long Nin = 0;
    for (auto& b : curB) Nin += (long)b.cf.size();
    maxGlobal = std::max(maxGlobal, Nin);

    int gNext = 0;
    std::vector<uint64_t> rowsNext(words, 0);
    if (n > 0) {
      const double est = Nin * growth;
      while (gNext < o.gmax && est / (double)(1L << gNext) > (double)o.target) gNext++;
      const double ts = now();
      double bestMax = 1e300;
      const long stride = std::max<long>(1, Nin / std::max<long>(1, o.sample));
      for (int c = 0; c < (gNext ? o.cands : 1); c++) {
        std::vector<uint64_t> rows = makeRows(rng, gNext, (n - 1) % 2 == 1);
        std::vector<long> h((size_t)1 << gNext, 0);
        for (auto& b : curB)
          for (size_t i = 0; i < b.cf.size(); i += stride) h[ownerCpu(&b.xz[i * words], rows, gNext)]++;
        const double mx = (double)*std::max_element(h.begin(), h.end());
        if (mx < bestMax) { bestMax = mx; rowsNext = rows; }
      }
      tSelect += now() - ts;
    }
    sh.nextB.assign((size_t)1 << gNext, Bucket());
    sh.nout = 0;
    // Largest buckets first, so the long tasks start early.
    std::sort(curB.begin(), curB.end(), [](const Bucket& a, const Bucket& b) { return a.cf.size() > b.cf.size(); });
    for (auto& b : curB)
      if (!b.cf.empty()) { sh.queue.push_back(Work()); sh.queue.back().b.xz.swap(b.xz); sh.queue.back().b.cf.swap(b.cf); sh.queue.back().startQ = NUM_CIRCUIT_QUBITS - 1; }
    sh.active = 0;

    std::vector<std::thread> th;
    for (int t = 0; t < o.gpus; t++) th.emplace_back([&, t] { workers[t].run(n, sh, gNext, rowsNext); });
    for (auto& x : th) x.join();

    const long Nout = sh.nout.load();
    if (Nin > 0 && Nout > 0) growth = std::max(1.0, std::min(8.0, (double)Nout / (double)Nin * 1.3));
    curB.swap(sh.nextB);
    maxGlobal = std::max(maxGlobal, Nout);
    printf("layer %2d: terms in=%ld out=%ld buckets next=%d elapsed=%.1f s\n", n, Nin, Nout, 1 << gNext, now() - tBegin);
    fflush(stdout);
  }
  const double wall = now() - tBegin;
  double tg = 0, tp = 0, th2 = 0, td2 = 0, ta = 0, ti = 0;
  for (auto& w : workers) { tg += w.tGates; tp += w.tPart; th2 += w.tH2D; td2 += w.tD2H; ta += w.tAppend; ti += w.tIdle; }
  printf("expectation=%.12g\n", sh.expec);
  printf("max_global_terms_at_layer_boundary=%ld largest_bucket_on_a_gpu=%ld terms bucket_runs=%ld mid_layer_splits=%ld\n",
         maxGlobal, sh.peak.load(), sh.runs.load(), sh.splits.load());
  printf("wall=%.3f s; summed over %d GPU workers: gates=%.3f partition=%.3f host->gpu=%.3f gpu->host=%.3f host append=%.3f idle=%.3f; row selection=%.3f\n",
         wall, o.gpus, tg, tp, th2, td2, ta, ti, tSelect);
  printf("MPI calls: 0, GPU-to-GPU transfers: 0\n");
  return 0;
}
