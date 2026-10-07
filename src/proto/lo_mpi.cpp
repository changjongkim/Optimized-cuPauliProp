// Parity-owned distributed Pauli propagation, one MPI rank per GPU.
//
// Every rank uses cuPauliProp in single-process mode on its own GPU. A term belongs to
// the rank given by g parities of its Z bits. X rotations never change Z bits, so a
// whole X layer runs with no MPI call at all and local deduplication and truncation
// are globally exact. The ZZ block changes Z bits, so terms are moved to their new
// owners once per Trotter step (a "re-key"), with the parity rows chosen from the
// current population. Gates never trigger communication; only re-keys do.

#include <cupauliprop.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../driver/ibm_eagle.hpp"

extern "C" {
void pp_owners(const uint64_t* xz, int64_t n, int words, const uint64_t* masks, int g, uint8_t* owners);
void pp_hist(const uint8_t* owners, int64_t n, int R, int64_t* hist);
void pp_owners_strided(const uint64_t* xz, int64_t n, int64_t stride, int words, const uint64_t* masks, int g, uint8_t* owners);
void pp_counts_sorted(const uint8_t* owners, int64_t n, int R, int64_t* cnt);
int pp_partition(const uint64_t* xz, const double* coef, uint8_t* owners, int64_t n, int words,
                 uint64_t* xzOut, double* coefOut, int32_t* idx);
}

static int g_rank = 0;
[[noreturn]] static void die(const char* what, int line) {
  fprintf(stderr, "[rank %d] FATAL: %s (line %d)\n", g_rank, what, line);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::exit(1);
}
#define CUPP(x) do { cupaulipropStatus_t s_ = (x); if (s_ != CUPAULIPROP_STATUS_SUCCESS) { \
  fprintf(stderr, "[rank %d] cuPauliProp status %d\n", g_rank, (int)s_); die(#x, __LINE__); } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "[rank %d] CUDA: %s\n", g_rank, cudaGetErrorString(e_)); die(#x, __LINE__); } } while (0)

struct Opts {
  int steps = 20;
  double thetaOverPi = 0.25;
  double coef = 1e-4;
  int weight = 0;
  int truncEvery = 10;
  int obsQubit = 62;
  double vramFrac = 0.85;     // share of free GPU memory this rank may use
  double expShare = 0.2;      // share of that memory per expansion buffer
  long nmin = 100000;         // below this many global terms everything stays on rank 0
  int cands = 8;              // candidate row sets evaluated at each re-key
  int seed = 7;
  double junk = 1e-13;        // coefficients at or below this are dropped before a re-key
  int frame = 0;              // 1: Clifford frame (ZZ layers absorbed into the rotation generators)
  long sample = 200000;       // terms per rank used to score candidate rows
  std::string tag = "lo";
  std::string outdir = ".";
};

struct Gate { cupaulipropQuantumOperator_t op; int kind; bool trunc; int layer; };

int main(int argc, char** argv) {
  // Select this rank's GPU before MPI_Init so Cray MPICH's GPU transport binds to it.
  {
    const char* lid = getenv("SLURM_LOCALID");
    int nd = 0;
    if (lid && cudaGetDeviceCount(&nd) == cudaSuccess && nd > 0) {
      cudaSetDevice(atoi(lid) % nd);
      void* warm = nullptr;
      if (cudaMalloc(&warm, 1 << 20) == cudaSuccess) cudaFree(warm);
    }
  }
  MPI_Init(&argc, &argv);
  int R = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &R);
  const int rank = g_rank;

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
    else if (a == "--nmin")        o.nmin = atol(next());
    else if (a == "--cands")       o.cands = atoi(next());
    else if (a == "--seed")        o.seed = atoi(next());
    else if (a == "--frame")       o.frame = atoi(next());
    else if (a == "--sample")      o.sample = atol(next());
    else if (a == "--tag")         o.tag = next();
    else if (a == "--outdir")      o.outdir = next();
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }
  int gFull = 0;
  while ((1 << gFull) < R) gFull++;
  if ((1 << gFull) != R || gFull > 8) die("rank count must be a power of two up to 256", __LINE__);

  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  CUPP(cupaulipropCreate(&handle));

  int32_t nw = 0;
  CUPP(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &nw));
  const int words = 2 * nw;
  const size_t xzPerTerm = words * sizeof(uint64_t);
  const size_t perTerm = xzPerTerm + sizeof(double);
  size_t freeMem = 0, totalMem = 0;
  CUDA(cudaMemGetInfo(&freeMem, &totalMem));
  const size_t usable = (size_t)(freeMem * o.vramFrac);
  const long capTerms = (long)((size_t)(usable * o.expShare) / perTerm);
  const size_t xzMem = (size_t)capTerms * xzPerTerm, cfMem = (size_t)capTerms * sizeof(double);
  const size_t scratchMem = (size_t)capTerms * 5;            // owner ids + permutation
  const size_t sortSlack = (size_t)capTerms * 6;             // temporary storage of the radix sort
  const size_t wsMem = usable - 2 * (xzMem + cfMem) - scratchMem - sortSlack;

  void *xz[2], *cf[2], *dWs;
  for (int b = 0; b < 2; b++) { CUDA(cudaMalloc(&xz[b], xzMem)); CUDA(cudaMalloc(&cf[b], cfMem)); }
  CUDA(cudaMalloc(&dWs, wsMem));
  uint8_t* dOwners; CUDA(cudaMalloc((void**)&dOwners, (size_t)capTerms));
  int32_t* dIdx; CUDA(cudaMalloc((void**)&dIdx, (size_t)capTerms * sizeof(int32_t)));
  int cur = 0;
  cupaulipropPauliExpansion_t ex[2];

  auto makeExpansions = [&](long nCur) {
    CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, xz[cur], xzMem, cf[cur], cfMem,
                                         CUDA_R_64F, nCur, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[cur]));
    CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, xz[1 - cur], xzMem, cf[1 - cur], cfMem,
                                         CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[1 - cur]));
  };
  if (rank == 0) {
    std::vector<uint64_t> bits(words, 0);
    bits[nw + o.obsQubit / 64] |= (1ULL << (o.obsQubit % 64));
    double one = 1.0;
    CUDA(cudaMemcpy(xz[0], bits.data(), xzPerTerm, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(cf[0], &one, sizeof(one), cudaMemcpyHostToDevice));
  }
  makeExpansions(rank == 0 ? 1 : 0);

  cupaulipropWorkspaceDescriptor_t ws;
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));

  cupaulipropCoefficientTruncationParams_t coefParams; coefParams.cutoff = o.coef;
  cupaulipropPauliWeightTruncationParams_t weightParams; weightParams.cutoff = o.weight;
  std::vector<cupaulipropTruncationStrategy_t> trunc;
  trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
  if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});
  cupaulipropCoefficientTruncationParams_t junkParams; junkParams.cutoff = o.junk;
  cupaulipropTruncationStrategy_t junkTrunc[1] = {{CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &junkParams}};

  const double PI = 3.14159265358979323846;
  std::vector<Gate> circuit;
  const int32_t (*zzLayers[3])[2] = {ZZ_QUBITS_RED, ZZ_QUBITS_BLUE, ZZ_QUBITS_GREEN};
  std::vector<std::vector<int32_t>> nb(NUM_CIRCUIT_QUBITS);
  for (int l = 0; l < 3; l++)
    for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) {
      nb[zzLayers[l][i][0]].push_back(zzLayers[l][i][1]);
      nb[zzLayers[l][i][1]].push_back(zzLayers[l][i][0]);
    }
  if (!o.frame) {
    for (int n = 0; n < o.steps; n++) {
      for (int32_t q = 0; q < NUM_CIRCUIT_QUBITS; q++) {
        cupaulipropPauliKind_t p[1] = {CUPAULIPROP_PAULI_X};
        Gate g; g.kind = 0; g.layer = n;
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, o.thetaOverPi * PI, 1, &q, p, &g.op));
        circuit.push_back(g);
      }
      for (int l = 0; l < 3; l++)
        for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) {
          cupaulipropPauliKind_t p[2] = {CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z};
          Gate g; g.kind = 1; g.layer = n;
          CUPP(cupaulipropCreatePauliRotationGateOperator(handle, -PI / 2.0, 2, zzLayers[l][i], p, &g.op));
          circuit.push_back(g);
        }
    }
    for (size_t i = 0; i < circuit.size(); i++) circuit[i].trunc = (i % o.truncEvery == 0);
  } else {
    // Clifford frame: the circuit becomes steps x 127 rotations about X_q (even layers) or about
    // X_q/Y_q times Z on the neighbours (odd layers); no ZZ gate touches the data. See ki_prof.cpp.
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
        Gate g; g.kind = 0; g.layer = n;
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, sign * o.thetaOverPi * PI, (int32_t)qs.size(), qs.data(), ps.data(), &g.op));
        g.trunc = ((271 * n + q) % o.truncEvery == 0) || (q == 0 && n > 0);
        circuit.push_back(g);
      }
    }
  }

  auto numTerms = [&]() { int64_t n = 0; CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &n)); return (long)n; };
  auto setWs = [&]() {
    int64_t need = 0;
    CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
    if (need > (int64_t)wsMem) { fprintf(stderr, "[rank %d] workspace exceeded: need %ld have %zu\n", rank, (long)need, wsMem); die("workspace", __LINE__); }
    CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
  };

  auto applyGate = [&](const Gate& g, int32_t nT, const cupaulipropTruncationStrategy_t* pT) {
    const long n = numTerms();
    if (n == 0) return;
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, n, &v));
    int64_t rp = 0, rc = 0;
    CUPP(cupaulipropPauliExpansionViewPrepareOperatorApplication(
      handle, v, g.op, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, wsMem, &rp, &rc, ws));
    if (rp > (int64_t)xzMem || rc > (int64_t)cfMem) {
      fprintf(stderr, "[rank %d] capacity exceeded: local terms=%ld capacity=%ld\n", rank, n, capTerms);
      die("capacity", __LINE__);
    }
    setWs();
    CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
      handle, v, ex[1 - cur], g.op, 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, ws, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    cur ^= 1;
  };

  auto dropJunk = [&]() {
    const long n = numTerms();
    if (n == 0) return;
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, n, &v));
    CUPP(cupaulipropPauliExpansionViewPrepareTruncation(handle, v, 1, junkTrunc, wsMem, ws));
    setWs();
    CUPP(cupaulipropPauliExpansionViewExecuteTruncation(handle, v, ex[1 - cur], 1, junkTrunc, ws, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    cur ^= 1;
  };

  double tGates = 0, tRekey = 0, tSelect = 0, tXfer = 0;
  long movedOut = 0, rekeys = 0, rekeysSkipped = 0, maxLocal = 0, collectives = 0, p2pMsgs = 0;
  double worstImbalance = 1.0;
  long rekeyIndex = 0;
  std::vector<std::string> rekeyLog;

  // Moves every term to the rank given by g parity rows. A row is a random set m of Z bits;
  // with oddFrame it also carries the X bits L*m (L = lattice Laplacian mod 2), which makes it
  // invariant under the odd-layer generators of the Clifford frame.
  auto rekey = [&](int gi, bool oddFrame) {
    const double t0 = MPI_Wtime();
    if (!o.frame) dropJunk();
    const long n = numTerms();
    long loc[2] = {n, rank == 0 ? n : 0}, glob[2] = {0, 0};
    MPI_Allreduce(loc, glob, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); collectives++;
    const long N = glob[0];
    const int g = (N >= o.nmin) ? gFull : 0;
    rekeyIndex++;
    if (g == 0 && N == glob[1]) { rekeysSkipped++; tRekey += MPI_Wtime() - t0; return; }
    const int Rg = 1 << g;

    // Row selection: every rank scores the same candidate row sets on its own terms.
    const double ts0 = MPI_Wtime();
    const int C = g ? o.cands : 1;
    std::vector<std::vector<uint64_t>> cand(C, std::vector<uint64_t>(std::max(g, 1) * words, 0));
    std::vector<int64_t> hLoc((size_t)C * Rg, 0), hGlob((size_t)C * Rg, 0);
    const long stride = std::max<long>(1, n / std::max<long>(1, o.sample));
    const long ns = n > 0 ? (n + stride - 1) / stride : 0;
    std::vector<uint8_t> hOwn((size_t)ns);
    for (int c = 0; c < C; c++) {
      std::mt19937_64 rng((uint64_t)o.seed * 1000003ULL + (uint64_t)rekeyIndex * 9176ULL + c);
      for (int j = 0; j < g; j++) {
        uint64_t* row = &cand[c][(size_t)j * words];
        for (int w = 0; w < nw; w++) {
          uint64_t v = rng();
          if (w == nw - 1 && NUM_CIRCUIT_QUBITS % 64) v &= (1ULL << (NUM_CIRCUIT_QUBITS % 64)) - 1;
          row[nw + w] = v;
        }
        if (oddFrame) {
          for (int a2 = 0; a2 < NUM_CIRCUIT_QUBITS; a2++) {
            if (!((row[nw + a2 / 64] >> (a2 % 64)) & 1ULL)) continue;
            if (nb[a2].size() % 2) row[a2 / 64] ^= (1ULL << (a2 % 64));
            for (int32_t b2 : nb[a2]) row[b2 / 64] ^= (1ULL << (b2 % 64));
          }
        }
      }
      if (ns > 0 && C > 1) {
        pp_owners_strided((const uint64_t*)xz[cur], n, stride, words, cand[c].data(), g, dOwners);
        CUDA(cudaMemcpy(hOwn.data(), dOwners, (size_t)ns, cudaMemcpyDeviceToHost));
        for (long i2 = 0; i2 < ns; i2++) hLoc[(size_t)c * Rg + hOwn[i2]] += stride;
      }
    }
    int bestC = 0;
    if (C > 1) {
      MPI_Allreduce(hLoc.data(), hGlob.data(), C * Rg, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD); collectives++;
      int64_t bestMax = INT64_MAX;
      for (int c = 0; c < C; c++) {
        const int64_t mx = *std::max_element(hGlob.begin() + (size_t)c * Rg, hGlob.begin() + (size_t)(c + 1) * Rg);
        if (mx < bestMax) { bestMax = mx; bestC = c; }
      }
    }
    const std::vector<uint64_t>& rows = cand[bestC];
    tSelect += MPI_Wtime() - ts0;

    // Partition local terms by destination into the spare buffer.
    std::vector<int64_t> cnt(R, 0), rcv(R, 0);
    if (n > 0) {
      std::vector<int64_t> h(Rg, 0);
      pp_owners((const uint64_t*)xz[cur], n, words, rows.data(), g, dOwners);
      if (pp_partition((const uint64_t*)xz[cur], (const double*)cf[cur], dOwners, n, words,
                       (uint64_t*)xz[1 - cur], (double*)cf[1 - cur], dIdx) != 0) die("partition", __LINE__);
      pp_counts_sorted(dOwners, n, Rg, h.data());
      for (int d = 0; d < Rg; d++) cnt[d] = h[d];
      CUDA(cudaDeviceSynchronize());
    }
    MPI_Alltoall(cnt.data(), 1, MPI_INT64_T, rcv.data(), 1, MPI_INT64_T, MPI_COMM_WORLD); collectives++;

    // Exchange: receive into the current buffer, send from the partitioned spare buffer.
    const double tx0 = MPI_Wtime();
    long nNew = 0;
    for (int s = 0; s < R; s++) nNew += rcv[s];
    if (nNew > capTerms) { fprintf(stderr, "[rank %d] re-key would deliver %ld terms, capacity %ld\n", rank, nNew, capTerms); die("re-key capacity", __LINE__); }
    std::vector<MPI_Request> req;
    long off = 0;
    std::vector<long> roff(R, 0);
    for (int s = 0; s < R; s++) { roff[s] = off; off += rcv[s]; }
    for (int s = 0; s < R; s++) {
      if (s == rank || rcv[s] == 0) continue;
      req.emplace_back(); req.emplace_back();
      MPI_Irecv((char*)xz[cur] + roff[s] * xzPerTerm, (int)(rcv[s] * words), MPI_UINT64_T, s, 100, MPI_COMM_WORLD, &req[req.size() - 2]);
      MPI_Irecv((char*)cf[cur] + roff[s] * sizeof(double), (int)rcv[s], MPI_DOUBLE, s, 101, MPI_COMM_WORLD, &req[req.size() - 1]);
    }
    long soff = 0;
    for (int d = 0; d < R; d++) {
      if (cnt[d] > 0) {
        if (d == rank) {
          CUDA(cudaMemcpy((char*)xz[cur] + roff[d] * xzPerTerm, (char*)xz[1 - cur] + soff * xzPerTerm, cnt[d] * xzPerTerm, cudaMemcpyDeviceToDevice));
          CUDA(cudaMemcpy((char*)cf[cur] + roff[d] * sizeof(double), (char*)cf[1 - cur] + soff * sizeof(double), cnt[d] * sizeof(double), cudaMemcpyDeviceToDevice));
        } else {
          req.emplace_back(); req.emplace_back();
          MPI_Isend((char*)xz[1 - cur] + soff * xzPerTerm, (int)(cnt[d] * words), MPI_UINT64_T, d, 100, MPI_COMM_WORLD, &req[req.size() - 2]);
          MPI_Isend((char*)cf[1 - cur] + soff * sizeof(double), (int)cnt[d], MPI_DOUBLE, d, 101, MPI_COMM_WORLD, &req[req.size() - 1]);
          movedOut += cnt[d]; p2pMsgs += 2;
        }
      }
      soff += cnt[d];
    }
    if (!req.empty()) MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
    CUDA(cudaDeviceSynchronize());
    tXfer += MPI_Wtime() - tx0;

    CUPP(cupaulipropDestroyPauliExpansion(ex[0]));
    CUPP(cupaulipropDestroyPauliExpansion(ex[1]));
    makeExpansions(nNew);
    if (g > 0) worstImbalance = std::max(worstImbalance, (double)nNew / ((double)N / Rg));
    rekeys++;
    const double dt = MPI_Wtime() - t0;
    tRekey += dt;
    if (rank == 0) {
      char line[256];
      snprintf(line, sizeof(line), "%d,%ld,%d,%ld,%.6f", gi, N, g, nNew, dt);
      rekeyLog.push_back(line);
    }
  };

  if (rank == 0) {
    printf("config: ranks=%d steps=%d theta=%.4fpi coef=%g weight=%d truncEvery=%d capTerms/rank=%ld workspace=%.1f GiB nmin=%ld cands=%d\n",
           R, o.steps, o.thetaOverPi, o.coef, o.weight, o.truncEvery, capTerms, wsMem / 1073741824.0, o.nmin, o.cands);
    fflush(stdout);
  }
  CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  const double tBegin = MPI_Wtime();
  int prevKind = -1, prevLayer = -1;

  for (int gi = (int)circuit.size() - 1; gi >= 0; --gi) {
    const Gate& g = circuit[gi];
    // Lab frame: entering an X layer after a ZZ block changed the Z bits, so re-key first.
    // Clifford frame: every layer has its own row family, so re-key at each layer start.
    if (!o.frame) { if (g.kind == 0 && prevKind != 0) rekey(gi, false); }
    else if (g.layer != prevLayer) rekey(gi, g.layer % 2 == 1);
    prevKind = g.kind; prevLayer = g.layer;
    const bool doTrunc = g.trunc;
    const double t0 = MPI_Wtime();
    applyGate(g, doTrunc ? (int32_t)trunc.size() : 0, doTrunc ? trunc.data() : nullptr);
    tGates += MPI_Wtime() - t0;
    maxLocal = std::max(maxLocal, numTerms());
  }

  // Trace with |0...0>: local traces are summed.
  double expLoc = 0;
  const long nFin = numTerms();
  if (nFin > 0) {
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, nFin, &v));
    CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, wsMem, ws));
    setWs();
    double sig = 0, e2 = 0;
    CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &e2, ws, stream));
    CUDA(cudaStreamSynchronize(stream));
    expLoc = sig * std::pow(2.0, e2);
    CUPP(cupaulipropDestroyPauliExpansionView(v));
  }
  double expec = 0;
  MPI_Allreduce(&expLoc, &expec, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); collectives++;
  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = MPI_Wtime() - tBegin;

  long li[4] = {nFin, movedOut, maxLocal, p2pMsgs}, gs[4], gm[4];
  MPI_Allreduce(li, gs, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(li, gm, 4, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  double ld[5] = {tGates, tRekey, tSelect, tXfer, worstImbalance}, gds[5], gdm[5];
  MPI_Allreduce(ld, gds, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(ld, gdm, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("expectation=%.12g\n", expec);
    printf("final_global_terms=%ld max_local_terms=%ld wall=%.4f s\n", gs[0], gm[2], wall);
    printf("re-keys=%ld (skipped while on one rank: %ld) moved_terms=%ld (%.3e bytes) worst max/mean bucket=%.3f\n",
           rekeys, rekeysSkipped, gs[1], gs[1] * (double)perTerm, gdm[4]);
    printf("collective MPI calls per rank=%ld, point-to-point messages sent (all ranks)=%ld, MPI calls during gates=0\n",
           collectives, gs[3]);
    printf("time per rank: gates mean=%.4f max=%.4f s, re-key mean=%.4f max=%.4f s (row selection mean=%.4f, exchange mean=%.4f)\n",
           gds[0] / R, gdm[0], gds[1] / R, gdm[1], gds[2] / R, gds[3] / R);
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.rekey.csv", o.outdir.c_str(), o.tag.c_str());
    FILE* f = fopen(path, "w");
    if (f) {
      fprintf(f, "gate,n_global,g_rows,n_rank0_after,t_rank0\n");
      for (auto& l : rekeyLog) fprintf(f, "%s\n", l.c_str());
      fclose(f);
    }
    fflush(stdout);
  }
  MPI_Finalize();
  return 0;
}
