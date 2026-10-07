// Emulation of parity-owned, barrier-free distributed Pauli propagation.
//
// R "virtual ranks" live in one process on one GPU; each owns a private pair of
// expansion buffers and uses cuPauliProp in single-process mode. A term belongs to
// the rank given by g parities of its Z bits. X rotations never change Z bits, so a
// whole X layer runs with no exchange between ranks and local deduplication and
// truncation are globally exact. The ZZ block changes Z bits, so terms are moved to
// their new owners once per Trotter step (a "re-key"), with the parity rows chosen
// from the current population. The emulation checks that the result equals the
// single-GPU result and measures how much data a real multi-GPU run would move.

#include <cupauliprop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
  int ranks = 4;
  int steps = 20;
  double thetaOverPi = 0.25;
  double zzOverPi = -0.5;     // ZZ rotation angle in units of pi; -0.5 is a Clifford angle
  double coef = 1e-4;
  int weight = 0;
  int truncEvery = 10;
  int obsQubit = 62;
  long capTerms = 20000000;   // per rank, per buffer
  double wsGiB = 14.0;
  long nmin = 100000;         // below this many global terms everything stays on rank 0
  int cands = 8;              // candidate row sets evaluated at each re-key
  int seed = 7;
  double junk = 1e-13;        // coefficients at or below this are dropped before a re-key
  long rekeyLimit = -1;       // >=0: after this many re-keys, ranks never exchange terms again
  std::string tag = "lo";
  std::string outdir = ".";
};

struct Gate { cupaulipropQuantumOperator_t op; int kind; };

struct Rank {
  void* xz[2];
  void* cf[2];
  cupaulipropPauliExpansion_t ex[2];
  int cur;
};

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { if (i + 1 >= argc) die("missing option value", __LINE__); return argv[++i]; };
    if      (a == "--ranks")       o.ranks = atoi(next());
    else if (a == "--steps")       o.steps = atoi(next());
    else if (a == "--theta")       o.thetaOverPi = atof(next());
    else if (a == "--zz-theta")    o.zzOverPi = atof(next());
    else if (a == "--coef")        o.coef = atof(next());
    else if (a == "--weight")      o.weight = atoi(next());
    else if (a == "--trunc-every") o.truncEvery = atoi(next());
    else if (a == "--obs-qubit")   o.obsQubit = atoi(next());
    else if (a == "--cap-terms")   o.capTerms = atol(next());
    else if (a == "--ws-gib")      o.wsGiB = atof(next());
    else if (a == "--nmin")        o.nmin = atol(next());
    else if (a == "--cands")       o.cands = atoi(next());
    else if (a == "--seed")        o.seed = atoi(next());
    else if (a == "--rekey-limit") o.rekeyLimit = atol(next());
    else if (a == "--tag")         o.tag = next();
    else if (a == "--outdir")      o.outdir = next();
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }
  const int R = o.ranks;
  int gFull = 0;
  while ((1 << gFull) < R) gFull++;
  if ((1 << gFull) != R || gFull > 8) die("--ranks must be a power of two up to 256", __LINE__);

  CUDA(cudaSetDevice(0));
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  CUPP(cupaulipropCreate(&handle));

  int32_t nw = 0;
  CUPP(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &nw));
  const int words = 2 * nw;
  const size_t xzPerTerm = words * sizeof(uint64_t);
  const size_t xzMem = (size_t)o.capTerms * xzPerTerm, cfMem = (size_t)o.capTerms * sizeof(double);
  const size_t wsMem = (size_t)(o.wsGiB * 1073741824.0);

  std::vector<Rank> rk(R);
  for (int r = 0; r < R; r++) {
    for (int b = 0; b < 2; b++) { CUDA(cudaMalloc(&rk[r].xz[b], xzMem)); CUDA(cudaMalloc(&rk[r].cf[b], cfMem)); }
    rk[r].cur = 0;
  }
  void* dWs; CUDA(cudaMalloc(&dWs, wsMem));
  uint8_t* dOwners; CUDA(cudaMalloc((void**)&dOwners, (size_t)o.capTerms));
  int32_t* dIdx; CUDA(cudaMalloc((void**)&dIdx, (size_t)o.capTerms * sizeof(int32_t)));

  auto makeExpansions = [&](int r, long nCur) {
    CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, rk[r].xz[rk[r].cur], xzMem, rk[r].cf[rk[r].cur], cfMem,
                                         CUDA_R_64F, nCur, CUPAULIPROP_SORT_ORDER_NONE, 0, &rk[r].ex[rk[r].cur]));
    CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, rk[r].xz[1 - rk[r].cur], xzMem, rk[r].cf[1 - rk[r].cur], cfMem,
                                         CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &rk[r].ex[1 - rk[r].cur]));
  };
  {
    std::vector<uint64_t> bits(words, 0);
    bits[nw + o.obsQubit / 64] |= (1ULL << (o.obsQubit % 64));
    double one = 1.0;
    CUDA(cudaMemcpy(rk[0].xz[0], bits.data(), xzPerTerm, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(rk[0].cf[0], &one, sizeof(one), cudaMemcpyHostToDevice));
  }
  for (int r = 0; r < R; r++) makeExpansions(r, r == 0 ? 1 : 0);

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
  for (int n = 0; n < o.steps; n++) {
    for (int32_t q = 0; q < NUM_CIRCUIT_QUBITS; q++) {
      cupaulipropPauliKind_t p[1] = {CUPAULIPROP_PAULI_X};
      Gate g; g.kind = 0;
      CUPP(cupaulipropCreatePauliRotationGateOperator(handle, o.thetaOverPi * PI, 1, &q, p, &g.op));
      circuit.push_back(g);
    }
    const int32_t (*layers[3])[2] = {ZZ_QUBITS_RED, ZZ_QUBITS_BLUE, ZZ_QUBITS_GREEN};
    for (int l = 0; l < 3; l++)
      for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) {
        cupaulipropPauliKind_t p[2] = {CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z};
        Gate g; g.kind = 1;
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, o.zzOverPi * PI, 2, layers[l][i], p, &g.op));
        circuit.push_back(g);
      }
  }

  auto numTerms = [&](int r) { int64_t n = 0; CUPP(cupaulipropPauliExpansionGetNumTerms(handle, rk[r].ex[rk[r].cur], &n)); return (long)n; };
  auto setWs = [&]() {
    int64_t need = 0;
    CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
    if (need > (int64_t)wsMem) { fprintf(stderr, "workspace exceeded: need %ld have %zu\n", (long)need, wsMem); die("workspace", __LINE__); }
    CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
  };

  // Applies one gate on one rank with local deduplication (and truncation if given).
  auto applyGate = [&](int r, const Gate& g, int32_t nT, const cupaulipropTruncationStrategy_t* pT) {
    const long n = numTerms(r);
    if (n == 0) return 0.0;
    const double t0 = now();
    Rank& k = rk[r];
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, k.ex[k.cur], 0, n, &v));
    int64_t rp = 0, rc = 0;
    CUPP(cupaulipropPauliExpansionViewPrepareOperatorApplication(
      handle, v, g.op, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, wsMem, &rp, &rc, ws));
    if (rp > (int64_t)xzMem || rc > (int64_t)cfMem) { fprintf(stderr, "rank %d capacity exceeded: n=%ld\n", r, n); die("capacity", __LINE__); }
    setWs();
    CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
      handle, v, k.ex[1 - k.cur], g.op, 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, ws, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    k.cur ^= 1;
    return now() - t0;
  };

  auto dropJunk = [&](int r) {
    const long n = numTerms(r);
    if (n == 0) return;
    Rank& k = rk[r];
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, k.ex[k.cur], 0, n, &v));
    CUPP(cupaulipropPauliExpansionViewPrepareTruncation(handle, v, 1, junkTrunc, wsMem, ws));
    setWs();
    CUPP(cupaulipropPauliExpansionViewExecuteTruncation(handle, v, k.ex[1 - k.cur], 1, junkTrunc, ws, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    k.cur ^= 1;
  };

  std::mt19937_64 rng(o.seed);
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s.rekey.csv", o.outdir.c_str(), o.tag.c_str());
  FILE* fre = fopen(path, "w");
  snprintf(path, sizeof(path), "%s/%s.gates.csv", o.outdir.c_str(), o.tag.c_str());
  FILE* fg = fopen(path, "w");
  if (!fre || !fg) die("cannot open csv", __LINE__);
  fprintf(fre, "gate,n_global,g_rows,moved_terms,max_bucket,mean_bucket,t_prep_max,t_prep_sum,t_select,t_copy\n");
  fprintf(fg, "gate,kind,n_global,active_ranks,t_max,t_sum\n");

  double tGatesPar = 0, tGatesSum = 0, tRekeyPar = 0, tRekeySum = 0, tSelect = 0;
  long movedTotal = 0, rekeys = 0, rekeysSkipped = 0, maxLocal = 0, maxGlobal = 0;
  double worstImbalance = 1.0;

  // Moves every term to the rank given by g parities of its Z bits (or X bits if useX).
  // Z-bit rows are unchanged by X rotations; X-bit rows are unchanged by ZZ rotations.
  auto rekey = [&](int gi, bool useX) {
    std::vector<long> n(R);
    long N = 0;
    for (int r = 0; r < R; r++) { dropJunk(r); n[r] = numTerms(r); N += n[r]; }
    const int g = (N >= o.nmin) ? gFull : 0;
    if (g == 0 && N == n[0]) { rekeysSkipped++; return; }   // everything already on rank 0
    if (o.rekeyLimit >= 0 && rekeys >= o.rekeyLimit) { rekeysSkipped++; return; }   // ranks stay independent
    const int Rg = 1 << g;

    // Candidate row sets: each row is a random subset of the Z bits. Keep the best balanced.
    const double ts0 = now();
    std::vector<uint64_t> best(std::max(g, 1) * words, 0);
    double bestMax = 1e300;
    for (int c = 0; c < (g ? o.cands : 1); c++) {
      std::vector<uint64_t> m(std::max(g, 1) * words, 0);
      for (int j = 0; j < g; j++)
        for (int w = 0; w < nw; w++) {
          uint64_t v = rng();
          if (w == nw - 1 && NUM_CIRCUIT_QUBITS % 64) v &= (1ULL << (NUM_CIRCUIT_QUBITS % 64)) - 1;
          m[j * words + (useX ? 0 : nw) + w] = v;
        }
      std::vector<int64_t> tot(Rg, 0), h(Rg);
      for (int r = 0; r < R; r++) {
        if (n[r] == 0) continue;
        pp_owners((const uint64_t*)rk[r].xz[rk[r].cur], n[r], words, m.data(), g, dOwners);
        pp_hist(dOwners, n[r], Rg, h.data());
        for (int d = 0; d < Rg; d++) tot[d] += h[d];
      }
      const double mx = (double)*std::max_element(tot.begin(), tot.end());
      if (mx < bestMax) { bestMax = mx; best = m; }
    }
    CUDA(cudaDeviceSynchronize());
    const double tSel = now() - ts0;

    // Partition each rank's terms by destination into its spare buffer.
    std::vector<std::vector<int64_t>> cnt(R, std::vector<int64_t>(Rg, 0));
    double prepMax = 0, prepSum = 0;
    for (int r = 0; r < R; r++) {
      if (n[r] == 0) continue;
      const double t0 = now();
      Rank& k = rk[r];
      pp_owners((const uint64_t*)k.xz[k.cur], n[r], words, best.data(), g, dOwners);
      pp_hist(dOwners, n[r], Rg, cnt[r].data());
      if (pp_partition((const uint64_t*)k.xz[k.cur], (const double*)k.cf[k.cur], dOwners, n[r], words,
                       (uint64_t*)k.xz[1 - k.cur], (double*)k.cf[1 - k.cur], dIdx) != 0) die("partition", __LINE__);
      CUDA(cudaDeviceSynchronize());
      const double dt = now() - t0;
      prepMax = std::max(prepMax, dt); prepSum += dt;
    }

    // Deliver: destination d receives its segment from every source into its current buffer.
    const double tc0 = now();
    long moved = 0;
    std::vector<long> nNew(R, 0);
    for (int d = 0; d < R; d++) {
      long off = 0;
      if (d < Rg) {
        for (int r = 0; r < R; r++) {
          if (cnt[r][d] == 0) continue;
          long src = 0;
          for (int e = 0; e < d; e++) src += cnt[r][e];
          if (off + cnt[r][d] > o.capTerms) die("re-key overflows destination capacity", __LINE__);
          CUDA(cudaMemcpy((char*)rk[d].xz[rk[d].cur] + off * xzPerTerm, (char*)rk[r].xz[1 - rk[r].cur] + src * xzPerTerm,
                          cnt[r][d] * xzPerTerm, cudaMemcpyDeviceToDevice));
          CUDA(cudaMemcpy((char*)rk[d].cf[rk[d].cur] + off * sizeof(double), (char*)rk[r].cf[1 - rk[r].cur] + src * sizeof(double),
                          cnt[r][d] * sizeof(double), cudaMemcpyDeviceToDevice));
          off += cnt[r][d];
          if (r != d) moved += cnt[r][d];
        }
      }
      nNew[d] = off;
    }
    CUDA(cudaDeviceSynchronize());
    const double tCopy = now() - tc0;
    for (int r = 0; r < R; r++) {
      CUPP(cupaulipropDestroyPauliExpansion(rk[r].ex[0]));
      CUPP(cupaulipropDestroyPauliExpansion(rk[r].ex[1]));
      makeExpansions(r, nNew[r]);
    }
    const long mxb = *std::max_element(nNew.begin(), nNew.end());
    const double mean = (double)N / Rg;
    if (g > 0) worstImbalance = std::max(worstImbalance, mxb / mean);
    fprintf(fre, "%d,%ld,%d,%ld,%ld,%.1f,%.6f,%.6f,%.6f,%.6f\n", gi, N, g, moved, mxb, mean, prepMax, prepSum, tSel, tCopy);
    movedTotal += moved; rekeys++;
    tRekeyPar += prepMax + tCopy / std::max(1, Rg); tRekeySum += prepSum + tCopy; tSelect += tSel;
  };

  printf("config: ranks=%d steps=%d theta=%.4fpi coef=%g weight=%d truncEvery=%d capTerms/rank=%ld nmin=%ld cands=%d\n",
         R, o.steps, o.thetaOverPi, o.coef, o.weight, o.truncEvery, o.capTerms, o.nmin, o.cands);
  CUDA(cudaStreamSynchronize(stream));
  const double tBegin = now();
  int prevKind = -1;
  const bool zzBranches = std::fabs(std::fabs(o.zzOverPi) - 0.5) > 1e-12;

  for (int gi = (int)circuit.size() - 1; gi >= 0; --gi) {
    const Gate& g = circuit[gi];
    // Entering an X layer: Z bits were changed by the preceding ZZ block, so re-key first.
    // A ZZ block with a non-Clifford angle branches on Z bits, so it runs under X-bit rows.
    if (g.kind == 0 && prevKind != 0) rekey(gi, false);
    if (g.kind == 1 && prevKind != 1 && zzBranches) rekey(gi, true);
    prevKind = g.kind;

    const bool doTrunc = (gi % o.truncEvery == 0);
    const int32_t nT = doTrunc ? (int32_t)trunc.size() : 0;
    const cupaulipropTruncationStrategy_t* pT = doTrunc ? trunc.data() : nullptr;
    double tmax = 0, tsum = 0; long N = 0; int active = 0;
    for (int r = 0; r < R; r++) {
      const long n = numTerms(r);
      N += n; maxLocal = std::max(maxLocal, n);
      if (n == 0) continue;
      active++;
      const double dt = applyGate(r, g, nT, pT);
      tmax = std::max(tmax, dt); tsum += dt;
    }
    maxGlobal = std::max(maxGlobal, N);
    tGatesPar += tmax; tGatesSum += tsum;
    fprintf(fg, "%d,%d,%ld,%d,%.7f,%.7f\n", gi, g.kind, N, active, tmax, tsum);
  }
  const double wall = now() - tBegin;

  double expec = 0; long nFinal = 0;
  for (int r = 0; r < R; r++) {
    const long n = numTerms(r);
    nFinal += n;
    if (n == 0) continue;
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, rk[r].ex[rk[r].cur], 0, n, &v));
    CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, wsMem, ws));
    setWs();
    double sig = 0, ex = 0;
    CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &ex, ws, stream));
    CUDA(cudaStreamSynchronize(stream));
    expec += sig * std::pow(2.0, ex);
    CUPP(cupaulipropDestroyPauliExpansionView(v));
  }
  fclose(fre); fclose(fg);

  printf("expectation=%.12g final_global_terms=%ld max_global_terms=%ld max_local_terms=%ld\n", expec, nFinal, maxGlobal, maxLocal);
  printf("re-keys=%ld (skipped while on one rank: %ld) moved_terms=%ld worst max/mean bucket=%.3f\n",
         rekeys, rekeysSkipped, movedTotal, worstImbalance);
  printf("emulation wall (all ranks run one after another on one GPU)=%.3f s\n", wall);
  printf("gate time: sum over ranks=%.3f s, slowest rank per gate (parallel estimate)=%.3f s\n", tGatesSum, tGatesPar);
  printf("re-key time: partition+copy sum=%.3f s, parallel estimate=%.3f s, row selection=%.3f s\n", tRekeySum, tRekeyPar, tSelect);
  printf("parallel estimate total=%.3f s\n", tGatesPar + tRekeyPar + tSelect);
  return 0;
}
