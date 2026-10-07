// Single-GPU stage breakdown for cuPauliProp on the 127-qubit kicked Ising circuit.
// Each gate is applied in separate stages -- branch (operator application that keeps
// duplicates), deduplication, truncation -- so the cost and the term count after each
// stage are visible. Deduplication can be deferred (--dedup-every k) to measure how
// fast duplicates accumulate, and truncation can be applied to non-deduplicated terms
// (--trunc-raw 1) to measure what that does to the result.

#include <cupauliprop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ibm_eagle.hpp"

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
  int dedupEvery = 1;   // deduplicate after every k-th gate
  int truncRaw = 0;     // 1: truncate without deduplicating first
  int obsQubit = 62;
  double vramFrac = 0.9;
  double expShare = 0.2;
  int maxGates = -1;
  std::string dumpAt;       // comma-separated applied-gate counts at which to dump the expansion
  long dumpMaxTerms = 4000000;
  std::string tag = "split";
  std::string outdir = ".";
};

struct Gate { cupaulipropQuantumOperator_t op; int kind; };

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
    else if (a == "--dedup-every") o.dedupEvery = atoi(next());
    else if (a == "--trunc-raw")   o.truncRaw = atoi(next());
    else if (a == "--obs-qubit")   o.obsQubit = atoi(next());
    else if (a == "--vram-frac")   o.vramFrac = atof(next());
    else if (a == "--exp-share")   o.expShare = atof(next());
    else if (a == "--max-gates")   o.maxGates = atoi(next());
    else if (a == "--dump-at")     o.dumpAt = next();
    else if (a == "--dump-max")    o.dumpMaxTerms = atol(next());
    else if (a == "--tag")         o.tag = next();
    else if (a == "--outdir")      o.outdir = next();
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }

  CUDA(cudaSetDevice(0));
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  CUPP(cupaulipropCreate(&handle));

  int32_t numPackedInts = 0;
  CUPP(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));
  const size_t pauliPerTerm = 2 * (size_t)numPackedInts * sizeof(cupaulipropPackedIntegerType_t);
  const size_t coefPerTerm = sizeof(double);
  size_t freeMem = 0, totalMem = 0;
  CUDA(cudaMemGetInfo(&freeMem, &totalMem));
  const size_t usable = (size_t)(freeMem * o.vramFrac);
  const size_t capTerms = (size_t)(usable * o.expShare) / (pauliPerTerm + coefPerTerm);
  const size_t wsMem = usable - 2 * capTerms * (pauliPerTerm + coefPerTerm);
  const size_t pauliMem = capTerms * pauliPerTerm, coefMem = capTerms * coefPerTerm;

  void *dPauli[2], *dCoef[2], *dWs;
  for (int i = 0; i < 2; i++) { CUDA(cudaMalloc(&dPauli[i], pauliMem)); CUDA(cudaMalloc(&dCoef[i], coefMem)); }
  CUDA(cudaMalloc(&dWs, wsMem));

  {
    std::vector<cupaulipropPackedIntegerType_t> bits(2 * numPackedInts, 0);
    const int w = 8 * sizeof(cupaulipropPackedIntegerType_t);
    bits[numPackedInts + o.obsQubit / w] |= (1ULL << (o.obsQubit % w));
    double one = 1.0;
    CUDA(cudaMemcpy(dPauli[0], bits.data(), bits.size() * sizeof(bits[0]), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dCoef[0], &one, sizeof(one), cudaMemcpyHostToDevice));
  }
  cupaulipropPauliExpansion_t expIn, expOut;
  CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, dPauli[0], pauliMem, dCoef[0], coefMem,
                                       CUDA_R_64F, 1, CUPAULIPROP_SORT_ORDER_NONE, 0, &expIn));
  CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, dPauli[1], pauliMem, dCoef[1], coefMem,
                                       CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &expOut));
  cupaulipropWorkspaceDescriptor_t ws;
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));

  cupaulipropCoefficientTruncationParams_t coefParams; coefParams.cutoff = o.coef;
  cupaulipropPauliWeightTruncationParams_t weightParams; weightParams.cutoff = o.weight;
  std::vector<cupaulipropTruncationStrategy_t> trunc;
  trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
  if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});

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
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, -PI / 2.0, 2, layers[l][i], p, &g.op));
        circuit.push_back(g);
      }
  }

  printf("config: steps=%d theta=%.4fpi coef=%g weight=%d truncEvery=%d dedupEvery=%d truncRaw=%d capTerms=%zu ws=%.0f MiB\n",
         o.steps, o.thetaOverPi, o.coef, o.weight, o.truncEvery, o.dedupEvery, o.truncRaw, capTerms, wsMem / 1048576.0);

  auto numTerms = [&](cupaulipropPauliExpansion_t e) { int64_t n = 0; CUPP(cupaulipropPauliExpansionGetNumTerms(handle, e, &n)); return (long)n; };
  auto setWs = [&]() {
    int64_t need = 0;
    CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
    if (need > (int64_t)wsMem) { fprintf(stderr, "workspace exceeded: need %ld have %zu\n", (long)need, wsMem); return false; }
    CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
    return true;
  };

  char path[1024];
  snprintf(path, sizeof(path), "%s/%s.csv", o.outdir.c_str(), o.tag.c_str());
  FILE* csv = fopen(path, "w");
  if (!csv) die("cannot open csv", __LINE__);
  fprintf(csv, "gate,kind,nin,n_branch,n_dedup,n_trunc,t_branch,t_dedup,t_trunc,did_dedup,did_trunc\n");

  std::vector<int> dumpPoints;
  for (size_t pos = 0; pos < o.dumpAt.size();) {
    size_t e = o.dumpAt.find(',', pos);
    if (e == std::string::npos) e = o.dumpAt.size();
    dumpPoints.push_back(atoi(o.dumpAt.substr(pos, e - pos).c_str()));
    pos = e + 1;
  }
  // Writes a strided sample of the current expansion: 4 x uint64 (x0,x1,z0,z1) and one double per term.
  auto dumpExpansion = [&](int appliedCount) {
    void *xz = nullptr, *cf = nullptr; int64_t xzSize = 0, cfSize = 0, n = 0; cupaulipropMemspace_t loc;
    CUPP(cupaulipropPauliExpansionGetStorageBuffer(handle, expIn, &xz, &xzSize, &cf, &cfSize, &n, &loc));
    const long stride = std::max<long>(1, (n + o.dumpMaxTerms - 1) / o.dumpMaxTerms);
    std::vector<uint64_t> hx((size_t)n * 2 * numPackedInts);
    std::vector<double> hc((size_t)n);
    CUDA(cudaMemcpy(hx.data(), xz, hx.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(hc.data(), cf, hc.size() * sizeof(double), cudaMemcpyDeviceToHost));
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/%s.g%05d.n%ld.s%ld.xz.bin", o.outdir.c_str(), o.tag.c_str(), appliedCount, (long)n, stride);
    snprintf(p2, sizeof(p2), "%s/%s.g%05d.n%ld.s%ld.coef.bin", o.outdir.c_str(), o.tag.c_str(), appliedCount, (long)n, stride);
    FILE* f1 = fopen(p1, "wb"); FILE* f2 = fopen(p2, "wb");
    if (!f1 || !f2) die("cannot open dump file", __LINE__);
    const size_t w = 2 * numPackedInts;
    for (long i = 0; i < n; i += stride) { fwrite(&hx[(size_t)i * w], sizeof(uint64_t), w, f1); fwrite(&hc[i], sizeof(double), 1, f2); }
    fclose(f1); fclose(f2);
  };

  double tBranch = 0, tDedup = 0, tTrunc = 0;
  long maxTerms = 0;
  int applied = 0;
  bool ok = true;
  CUDA(cudaStreamSynchronize(stream));
  const double tBegin = now();

  for (int gi = (int)circuit.size() - 1; gi >= 0 && ok; --gi) {
    if (o.maxGates >= 0 && applied >= o.maxGates) break;
    const long nin = numTerms(expIn);
    long nBranch = nin, nDedup = nin, nTrunc = nin;
    double dB = 0, dD = 0, dT = 0;
    int didDedup = 0, didTrunc = 0;

    // ---- branch: apply the gate, keep duplicates, no truncation ----
    {
      const double t0 = now();
      cupaulipropPauliExpansionView_t v;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, expIn, 0, nin, &v));
      int64_t rp = 0, rc = 0;
      CUPP(cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, v, circuit[gi].op, CUPAULIPROP_SORT_ORDER_NONE, 1, 0, nullptr, wsMem, &rp, &rc, ws));
      if (rp > (int64_t)pauliMem || rc > (int64_t)coefMem || !setWs()) {
        fprintf(stderr, "capacity exceeded in branch at gate %d (applied %d): nin=%ld\n", gi, applied, nin);
        ok = false; break;
      }
      CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, v, expOut, circuit[gi].op, 1, CUPAULIPROP_SORT_ORDER_NONE, 1, 0, nullptr, ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      std::swap(expIn, expOut);
      nBranch = numTerms(expIn);
      dB = now() - t0;
    }
    applied++;
    const bool doTrunc = (gi % o.truncEvery == 0);
    const bool doDedup = (applied % o.dedupEvery == 0) || (doTrunc && !o.truncRaw) || gi == 0;
    nDedup = nBranch;

    // ---- deduplication ----
    if (doDedup) {
      int32_t isDedup = 0;
      CUPP(cupaulipropPauliExpansionIsDeduplicated(handle, expIn, &isDedup));
      if (!isDedup) {
        const double t0 = now();
        cupaulipropPauliExpansionView_t v;
        CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, expIn, 0, nBranch, &v));
        CUPP(cupaulipropPauliExpansionViewPrepareDeduplication(handle, v, CUPAULIPROP_SORT_ORDER_NONE, wsMem, ws));
        if (!setWs()) { fprintf(stderr, "capacity exceeded in dedup at gate %d: n=%ld\n", gi, nBranch); ok = false; break; }
        CUPP(cupaulipropPauliExpansionViewExecuteDeduplication(handle, v, expOut, CUPAULIPROP_SORT_ORDER_NONE, ws, stream));
        CUDA(cudaStreamSynchronize(stream));
        CUPP(cupaulipropDestroyPauliExpansionView(v));
        std::swap(expIn, expOut);
        nDedup = numTerms(expIn);
        dD = now() - t0;
        didDedup = 1;
      }
    }
    nTrunc = nDedup;

    // ---- truncation ----
    if (doTrunc && nDedup > 0) {
      const double t0 = now();
      cupaulipropPauliExpansionView_t v;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, expIn, 0, nDedup, &v));
      CUPP(cupaulipropPauliExpansionViewPrepareTruncation(handle, v, (int32_t)trunc.size(), trunc.data(), wsMem, ws));
      if (!setWs()) { fprintf(stderr, "capacity exceeded in truncation at gate %d\n", gi); ok = false; break; }
      CUPP(cupaulipropPauliExpansionViewExecuteTruncation(handle, v, expOut, (int32_t)trunc.size(), trunc.data(), ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      std::swap(expIn, expOut);
      nTrunc = numTerms(expIn);
      dT = now() - t0;
      didTrunc = 1;
    }

    if (std::find(dumpPoints.begin(), dumpPoints.end(), applied) != dumpPoints.end()) dumpExpansion(applied);
    tBranch += dB; tDedup += dD; tTrunc += dT;
    maxTerms = std::max(maxTerms, std::max(nBranch, nin));
    fprintf(csv, "%d,%d,%ld,%ld,%ld,%ld,%.7f,%.7f,%.7f,%d,%d\n", gi, circuit[gi].kind, nin, nBranch, nDedup, nTrunc,
            dB, dD, dT, didDedup, didTrunc);
  }
  const double wall = now() - tBegin;
  fclose(csv);

  double expec = NAN;
  if (ok) {
    const long nfin = numTerms(expIn);
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, expIn, 0, nfin, &v));
    CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, wsMem, ws));
    if (setWs()) {
      double sig = 0, ex = 0;
      CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &ex, ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      expec = sig * std::pow(2.0, ex);
    }
    CUPP(cupaulipropDestroyPauliExpansionView(v));
  }
  printf("ok=%d gates_applied=%d expectation=%.12g final_terms=%ld max_terms=%ld wall=%.4f s\n",
         (int)ok, applied, expec, numTerms(expIn), maxTerms, wall);
  printf("stage time: branch=%.4f s dedup=%.4f s truncation=%.4f s\n", tBranch, tDedup, tTrunc);
  return ok ? 0 : 2;
}
