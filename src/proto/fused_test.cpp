// Single-GPU test of cuPauliProp's experimental fused operator application.
//
// The library applies several operators in one call (cupaulipropPauliExpansionViewComputeOperatorFusedApplication).
// In cuPauliProp 0.5.0 this works for Clifford gates and Pauli noise channels only, all operators of one
// call of the same kind; rotations cannot be fused. This program runs the engine's operation list of the
// noisy kicked Ising circuit on one GPU, in memory, and compares
//   --fuse 0   one library call per operation (what the engine does)
//   --fuse k   up to k consecutive Clifford gates, or noise channels, in one call. A group ends at an
//              operation that truncates, so the truncation points are the same and so is the answer.
// It prints the expectation value, the time per kind of operation and the number of library calls.
// Not part of the engine.
#include <cuda_runtime.h>
#include <cupauliprop.h>

#include <chrono>
#include <cstdint>
#include <cstring>

#include "../engine/circuit.hpp"

static void die(const char* what, int line) { fprintf(stderr, "FATAL: %s (line %d)\n", what, line); std::exit(1); }
#define CUPP(x) do { cupaulipropStatus_t s_ = (x); if (s_ != CUPAULIPROP_STATUS_SUCCESS) { \
  fprintf(stderr, "cuPauliProp status %d at line %d\n", (int)s_, __LINE__); std::exit(1); } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  int fuse = 0, steps = 20, truncEvery = 10;
  double coef = 1e-4, noise = 0.005, theta = 0.25, vramFrac = 0.5, junkCut = 1e-13;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) die("missing value", __LINE__); return argv[++i]; };
    if (a == "--fuse") fuse = atoi(next());
    else if (a == "--coef") coef = atof(next());
    else if (a == "--noise") noise = atof(next());
    else if (a == "--steps") steps = atoi(next());
    else if (a == "--trunc-every") truncEvery = atoi(next());
    else if (a == "--vram-frac") vramFrac = atof(next());
    else die("unknown option", __LINE__);
  }
  Problem P = makeIsing("noisy kicked Ising, Eagle heavy-hex", NUM_CIRCUIT_QUBITS, heavyHexEagle(), steps, theta, -0.5, noise, 62, truncEvery);
  const Layout L = P.L;
  int deferred = 0;
  const std::vector<Op> ops = compileForEngine(P, true, &deferred);
  const int M = (int)ops.size();
  int nRot = 0, nRotCliff = 0, nCliff = 0, nNoise = 0;
  for (auto& p : ops) {
    if (p.type == OP_ROT) (p.branches ? nRot : nRotCliff)++;
    else if (p.type == OP_CLIFF) nCliff++;
    else nNoise++;
  }
  printf("circuit: %s; %d operations: %d branching rotations, %d rotations with a Clifford angle, %d Clifford gates, %d noise channels\n",
         P.name.c_str(), M, nRot, nRotCliff, nCliff, nNoise);

  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  CUPP(cupaulipropCreate(&handle));
  const size_t xzPerTerm = L.words() * sizeof(uint64_t), perTerm = xzPerTerm + sizeof(double);
  size_t freeMem = 0, totalMem = 0;
  CUDA(cudaMemGetInfo(&freeMem, &totalMem));
  const size_t usable = (size_t)(freeMem * vramFrac);
  const size_t capTerms = (size_t)(usable * 0.2) / perTerm;
  const size_t xzMem = capTerms * xzPerTerm, cfMem = capTerms * sizeof(double), wsMem = usable - 2 * (xzMem + cfMem);
  void *xz[2], *cf[2], *dWs;
  for (int b = 0; b < 2; b++) { CUDA(cudaMalloc(&xz[b], xzMem)); CUDA(cudaMalloc(&cf[b], cfMem)); }
  CUDA(cudaMalloc(&dWs, wsMem));
  std::vector<uint64_t> hx; std::vector<double> hc;
  for (auto& t : P.observable) { hx.insert(hx.end(), t.first.begin(), t.first.end()); hc.push_back(t.second); }
  CUDA(cudaMemcpy(xz[0], hx.data(), hx.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(cf[0], hc.data(), hc.size() * sizeof(double), cudaMemcpyHostToDevice));
  cupaulipropPauliExpansion_t ex[2];
  CUPP(cupaulipropCreatePauliExpansion(handle, L.n, xz[0], xzMem, cf[0], cfMem, CUDA_R_64F, (int64_t)hc.size(), CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[0]));
  CUPP(cupaulipropCreatePauliExpansion(handle, L.n, xz[1], xzMem, cf[1], cfMem, CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[1]));
  cupaulipropWorkspaceDescriptor_t ws, wsMin, wsAvg, wsMax;
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &wsMin));
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &wsAvg));
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &wsMax));
  cupaulipropCoefficientTruncationParams_t coefParams, junkParams; coefParams.cutoff = coef; junkParams.cutoff = junkCut;
  std::vector<cupaulipropTruncationStrategy_t> trunc, junk;
  trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
  junk.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &junkParams});
  std::vector<cupaulipropQuantumOperator_t> oper(M);
  for (int i = 0; i < M; i++) {
    const Op& p = ops[i];
    if (p.type == OP_ROT) CUPP(cupaulipropCreatePauliRotationGateOperator(handle, p.angle, (int32_t)p.qubits.size(), p.qubits.data(), p.paulis.data(), &oper[i]));
    else if (p.type == OP_CLIFF) CUPP(cupaulipropCreateCliffordGateOperator(handle, p.cliff, p.qubits.data(), &oper[i]));
    else CUPP(cupaulipropCreatePauliNoiseChannelOperator(handle, (int32_t)p.qubits.size(), p.qubits.data(), p.prob, &oper[i]));
  }
  std::vector<int32_t> adj(std::max(fuse, 1), 1);

  // time and library calls per kind: 0 branching rotation, 1 Clifford gate, 2 noise channel, 3 rotation with a Clifford angle
  double tKind[4] = {0, 0, 0, 0};
  long calls[4] = {0, 0, 0, 0}, opsDone[4] = {0, 0, 0, 0}, fusedCalls = 0, fusedOps = 0, refused = 0, maxTerms = 0;
  int cur = 0;
  CUDA(cudaStreamSynchronize(stream));
  const double tStart = now();
  for (int k = 0; k < M;) {
    const Op& op = ops[k];
    const int kind = op.type == OP_ROT ? (op.branches ? 0 : 3) : (op.type == OP_CLIFF ? 1 : 2);
    int64_t m = 0;
    CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &m));
    maxTerms = std::max<long>(maxTerms, m);
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
    // A group: consecutive operations of this kind, ending at the first one that truncates.
    int n = 1;
    if (fuse > 1 && (kind == 1 || kind == 2))
      while (n < fuse && k + n < M && ops[k + n].type == op.type && !ops[k + n - 1].trunc) n++;
    const double t0 = now();
    bool done = false;
    while (n > 1 && !done) {
      const bool tr = ops[k + n - 1].trunc;
      const int32_t nT = tr ? (int32_t)trunc.size() : 0;
      const cupaulipropTruncationStrategy_t* pT = tr ? trunc.data() : nullptr;
      int64_t minCap = 0, avgCap = 0, maxCap = 0, need = 0;
      cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorFusedApplication(
        handle, v, n, &oper[k], adj.data(), nT, pT, (int64_t)wsMem, &minCap, wsMin, &avgCap, wsAvg, &maxCap, wsMax);
      if (sp == CUPAULIPROP_STATUS_INSUFFICIENT_DEVICE_PROPERTY) { refused++; n = n / 2; continue; }   // too many operators for one launch
      if (sp != CUPAULIPROP_STATUS_SUCCESS) { fprintf(stderr, "fused prepare: status %d at operation %d with %d operators\n", (int)sp, k, n); die("fused prepare", __LINE__); }
      CUPP(cupaulipropWorkspaceGetMemorySize(handle, wsMax, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
      if (maxCap > (int64_t)capTerms || need > (int64_t)wsMem) die("fused application does not fit", __LINE__);
      CUPP(cupaulipropWorkspaceSetMemory(handle, wsMax, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      cupaulipropStatus_t sc = cupaulipropPauliExpansionViewComputeOperatorFusedApplication(
        handle, v, ex[1 - cur], n, &oper[k], adj.data(), nT, pT, wsMax, stream);
      if (sc == CUPAULIPROP_STATUS_INSUFFICIENT_DEVICE_PROPERTY) { refused++; n = n / 2; continue; }
      if (sc != CUPAULIPROP_STATUS_SUCCESS) { fprintf(stderr, "fused compute: status %d at operation %d with %d operators\n", (int)sc, k, n); die("fused compute", __LINE__); }
      done = true; fusedCalls++; fusedOps += n;
    }
    if (!done) {
      n = 1;
      const bool dust = (op.type == OP_ROT && !op.branches);
      const int32_t nT = op.trunc ? (int32_t)trunc.size() : (dust ? 1 : 0);
      const cupaulipropTruncationStrategy_t* pT = op.trunc ? trunc.data() : (dust ? junk.data() : nullptr);
      int64_t rp = 0, rc = 0;
      cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, v, oper[k], CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, (int64_t)wsMem, &rp, &rc, ws);
      if (sp != CUPAULIPROP_STATUS_SUCCESS || rp > (int64_t)xzMem || rc > (int64_t)cfMem) {
        fprintf(stderr, "out of memory at operation %d of %d with %ld terms (status %d)\n", k, M, (long)m, (int)sp);
        die("capacity", __LINE__);
      }
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, v, ex[1 - cur], oper[k], 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, ws, stream));
    }
    CUDA(cudaStreamSynchronize(stream));
    tKind[kind] += now() - t0;
    calls[kind]++; opsDone[kind] += n;
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    cur ^= 1;
    k += n;
  }
  int64_t m = 0;
  CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &m));
  cupaulipropPauliExpansionView_t v;
  CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
  CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, (int64_t)wsMem, ws));
  CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
  double sig = 0, e2 = 0;
  CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &e2, ws, stream));
  CUDA(cudaStreamSynchronize(stream));
  const double wall = now() - tStart;
  printf("fuse=%d coef=%g expectation=%.12g final_terms=%ld largest_expansion=%ld terms\n", fuse, coef, sig * std::pow(2.0, e2), (long)m, maxTerms);
  printf("wall=%.3f s; branching rotations=%.3f s in %ld calls; Clifford-angle rotations=%.3f s in %ld calls; Clifford gates=%.3f s in %ld calls (%ld operations); "
         "noise channels=%.3f s in %ld calls (%ld operations)\n",
         wall, tKind[0], calls[0], tKind[3], calls[3], tKind[1], calls[1], opsDone[1], tKind[2], calls[2], opsDone[2]);
  printf("fused calls=%ld covering %ld operations; group sizes refused by the device=%ld\n", fusedCalls, fusedOps, refused);
  return 0;
}
