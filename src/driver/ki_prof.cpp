// Per-gate profiling driver for cuPauliProp on the 127-qubit kicked Ising circuit.
// Runs single-process or distributed (MPI provider) and records, for every gate
// and every rank: local term counts, prepare/compute wall time, the memory the
// library asks for, and the MPI activity reported by the profiling plugin.

#include <cupauliprop.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ibm_eagle.hpp"

static int g_rank = 0;

[[noreturn]] static void die(const char* what, int line) {
  fprintf(stderr, "[rank %d] FATAL: %s (line %d)\n", g_rank, what, line);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort();
}
#define CUPP(x) do { cupaulipropStatus_t s_ = (x); if (s_ != CUPAULIPROP_STATUS_SUCCESS) { \
  fprintf(stderr, "[rank %d] cuPauliProp status %d\n", g_rank, (int)s_); die(#x, __LINE__); } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "[rank %d] CUDA: %s\n", g_rank, cudaGetErrorString(e_)); die(#x, __LINE__); } } while (0)

struct Opts {
  int steps = 20;
  double thetaOverPi = 0.25;
  double zzOverPi = -0.5;  // ZZ rotation angle in units of pi (-0.5 is the Clifford angle of the kicked Ising circuit)
  double coef = 1e-4;
  int weight = 8;          // 0 disables weight truncation
  int truncEvery = 10;     // truncate when gateInd % truncEvery == 0
  int keepDup = 0;         // 1: no global deduplication inside operator application
  int dedupEvery = 0;      // with keepDup=1: explicit global deduplication every k gates (0: never)
  int obsQubit = 62;
  double vramFrac = 0.0;   // >0: size buffers from free GPU memory
  double expShare = 0.2;   // share of usable memory per expansion when vramFrac>0
  long capTerms = 1L << 19;
  long wsMiB = 128;
  int dist = 1;            // 0: never enable the distributed configuration
  int maxGates = -1;
  int frame = 0;           // 1: absorb the ZZ(pi/2) Clifford layers into the rotation generators (no ZZ gate touches data)
  int frameDir = 1;
  std::string tag = "run";
  std::string outdir = ".";
};

static Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { if (i + 1 >= argc) die("missing option value", __LINE__); return argv[++i]; };
    if      (a == "--steps")       o.steps = atoi(next());
    else if (a == "--theta")       o.thetaOverPi = atof(next());
    else if (a == "--zz-theta")    o.zzOverPi = atof(next());
    else if (a == "--coef")        o.coef = atof(next());
    else if (a == "--weight")      o.weight = atoi(next());
    else if (a == "--trunc-every") o.truncEvery = atoi(next());
    else if (a == "--keepdup")     o.keepDup = atoi(next());
    else if (a == "--dedup-every") o.dedupEvery = atoi(next());
    else if (a == "--obs-qubit")   o.obsQubit = atoi(next());
    else if (a == "--vram-frac")   o.vramFrac = atof(next());
    else if (a == "--exp-share")   o.expShare = atof(next());
    else if (a == "--cap-terms")   o.capTerms = atol(next());
    else if (a == "--ws-mib")      o.wsMiB = atol(next());
    else if (a == "--dist")        o.dist = atoi(next());
    else if (a == "--max-gates")   o.maxGates = atoi(next());
    else if (a == "--frame")       o.frame = atoi(next());
    else if (a == "--frame-dir")   o.frameDir = atoi(next());
    else if (a == "--tag")         o.tag = next();
    else if (a == "--outdir")      o.outdir = next();
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }
  return o;
}

// ---- profiler hooks exported by the profiling plugin (optional) ----
typedef void (*SnapFn)(double*);
typedef void (*MarkFn)(void);
typedef int  (*NkFn)(void);
static SnapFn g_snap = nullptr;
static MarkFn g_mark = nullptr;
static int g_nk = 0;
static const int MAXP = 64;
static const char* KIND_NAMES[] = {"barrier", "wait", "test", "send", "isend", "recv", "irecv",
                                   "bcast", "allred", "allredmin", "minloc", "allgather"};

static void hookProfiler() {
  const char* lib = getenv("CUPAULIPROP_COMM_LIB");
  if (!lib) return;
  void* h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
  if (!h) return;
  NkFn nk = (NkFn)dlsym(h, "cuppprof_num_kinds");
  g_snap = (SnapFn)dlsym(h, "cuppprof_snapshot");
  g_mark = (MarkFn)dlsym(h, "cuppprof_mark");
  if (nk && g_snap && g_mark) g_nk = nk(); else { g_snap = nullptr; g_mark = nullptr; }
}

enum RecKind { REC_X = 0, REC_ZZ = 1, REC_DEDUP = 2, REC_TRACE = 3, REC_TRUNC = 4 };

struct Rec {
  int gate, kind;
  long nin, nout;
  double tStart, tPrep, tComp;
  long reqPauli, reqCoef, reqWs;
  double prof[MAXP];
};

struct Gate { cupaulipropQuantumOperator_t op; int kind; bool trunc; };

int main(int argc, char** argv) {
  // Cray MPICH's GPU transport binds to the CUDA device that is current at MPI_Init.
  // Selecting the rank's device first keeps its IPC path enabled; PP_EARLY_DEVICE=0
  // restores the order used by NVIDIA's sample (device selected after MPI_Init).
  {
    const char* lid = getenv("SLURM_LOCALID");
    const char* early = getenv("PP_EARLY_DEVICE");
    int nd = 0;
    if (lid && !(early && atoi(early) == 0) && cudaGetDeviceCount(&nd) == cudaSuccess && nd > 0) {
      cudaSetDevice(atoi(lid) % nd);
      void* warm = nullptr;
      if (cudaMalloc(&warm, 1 << 20) == cudaSuccess) cudaFree(warm);  // force context creation
    }
  }
  MPI_Init(&argc, &argv);
  int numRanks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  const int rank = g_rank;
  Opts o = parse(argc, argv);

  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
  int localRank = 0;
  MPI_Comm_rank(localComm, &localRank);
  MPI_Comm_free(&localComm);
  int numDevices = 0;
  CUDA(cudaGetDeviceCount(&numDevices));
  if (numDevices == 0) die("no CUDA device", __LINE__);
  CUDA(cudaSetDevice(localRank % numDevices));

  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  CUPP(cupaulipropCreate(&handle));

  const bool distributed = o.dist && getenv("CUPAULIPROP_COMM_LIB") != nullptr;
  MPI_Comm cuppComm = MPI_COMM_NULL;
  if (distributed) {
    hookProfiler();
    MPI_Comm_dup(MPI_COMM_WORLD, &cuppComm);
    CUPP(cupaulipropResetDistributedConfiguration(
      handle, CUPAULIPROP_DISTRIBUTED_PROVIDER_MPI, &cuppComm, sizeof(cuppComm)));
  } else if (numRanks > 1) {
    die("multiple ranks need CUPAULIPROP_COMM_LIB and --dist 1", __LINE__);
  }

  // ---- memory ----
  int32_t numPackedInts = 0;
  CUPP(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));
  const size_t pauliPerTerm = 2 * (size_t)numPackedInts * sizeof(cupaulipropPackedIntegerType_t);
  const size_t coefPerTerm = sizeof(double);
  size_t freeMem = 0, totalMem = 0;
  CUDA(cudaMemGetInfo(&freeMem, &totalMem));
  size_t capTerms, wsMem;
  if (o.vramFrac > 0) {
    size_t usable = (size_t)(freeMem * o.vramFrac);
    capTerms = (size_t)(usable * o.expShare) / (pauliPerTerm + coefPerTerm);
    wsMem = usable - 2 * capTerms * (pauliPerTerm + coefPerTerm);
  } else {
    capTerms = (size_t)o.capTerms;
    wsMem = (size_t)o.wsMiB << 20;
  }
  const size_t pauliMem = capTerms * pauliPerTerm;
  const size_t coefMem = capTerms * coefPerTerm;

  void *dPauli[2], *dCoef[2], *dWs;
  for (int i = 0; i < 2; i++) { CUDA(cudaMalloc(&dPauli[i], pauliMem)); CUDA(cudaMalloc(&dCoef[i], coefMem)); }
  CUDA(cudaMalloc(&dWs, wsMem));

  // ---- observable Z_q on rank 0 ----
  long numLocalTerms = 0;
  if (rank == 0) {
    numLocalTerms = 1;
    std::vector<cupaulipropPackedIntegerType_t> bits(2 * numPackedInts, 0);
    const int w = 8 * sizeof(cupaulipropPackedIntegerType_t);
    bits[numPackedInts + o.obsQubit / w] |= (1ULL << (o.obsQubit % w));
    double one = 1.0;
    CUDA(cudaMemcpy(dPauli[0], bits.data(), bits.size() * sizeof(bits[0]), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dCoef[0], &one, sizeof(one), cudaMemcpyHostToDevice));
  }

  cupaulipropPauliExpansion_t expIn, expOut;
  CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, dPauli[0], pauliMem, dCoef[0], coefMem,
                                       CUDA_R_64F, numLocalTerms, CUPAULIPROP_SORT_ORDER_NONE, 0, &expIn));
  CUPP(cupaulipropCreatePauliExpansion(handle, NUM_CIRCUIT_QUBITS, dPauli[1], pauliMem, dCoef[1], coefMem,
                                       CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &expOut));

  cupaulipropWorkspaceDescriptor_t ws;
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));
  CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE,
                                     CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));

  // ---- truncation ----
  cupaulipropCoefficientTruncationParams_t coefParams; coefParams.cutoff = o.coef;
  cupaulipropPauliWeightTruncationParams_t weightParams; weightParams.cutoff = o.weight;
  std::vector<cupaulipropTruncationStrategy_t> trunc;
  trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
  if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});

  // ---- circuit ----
  const double PI = 3.14159265358979323846;
  std::vector<Gate> circuit;
  const int32_t (*zzLayers[3])[2] = {ZZ_QUBITS_RED, ZZ_QUBITS_BLUE, ZZ_QUBITS_GREEN};
  if (!o.frame) {
    for (int n = 0; n < o.steps; n++) {
      for (int32_t q = 0; q < NUM_CIRCUIT_QUBITS; q++) {
        cupaulipropPauliKind_t p[1] = {CUPAULIPROP_PAULI_X};
        Gate g; g.kind = REC_X;
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, o.thetaOverPi * PI, 1, &q, p, &g.op));
        circuit.push_back(g);
      }
      for (int l = 0; l < 3; l++) {
        for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) {
          cupaulipropPauliKind_t p[2] = {CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z};
          Gate g; g.kind = REC_ZZ;
          CUPP(cupaulipropCreatePauliRotationGateOperator(handle, o.zzOverPi * PI, 2, zzLayers[l][i], p, &g.op));
          circuit.push_back(g);
        }
      }
    }
    for (size_t i = 0; i < circuit.size(); i++) circuit[i].trunc = (i % o.truncEvery == 0);
  } else {
    // Clifford frame. With C = prod_edges exp(+i pi/4 Z_a Z_b) the circuit (C R)^steps equals
    // C^steps R^(steps-1) ... R^(0), where R^(s) rotates about C^{-s} X_q C^{s}. C^2 is a Pauli,
    // so these generators are X_q (even s) or X_q/Y_q times Z on the neighbours (odd s), up to a sign.
    // C^steps commutes with the Z observable. Truncation points match the original gate indices; the
    // truncations that fell inside a ZZ block become one truncation after the preceding X layer.
    std::vector<std::vector<int32_t>> nb(NUM_CIRCUIT_QUBITS);
    for (int l = 0; l < 3; l++)
      for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) {
        nb[zzLayers[l][i][0]].push_back(zzLayers[l][i][1]);
        nb[zzLayers[l][i][1]].push_back(zzLayers[l][i][0]);
      }
    for (int n = 0; n < o.steps; n++) {
      for (int32_t q = 0; q < NUM_CIRCUIT_QUBITS; q++) {
        const int deg = (int)nb[q].size();
        double sign = ((deg % 2) && ((n / 2) % 2)) ? -1.0 : 1.0;      // (-1)^(deg * floor(n/2))
        std::vector<int32_t> qs; std::vector<cupaulipropPauliKind_t> ps;
        if (n % 2 == 0) {
          qs.push_back(q); ps.push_back(CUPAULIPROP_PAULI_X);
        } else {
          qs.push_back(q); ps.push_back(deg % 2 ? CUPAULIPROP_PAULI_Y : CUPAULIPROP_PAULI_X);
          for (int32_t b : nb[q]) { qs.push_back(b); ps.push_back(CUPAULIPROP_PAULI_Z); }
          if (deg % 4 == 1) sign *= o.frameDir; else if (deg % 4 == 2) sign *= -1.0; else if (deg % 4 == 3) sign *= -o.frameDir;
        }
        Gate g; g.kind = REC_X;
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, sign * o.thetaOverPi * PI, (int32_t)qs.size(), qs.data(), ps.data(), &g.op));
        g.trunc = ((271 * n + q) % o.truncEvery == 0) || (q == 0 && n > 0);
        circuit.push_back(g);
      }
    }
  }

  if (rank == 0) {
    printf("config: ranks=%d distributed=%d profiler_kinds=%d steps=%d theta=%.4fpi coef=%g weight=%d "
           "truncEvery=%d keepDup=%d dedupEvery=%d\n",
           numRanks, (int)distributed, g_nk, o.steps, o.thetaOverPi, o.coef, o.weight,
           o.truncEvery, o.keepDup, o.dedupEvery);
    printf("memory per rank: free=%.1f MiB capTerms=%zu (%.1f MiB per expansion) workspace=%.1f MiB\n",
           freeMem / 1048576.0, capTerms, (pauliMem + coefMem) / 1048576.0, wsMem / 1048576.0);
    printf("gates=%zu\n", circuit.size());
    fflush(stdout);
  }

  std::vector<Rec> recs;
  recs.reserve(circuit.size() * 2 + 8);
  const int np = g_nk ? 3 * g_nk + 2 : 0;
  double s0[MAXP] = {0}, s1[MAXP] = {0};

  // Per-rank CSV, flushed every FLUSH_EVERY records so an aborted run keeps its trace.
  char csvPath[1024];
  snprintf(csvPath, sizeof(csvPath), "%s/%s.rank%04d.csv", o.outdir.c_str(), o.tag.c_str(), rank);
  FILE* csv = fopen(csvPath, "w");
  if (!csv) die("cannot open output csv", __LINE__);
  fprintf(csv, "gate,kind,nin,nout,t_start,t_prep,t_comp,req_pauli,req_coef,req_ws");
  for (int k = 0; k < g_nk; k++) fprintf(csv, ",%s_calls", KIND_NAMES[k]);
  for (int k = 0; k < g_nk; k++) fprintf(csv, ",%s_time", KIND_NAMES[k]);
  for (int k = 0; k < g_nk; k++) fprintf(csv, ",%s_bytes", KIND_NAMES[k]);
  if (g_nk) fprintf(csv, ",dest_pairs,max_msg_bytes");
  fprintf(csv, "\n");
  size_t flushed = 0;
  const size_t FLUSH_EVERY = 256;
  auto flushRecs = [&](bool force) {
    if (!force && recs.size() - flushed < FLUSH_EVERY) return;
    for (; flushed < recs.size(); flushed++) {
      const Rec& r = recs[flushed];
      fprintf(csv, "%d,%d,%ld,%ld,%.6f,%.6f,%.6f,%ld,%ld,%ld", r.gate, r.kind, r.nin, r.nout,
              r.tStart, r.tPrep, r.tComp, r.reqPauli, r.reqCoef, r.reqWs);
      for (int k = 0; k < np; k++) fprintf(csv, ",%.9g", r.prof[k]);
      fprintf(csv, "\n");
    }
    fflush(csv);
  };

  CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  const double tBegin = MPI_Wtime();
  long maxLocal = 0;
  int applied = 0;

  for (int gi = (int)circuit.size() - 1; gi >= 0; --gi) {
    if (o.maxGates >= 0 && applied >= o.maxGates) break;
    Rec r; memset(&r, 0, sizeof(r));
    r.gate = gi; r.kind = circuit[gi].kind;
    if (g_mark) { g_mark(); g_snap(s0); }
    const double t0 = MPI_Wtime();

    int64_t nin = 0;
    CUPP(cupaulipropPauliExpansionGetNumTerms(handle, expIn, &nin));
    cupaulipropPauliExpansionView_t view;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, expIn, 0, nin, &view));
    maxLocal = std::max<long>(maxLocal, nin);

    // With keepDup the library truncates rank-locally, before any global merge, which
    // changes the result. In that mode truncation is done explicitly after a global
    // deduplication instead (see below).
    const bool doTrunc = circuit[gi].trunc;
    const bool fusedTrunc = doTrunc && !o.keepDup;
    const int32_t nTrunc = fusedTrunc ? (int32_t)trunc.size() : 0;
    const cupaulipropTruncationStrategy_t* pTrunc = fusedTrunc ? trunc.data() : nullptr;

    int64_t reqPauli = 0, reqCoef = 0, reqWs = 0;
    {
      cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, view, circuit[gi].op, CUPAULIPROP_SORT_ORDER_NONE, o.keepDup,
        nTrunc, pTrunc, wsMem, &reqPauli, &reqCoef, ws);
      if (sp != CUPAULIPROP_STATUS_SUCCESS) {
        fprintf(stderr, "[rank %d] prepare failed with status %d at gate %d (applied %d): local nin=%ld "
                "(%.3f GiB live), expansion capacity=%zu terms, workspace=%.2f GiB\n",
                rank, (int)sp, gi, applied, (long)nin, nin * 40.0 / 1073741824.0, capTerms, wsMem / 1073741824.0);
        flushRecs(true);
        die("prepare operator application", __LINE__);
      }
    }
    CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE,
                                           CUPAULIPROP_WORKSPACE_SCRATCH, &reqWs));
    if (reqPauli > (int64_t)pauliMem || reqCoef > (int64_t)coefMem || reqWs > (int64_t)wsMem) {
      fprintf(stderr, "[rank %d] capacity exceeded at gate %d (applied %d): nin=%ld reqPauli=%ld/%zu "
              "reqCoef=%ld/%zu reqWs=%ld/%zu\n", rank, gi, applied, (long)nin,
              (long)reqPauli, pauliMem, (long)reqCoef, coefMem, (long)reqWs, wsMem);
      flushRecs(true);
      die("capacity exceeded", __LINE__);
    }
    CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE,
                                       CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
    const double t1 = MPI_Wtime();

    cupaulipropStatus_t st = cupaulipropPauliExpansionViewComputeOperatorApplication(
      handle, view, expOut, circuit[gi].op, 1, CUPAULIPROP_SORT_ORDER_NONE, o.keepDup,
      nTrunc, pTrunc, ws, stream);
    if (st != CUPAULIPROP_STATUS_SUCCESS) {
      fprintf(stderr, "[rank %d] operator application failed with status %d at gate %d (applied %d), nin=%ld\n",
              rank, (int)st, gi, applied, (long)nin);
      flushRecs(true);
      die("operator application", __LINE__);
    }
    CUDA(cudaStreamSynchronize(stream));
    const double t2 = MPI_Wtime();
    if (g_snap) g_snap(s1);

    int64_t nout = 0;
    CUPP(cupaulipropPauliExpansionGetNumTerms(handle, expOut, &nout));
    r.nin = nin; r.nout = nout;
    r.tStart = t0 - tBegin; r.tPrep = t1 - t0; r.tComp = t2 - t1;
    r.reqPauli = reqPauli; r.reqCoef = reqCoef; r.reqWs = reqWs;
    for (int k = 0; k < np; k++) r.prof[k] = s1[k] - s0[k];
    recs.push_back(r);

    CUPP(cupaulipropDestroyPauliExpansionView(view));
    std::swap(expIn, expOut);
    applied++;
    flushRecs(false);

    const bool explicitDedup = o.keepDup && (doTrunc || (o.dedupEvery > 0 && applied % o.dedupEvery == 0));
    if (explicitDedup) {
      Rec d; memset(&d, 0, sizeof(d));
      d.gate = gi; d.kind = REC_DEDUP;
      if (g_mark) { g_mark(); g_snap(s0); }
      const double u0 = MPI_Wtime();
      int64_t n2 = 0;
      CUPP(cupaulipropPauliExpansionGetNumTerms(handle, expIn, &n2));
      cupaulipropPauliExpansionView_t v2;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, expIn, 0, n2, &v2));
      cupaulipropStatus_t sq = cupaulipropPauliExpansionViewPrepareDeduplication(
        handle, v2, CUPAULIPROP_SORT_ORDER_NONE, wsMem, ws);
      int64_t w2 = 0;
      if (sq == CUPAULIPROP_STATUS_SUCCESS)
        CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE,
                                               CUPAULIPROP_WORKSPACE_SCRATCH, &w2));
      if (sq != CUPAULIPROP_STATUS_SUCCESS || w2 > (int64_t)wsMem) {
        fprintf(stderr, "[rank %d] dedup prepare failed (status %d) at gate %d: local nin=%ld reqWs=%ld/%zu\n",
                rank, (int)sq, gi, (long)n2, (long)w2, wsMem);
        flushRecs(true);
        die("dedup prepare", __LINE__);
      }
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE,
                                         CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      const double u1 = MPI_Wtime();
      cupaulipropStatus_t sd = cupaulipropPauliExpansionViewExecuteDeduplication(
        handle, v2, expOut, CUPAULIPROP_SORT_ORDER_NONE, ws, stream);
      if (sd != CUPAULIPROP_STATUS_SUCCESS) {
        fprintf(stderr, "[rank %d] deduplication failed with status %d at gate %d, local nin=%ld\n",
                rank, (int)sd, gi, (long)n2);
        flushRecs(true);
        die("deduplication", __LINE__);
      }
      CUDA(cudaStreamSynchronize(stream));
      const double u2 = MPI_Wtime();
      if (g_snap) g_snap(s1);
      int64_t n3 = 0;
      CUPP(cupaulipropPauliExpansionGetNumTerms(handle, expOut, &n3));
      d.nin = n2; d.nout = n3; d.tStart = u0 - tBegin; d.tPrep = u1 - u0; d.tComp = u2 - u1; d.reqWs = w2;
      for (int k = 0; k < np; k++) d.prof[k] = s1[k] - s0[k];
      recs.push_back(d);
      CUPP(cupaulipropDestroyPauliExpansionView(v2));
      std::swap(expIn, expOut);
    }

    // Explicit truncation on globally deduplicated terms (keepDup mode only).
    if (o.keepDup && doTrunc) {
      Rec d; memset(&d, 0, sizeof(d));
      d.gate = gi; d.kind = REC_TRUNC;
      if (g_mark) { g_mark(); g_snap(s0); }
      const double u0 = MPI_Wtime();
      int64_t n2 = 0;
      CUPP(cupaulipropPauliExpansionGetNumTerms(handle, expIn, &n2));
      cupaulipropPauliExpansionView_t v2;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, expIn, 0, n2, &v2));
      CUPP(cupaulipropPauliExpansionViewPrepareTruncation(
        handle, v2, (int32_t)trunc.size(), trunc.data(), wsMem, ws));
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE,
                                         CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      const double u1 = MPI_Wtime();
      CUPP(cupaulipropPauliExpansionViewExecuteTruncation(
        handle, v2, expOut, (int32_t)trunc.size(), trunc.data(), ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      const double u2 = MPI_Wtime();
      if (g_snap) g_snap(s1);
      int64_t n3 = 0;
      CUPP(cupaulipropPauliExpansionGetNumTerms(handle, expOut, &n3));
      d.nin = n2; d.nout = n3; d.tStart = u0 - tBegin; d.tPrep = u1 - u0; d.tComp = u2 - u1;
      for (int k = 0; k < np; k++) d.prof[k] = s1[k] - s0[k];
      recs.push_back(d);
      CUPP(cupaulipropDestroyPauliExpansionView(v2));
      std::swap(expIn, expOut);
    }
  }

  // ---- trace with |0...0> ----
  Rec tr; memset(&tr, 0, sizeof(tr));
  tr.gate = -1; tr.kind = REC_TRACE;
  if (g_mark) { g_mark(); g_snap(s0); }
  const double v0 = MPI_Wtime();
  int64_t nfin = 0;
  CUPP(cupaulipropPauliExpansionGetNumTerms(handle, expIn, &nfin));
  maxLocal = std::max<long>(maxLocal, nfin);
  cupaulipropPauliExpansionView_t fview;
  CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, expIn, 0, nfin, &fview));
  CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, fview, wsMem, ws));
  CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE,
                                     CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
  double sig = 0, expo = 0;
  CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, fview, &sig, &expo, ws, stream));
  CUDA(cudaStreamSynchronize(stream));
  const double v1 = MPI_Wtime();
  if (g_snap) g_snap(s1);
  tr.nin = nfin; tr.nout = nfin; tr.tStart = v0 - tBegin; tr.tComp = v1 - v0;
  for (int k = 0; k < np; k++) tr.prof[k] = s1[k] - s0[k];
  recs.push_back(tr);

  MPI_Barrier(MPI_COMM_WORLD);
  const double tEnd = MPI_Wtime();

  // ---- summary ----
  long nfinGlobal = 0, maxLocalGlobal = 0, nfinL = nfin;
  MPI_Allreduce(&nfinL, &nfinGlobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&maxLocal, &maxLocalGlobal, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  double loc[MAXP + 2] = {0}, glob[MAXP + 2] = {0};
  for (const Rec& r : recs) {
    for (int k = 0; k < np; k++) loc[k] += r.prof[k];
    loc[MAXP] += r.tPrep; loc[MAXP + 1] += r.tComp;
  }
  MPI_Allreduce(loc, glob, MAXP + 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    const double wall = tEnd - tBegin;
    printf("expectation=%.12g\n", sig * std::pow(2.0, expo));
    printf("gates_applied=%d final_global_terms=%ld max_local_terms=%ld wall=%.4f s\n",
           applied, nfinGlobal, maxLocalGlobal, wall);
    printf("mean per rank: prepare=%.4f s compute=%.4f s\n", glob[MAXP] / numRanks, glob[MAXP + 1] / numRanks);
    double mpiTot = 0;
    for (int k = 0; k < g_nk; k++) mpiTot += glob[g_nk + k];
    for (int k = 0; k < g_nk; k++) {
      if (glob[k] == 0) continue;
      printf("  mpi %-10s calls/rank=%.0f time/rank=%.4f s (%.1f%% of wall) bytes/rank=%.3e\n",
             KIND_NAMES[k], glob[k] / numRanks, glob[g_nk + k] / numRanks,
             100.0 * glob[g_nk + k] / numRanks / wall, glob[2 * g_nk + k] / numRanks);
    }
    if (g_nk) printf("mpi total time/rank=%.4f s (%.1f%% of wall)\n", mpiTot / numRanks, 100.0 * mpiTot / numRanks / wall);
    fflush(stdout);
  }

  flushRecs(true);
  fclose(csv);

  CUPP(cupaulipropDestroyPauliExpansionView(fview));
  for (auto& g : circuit) CUPP(cupaulipropDestroyOperator(g.op));
  CUPP(cupaulipropDestroyWorkspaceDescriptor(ws));
  CUPP(cupaulipropDestroyPauliExpansion(expIn));
  CUPP(cupaulipropDestroyPauliExpansion(expOut));
  CUDA(cudaFree(dWs));
  for (int i = 0; i < 2; i++) { CUDA(cudaFree(dPauli[i])); CUDA(cudaFree(dCoef[i])); }
  CUPP(cupaulipropDestroy(handle));
  if (cuppComm != MPI_COMM_NULL) MPI_Comm_free(&cuppComm);
  MPI_Finalize();
  return 0;
}
