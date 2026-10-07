// Unified engine for partitioned Pauli propagation.
//
// One MPI process per node, one worker thread per GPU. The operator lives in host memory as
// buckets; a bucket is the set of terms that share the values of g parity rows. For the
// current window of operations the rows are chosen so that every branching rotation leaves
// them unchanged, which makes the buckets independent tasks: a worker loads a bucket, applies
// the whole window on its GPU with local deduplication and truncation, partitions the result
// by the next window's rows and appends the pieces to the next pool.
//
//   windows   found at run time: rotations are added while the rows that survive them
//             (the GF(2) annihilator of their generators) still have enough freedom
//   rows      random elements of that annihilator, the best-balanced set on a sample of terms
//   overflow  a bucket that outgrows its GPU is split by one more row and requeued
//   frame     Clifford operations are deferred and folded into later rotations
//   restart   the pool at a window boundary is a checkpoint
//   nodes     bucket d belongs to node d mod (number of nodes); after a window every node sends the
//             pieces it produced for other nodes' buckets from host memory, in whole-bucket messages.
//             GPUs never send to each other and no MPI call happens while a window runs.
//
// --mode ref runs the same operation list as one bucket on one GPU (plain cuPauliProp).

#include <cupauliprop.h>
#include <cuda_runtime.h>
#include <mpi.h>

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

#include "circuit.hpp"

extern "C" {
void pp_owners(const uint64_t* xz, int64_t n, int words, const uint64_t* masks, int g, uint8_t* owners);
void pp_counts_sorted(const uint8_t* owners, int64_t n, int R, int64_t* cnt);
int pp_partition(const uint64_t* xz, const double* coef, uint8_t* owners, int64_t n, int words,
                 uint64_t* xzOut, double* coefOut, int32_t* idx);
}

static int g_rank = 0, g_R = 1;
[[noreturn]] static void die(const char* what, int line) {
  fprintf(stderr, "[node %d] FATAL: %s (line %d)\n", g_rank, what, line);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
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
  std::string circuit = "ki";
  std::string mode = "engine";   // engine | ref
  int gpus = 1;
  int frame = 1;                 // defer Clifford operations
  // circuit parameters
  int steps = 20, lx = 5, ly = 6, n = 0, obsQ = -1, seed = 7, layers = 4;
  double theta = 0.25, phi = -0.5, jxy = 0.1, jz = 0.15, noise = 0.0, amp = 0.15;
  // truncation
  double coef = 1e-4;
  int weight = 0, truncEvery = 10;
  double junk = 1e-13;           // dropped after a Clifford-angle rotation applied to the data
  // engine
  double vramFrac = 0.9, expShare = 0.2;
  long target = 4000000;         // desired terms per bucket at the start of a window
  int gmax = 8, cands = 8, annMin = -1;
  long sample = 200000;
  std::string ckptFile, resume;
  int ckptWindow = -1, stopAtCkpt = 0;
};

struct Bucket { std::vector<uint64_t> xz; std::vector<double> cf; };
struct Work { Bucket b; int startOp; };

struct Window {
  int b = 0, e = 0;                 // operations [b, e) in processing order
  int g = 0;                        // 2^g buckets
  std::vector<uint64_t> rows;       // g masks, valid for strings at position b
  std::vector<BV> spare;            // further rows of the same window, for splitting
  int annDim = 0, generators = 0;
};

static Layout g_L;
static std::vector<Op> g_ops;       // operations in processing order

static int ownerCpu(const uint64_t* t, const std::vector<uint64_t>& rows, int g) {
  int ow = 0;
  for (int j = 0; j < g; j++) ow |= dotParity(t, &rows[(size_t)j * g_L.words()], g_L.words()) << j;
  return ow;
}

// Finds the window that starts at b and the rows that survive it.
//   sample: strings as they will be at position b (may be empty); gWanted: desired log2(buckets).
static Window planWindow(int b, int gWanted, const std::vector<uint64_t>& sample, const Opts& o, std::mt19937_64& rng) {
  const int W = g_L.words(), M = (int)g_ops.size();
  const int annMin = std::min(2 * g_L.n - 1, std::max(o.annMin < 0 ? g_L.n : o.annMin, 1));
  Window w; w.b = b;
  Span V; V.L = g_L;
  std::vector<int> maps;            // operations in the window that change bits without branching
  int k = b;
  for (; k < M; k++) {
    const Op& op = g_ops[k];
    if (op.type == OP_ROT && op.branches) {
      BV v = op.gen;                // generator as seen from position b
      for (int j = (int)maps.size() - 1; j >= 0; --j) mapVec(g_ops[maps[j]], v.data(), g_L);
      BV t = v;
      if (V.reduce(t)) {
        if (V.dim() + 1 > 2 * g_L.n - annMin) break;
        V.add(v);
      }
      w.generators++;
    } else if (changesBits(op)) {
      maps.push_back(k);
    }
  }
  w.e = k;
  std::vector<BV> ann = V.annihilator();
  w.annDim = (int)ann.size();
  w.g = std::max(0, std::min(std::min(gWanted, o.gmax), w.annDim));

  auto randomRow = [&]() {
    BV r(W, 0);
    bool any = false;
    while (!any)
      for (auto& a : ann) if (rng() & 1ULL) { any = true; for (int x = 0; x < W; x++) r[x] ^= a[x]; }
    return r;
  };
  const size_t ns = sample.size() / W;
  double bestMax = 1e300;
  for (int c = 0; c < (w.g && ns ? o.cands : 1); c++) {
    std::vector<uint64_t> rows((size_t)std::max(w.g, 1) * W, 0);
    for (int j = 0; j < w.g; j++) { BV r = randomRow(); std::copy(r.begin(), r.end(), rows.begin() + (size_t)j * W); }
    double mx = 0;
    if (w.g && ns) {
      std::vector<long> h((size_t)1 << w.g, 0);
      for (size_t i = 0; i < ns; i++) h[ownerCpu(&sample[i * W], rows, w.g)]++;
      mx = (double)*std::max_element(h.begin(), h.end());
    }
    if (mx < bestMax) { bestMax = mx; w.rows = rows; }
  }
  if (!ann.empty()) for (int i = 0; i < 32; i++) w.spare.push_back(randomRow());
  return w;
}

// Rank 0 plans the window from the samples of all nodes and broadcasts it.
static Window planShared(int b, int gWanted, std::vector<uint64_t>& localSample, const Opts& o, std::mt19937_64& rng) {
  const int W = g_L.words();
  std::vector<uint64_t> sample;
  if (g_R > 1) {
    int mine = (int)localSample.size();
    std::vector<int> cnt(g_R, 0), dsp(g_R, 0);
    MPI_Gather(&mine, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (g_rank == 0) { int tot = 0; for (int r = 0; r < g_R; r++) { dsp[r] = tot; tot += cnt[r]; } sample.resize(tot); }
    MPI_Gatherv(localSample.data(), mine, MPI_UINT64_T, sample.data(), cnt.data(), dsp.data(), MPI_UINT64_T, 0, MPI_COMM_WORLD);
  } else {
    sample.swap(localSample);
  }
  Window w;
  int hdr[6] = {0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> payload;
  if (g_rank == 0) {
    w = planWindow(b, gWanted, sample, o, rng);
    hdr[0] = w.b; hdr[1] = w.e; hdr[2] = w.g; hdr[3] = w.annDim; hdr[4] = w.generators; hdr[5] = (int)w.spare.size();
    payload = w.rows;
    payload.resize((size_t)std::max(w.g, 1) * W, 0);
    for (auto& r : w.spare) payload.insert(payload.end(), r.begin(), r.end());
  }
  if (g_R > 1) {
    MPI_Bcast(hdr, 6, MPI_INT, 0, MPI_COMM_WORLD);
    const int len = (std::max(hdr[2], 1) + hdr[5]) * W;
    payload.resize(len);
    MPI_Bcast(payload.data(), len, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (g_rank != 0) {
      w.b = hdr[0]; w.e = hdr[1]; w.g = hdr[2]; w.annDim = hdr[3]; w.generators = hdr[4];
      const size_t nr = (size_t)std::max(w.g, 1) * W;
      w.rows.assign(payload.begin(), payload.begin() + nr);
      for (int i = 0; i < hdr[5]; i++) w.spare.emplace_back(payload.begin() + nr + (size_t)i * W, payload.begin() + nr + (size_t)(i + 1) * W);
    }
  }
  return w;
}

// After a window: bucket d lives on node d mod g_R. Every node sends the pieces it produced for
// buckets of other nodes, straight from host memory, one pair of messages per bucket.
static void exchangeBuckets(std::vector<Bucket>& nb, double& seconds, double& bytes) {
  if (g_R == 1) return;
  const double t0 = MPI_Wtime();
  const int B = (int)nb.size(), W = g_L.words();
  std::vector<int64_t> mine(B), all((size_t)B * g_R);
  for (int d = 0; d < B; d++) mine[d] = (int64_t)nb[d].cf.size();
  MPI_Allgather(mine.data(), B, MPI_INT64_T, all.data(), B, MPI_INT64_T, MPI_COMM_WORLD);
  std::vector<MPI_Request> req;
  for (int d = 0; d < B; d++) {
    if (d % g_R != g_rank) continue;
    size_t total = nb[d].cf.size();
    for (int r = 0; r < g_R; r++) if (r != g_rank) total += (size_t)all[(size_t)r * B + d];
    size_t off = nb[d].cf.size();
    nb[d].xz.resize(total * W); nb[d].cf.resize(total);
    for (int r = 0; r < g_R; r++) {
      const int64_t c = all[(size_t)r * B + d];
      if (r == g_rank || c == 0) continue;
      if (c * W > 2000000000LL) die("bucket piece too large for one message", __LINE__);
      req.emplace_back(); MPI_Irecv(nb[d].xz.data() + off * W, (int)(c * W), MPI_UINT64_T, r, 2 * d, MPI_COMM_WORLD, &req.back());
      req.emplace_back(); MPI_Irecv(nb[d].cf.data() + off, (int)c, MPI_DOUBLE, r, 2 * d + 1, MPI_COMM_WORLD, &req.back());
      off += (size_t)c;
    }
  }
  for (int d = 0; d < B; d++) {
    if (d % g_R == g_rank || mine[d] == 0) continue;
    if (mine[d] * W > 2000000000LL) die("bucket piece too large for one message", __LINE__);
    req.emplace_back(); MPI_Isend(nb[d].xz.data(), (int)(mine[d] * W), MPI_UINT64_T, d % g_R, 2 * d, MPI_COMM_WORLD, &req.back());
    req.emplace_back(); MPI_Isend(nb[d].cf.data(), (int)mine[d], MPI_DOUBLE, d % g_R, 2 * d + 1, MPI_COMM_WORLD, &req.back());
    bytes += (double)mine[d] * (W * 8 + 8);
  }
  if (!req.empty()) MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
  for (int d = 0; d < B; d++)
    if (d % g_R != g_rank) { std::vector<uint64_t>().swap(nb[d].xz); std::vector<double>().swap(nb[d].cf); }
  seconds += MPI_Wtime() - t0;
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
  std::atomic<long> nout{0}, splits{0}, runs{0}, peak{0}, hostSplits{0};
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
  cupaulipropCoefficientTruncationParams_t coefParams, junkParams;
  cupaulipropPauliWeightTruncationParams_t weightParams;
  std::vector<cupaulipropTruncationStrategy_t> trunc, junk;
  std::vector<cupaulipropQuantumOperator_t> oper;
  std::mt19937_64 rng;
  double tOps = 0, tPart = 0, tH2D = 0, tD2H = 0, tIdle = 0;

  void init(int device, const Opts& opts) {
    dev = device; o = opts;
    CUDA(cudaSetDevice(dev));
    CUPP(cupaulipropCreate(&handle));
    xzPerTerm = g_L.words() * sizeof(uint64_t);
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
    coefParams.cutoff = o.coef; weightParams.cutoff = o.weight; junkParams.cutoff = o.junk;
    trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
    if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});
    junk.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &junkParams});
    rng.seed((uint64_t)o.seed * 7919ULL + dev);
    oper.resize(g_ops.size());
    for (size_t i = 0; i < g_ops.size(); i++) {
      const Op& p = g_ops[i];
      if (p.type == OP_ROT)
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, p.angle, (int32_t)p.qubits.size(), p.qubits.data(), p.paulis.data(), &oper[i]));
      else if (p.type == OP_CLIFF)
        CUPP(cupaulipropCreateCliffordGateOperator(handle, p.cliff, p.qubits.data(), &oper[i]));
      else
        CUPP(cupaulipropCreatePauliNoiseChannelOperator(handle, (int32_t)p.qubits.size(), p.qubits.data(), p.prob, &oper[i]));
    }
  }

  void makeExpansions(long nCur) {
    if (haveEx) { CUPP(cupaulipropDestroyPauliExpansion(ex[0])); CUPP(cupaulipropDestroyPauliExpansion(ex[1])); }
    cur = 0;
    CUPP(cupaulipropCreatePauliExpansion(handle, g_L.n, xz[0], xzMem, cf[0], cfMem,
                                         CUDA_R_64F, nCur, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[0]));
    CUPP(cupaulipropCreatePauliExpansion(handle, g_L.n, xz[1], xzMem, cf[1], cfMem,
                                         CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[1]));
    haveEx = true;
  }
  long numTerms() { int64_t n = 0; CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &n)); return (long)n; }

  // A spare row of the window, as a mask for strings at position k.
  BV splitRow(const Window& w, int k) {
    if (w.spare.empty()) die("window has no row left to split a bucket", __LINE__);
    BV r = w.spare[rng() % w.spare.size()];
    for (int j = w.b; j < k; j++) mapMask(g_ops[j], r.data(), g_L);
    return r;
  }

  void requeue(Shared& sh, Bucket& piece, int startOp) {
    { std::lock_guard<std::mutex> lk(sh.qm);
      sh.queue.push_front(Work());
      sh.queue.front().b.xz.swap(piece.xz); sh.queue.front().b.cf.swap(piece.cf); sh.queue.front().startOp = startOp; }
    sh.qcv.notify_all();
  }

  void run(const Window& w, const Window* next, Shared& sh, bool allowSplit) {
    CUDA(cudaSetDevice(dev));
    while (true) {
      Work wk;
      {
        const double ti = now();
        std::unique_lock<std::mutex> lk(sh.qm);
        sh.qcv.wait(lk, [&] { return !sh.queue.empty() || sh.active == 0; });
        tIdle += now() - ti;
        if (sh.queue.empty()) { lk.unlock(); sh.qcv.notify_all(); return; }
        wk.b.xz.swap(sh.queue.front().b.xz); wk.b.cf.swap(sh.queue.front().b.cf); wk.startOp = sh.queue.front().startOp;
        sh.queue.pop_front();
        sh.active++;
      }
      process(w, next, sh, wk, allowSplit);
      { std::lock_guard<std::mutex> lk(sh.qm); sh.active--; }
      sh.qcv.notify_all();
    }
  }

  void process(const Window& w, const Window* next, Shared& sh, Work& wk, bool allowSplit) {
    const int words = g_L.words();
    Bucket& b = wk.b;
    const long nb0 = (long)b.cf.size();
    if (nb0 > capTerms / 2) {
      // Too large to load with room to work: split on the host and requeue.
      if (!allowSplit) die("bucket does not fit the GPU (reference mode cannot split)", __LINE__);
      BV row = splitRow(w, wk.startOp);
      Bucket h[2];
      for (long i = 0; i < nb0; i++) {
        Bucket& d = h[dotParity(&b.xz[(size_t)i * words], row.data(), words)];
        d.xz.insert(d.xz.end(), b.xz.begin() + (size_t)i * words, b.xz.begin() + (size_t)(i + 1) * words);
        d.cf.push_back(b.cf[i]);
      }
      if (h[0].cf.empty() || h[1].cf.empty()) { static std::atomic<int> tries{0}; if (++tries > 100000) die("host split makes no progress", __LINE__); }
      for (int s = 0; s < 2; s++) if (!h[s].cf.empty()) requeue(sh, h[s], wk.startOp);
      sh.hostSplits++;
      return;
    }
    sh.runs++;
    double t0 = now();
    CUDA(cudaMemcpy(xz[0], b.xz.data(), (size_t)nb0 * xzPerTerm, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(cf[0], b.cf.data(), (size_t)nb0 * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<uint64_t>().swap(b.xz); std::vector<double>().swap(b.cf);
    makeExpansions(nb0);
    tH2D += now() - t0;

    t0 = now();
    long peak = nb0;
    for (int k = wk.startOp; k < w.e; k++) {
      const Op& op = g_ops[k];
      const long m = numTerms();
      if (m == 0) break;
      peak = std::max(peak, m);
      // Truncation: the scheduled one, or only numerical dust after a Clifford-angle rotation.
      const bool dust = (op.type == OP_ROT && !op.branches);
      const int32_t nT = op.trunc ? (int32_t)trunc.size() : (dust ? 1 : 0);
      const cupaulipropTruncationStrategy_t* pT = op.trunc ? trunc.data() : (dust ? junk.data() : nullptr);
      cupaulipropPauliExpansionView_t v;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
      int64_t rp = 0, rc = 0, need = 0;
      cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, v, oper[k], CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, wsMem, &rp, &rc, ws);
      if (sp == CUPAULIPROP_STATUS_SUCCESS)
        CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
      if (sp != CUPAULIPROP_STATUS_SUCCESS || rp > (int64_t)xzMem || rc > (int64_t)cfMem || need > (int64_t)wsMem) {
        // Outgrew this GPU: split by one more row of the window and requeue both halves.
        CUPP(cupaulipropDestroyPauliExpansionView(v));
        if (!allowSplit) { fprintf(stderr, "operation %d: %ld terms do not fit the GPU\n", k, m); die("capacity (reference mode cannot split)", __LINE__); }
        if (m < 1000) die("cannot split further", __LINE__);
        BV row = splitRow(w, k);
        int64_t c2[2] = {0, 0};
        pp_owners((const uint64_t*)xz[cur], m, words, row.data(), 1, dOwners);
        if (pp_partition((const uint64_t*)xz[cur], (const double*)cf[cur], dOwners, m, words,
                         (uint64_t*)xz[1 - cur], (double*)cf[1 - cur], dIdx) != 0) die("partition", __LINE__);
        pp_counts_sorted(dOwners, m, 2, c2);
        long off2 = 0;
        for (int h = 0; h < 2; h++) {
          if (c2[h] == 0) continue;
          Bucket piece;
          piece.xz.resize((size_t)c2[h] * words); piece.cf.resize((size_t)c2[h]);
          CUDA(cudaMemcpy(piece.xz.data(), (char*)xz[1 - cur] + off2 * xzPerTerm, (size_t)c2[h] * xzPerTerm, cudaMemcpyDeviceToHost));
          CUDA(cudaMemcpy(piece.cf.data(), (char*)cf[1 - cur] + off2 * sizeof(double), (size_t)c2[h] * sizeof(double), cudaMemcpyDeviceToHost));
          off2 += c2[h];
          requeue(sh, piece, k);
        }
        sh.splits++;
        tOps += now() - t0;
        return;
      }
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, v, ex[1 - cur], oper[k], 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      cur ^= 1;
    }
    long pk = sh.peak.load();
    while (peak > pk && !sh.peak.compare_exchange_weak(pk, peak)) {}
    const long m = numTerms();
    sh.nout += m;
    tOps += now() - t0;
    if (m == 0) return;

    if (!next) {                                      // last window: trace with |0...0>
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

    // Partition the output by the next window's rows; copy each piece straight to its bucket.
    t0 = now();
    const int gN = next->g;
    std::vector<int64_t> cnt((size_t)1 << gN, 0);
    const void *srcXz = xz[cur], *srcCf = cf[cur];
    if (gN > 0) {
      pp_owners((const uint64_t*)xz[cur], m, words, next->rows.data(), gN, dOwners);
      if (pp_partition((const uint64_t*)xz[cur], (const double*)cf[cur], dOwners, m, words,
                       (uint64_t*)xz[1 - cur], (double*)cf[1 - cur], dIdx) != 0) die("partition", __LINE__);
      pp_counts_sorted(dOwners, m, 1 << gN, cnt.data());
      srcXz = xz[1 - cur]; srcCf = cf[1 - cur];
    } else {
      cnt[0] = m;
    }
    CUDA(cudaDeviceSynchronize());
    tPart += now() - t0;
    t0 = now();
    long off = 0;
    for (size_t d = 0; d < cnt.size(); d++) {
      if (cnt[d] == 0) continue;
      std::lock_guard<std::mutex> lk(sh.bm[d]);
      Bucket& nb = sh.nextB[d];
      const size_t o1 = nb.xz.size(), o2 = nb.cf.size();
      nb.xz.resize(o1 + (size_t)cnt[d] * words); nb.cf.resize(o2 + (size_t)cnt[d]);
      CUDA(cudaMemcpy(nb.xz.data() + o1, (const char*)srcXz + off * xzPerTerm, (size_t)cnt[d] * xzPerTerm, cudaMemcpyDeviceToHost));
      CUDA(cudaMemcpy(nb.cf.data() + o2, (const char*)srcCf + off * sizeof(double), (size_t)cnt[d] * sizeof(double), cudaMemcpyDeviceToHost));
      off += cnt[d];
    }
    tD2H += now() - t0;
  }
};

static Problem buildProblem(const Opts& o) {
  const std::string& c = o.circuit;
  if (c == "ki")       return makeKickedIsingFrame(o.steps, o.theta, o.obsQ < 0 ? 62 : o.obsQ, o.truncEvery);
  if (c == "ki_lab")   return makeIsing("kicked Ising, Eagle heavy-hex", NUM_CIRCUIT_QUBITS, heavyHexEagle(), o.steps, o.theta, -0.5, 0.0, o.obsQ < 0 ? 62 : o.obsQ, o.truncEvery);
  if (c == "tfim_hh")  return makeIsing("TFIM Trotter, Eagle heavy-hex", NUM_CIRCUIT_QUBITS, heavyHexEagle(), o.steps, o.theta, o.phi, 0.0, o.obsQ < 0 ? 62 : o.obsQ, o.truncEvery);
  if (c == "tfim_sq")  return makeIsing("TFIM Trotter, square lattice", o.lx * o.ly, squareLattice(o.lx, o.ly), o.steps, o.theta, o.phi, 0.0, o.obsQ < 0 ? (o.ly / 2) * o.lx + o.lx / 2 : o.obsQ, o.truncEvery);
  if (c == "xxz")      return makeXXZ("XXZ Trotter, square lattice", o.lx * o.ly, squareLattice(o.lx, o.ly), o.steps, o.jxy, o.jz, o.obsQ < 0 ? (o.ly / 2) * o.lx + o.lx / 2 : o.obsQ, o.truncEvery);
  if (c == "qaoa")     return makeQaoa(o.n > 0 ? o.n : 100, o.layers, o.seed, o.truncEvery);
  if (c == "hea")      return makeHea(o.n > 0 ? o.n : 100, o.layers, o.seed, o.amp, o.obsQ < 0 ? (o.n > 0 ? o.n : 100) / 2 : o.obsQ, o.truncEvery);
  if (c == "ki_noisy") return makeIsing("noisy kicked Ising, Eagle heavy-hex", NUM_CIRCUIT_QUBITS, heavyHexEagle(), o.steps, o.theta, -0.5, o.noise > 0 ? o.noise : 0.005, o.obsQ < 0 ? 62 : o.obsQ, o.truncEvery);
  fprintf(stderr, "unknown circuit %s (ki, ki_lab, tfim_hh, tfim_sq, xxz, qaoa, hea, ki_noisy)\n", c.c_str());
  std::exit(1);
}

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_R);
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { if (i + 1 >= argc) die("missing option value", __LINE__); return argv[++i]; };
    if      (a == "--circuit")     o.circuit = next();
    else if (a == "--mode")        o.mode = next();
    else if (a == "--gpus")        o.gpus = atoi(next());
    else if (a == "--frame")       o.frame = atoi(next());
    else if (a == "--steps")       o.steps = atoi(next());
    else if (a == "--layers")      o.layers = atoi(next());
    else if (a == "--lx")          o.lx = atoi(next());
    else if (a == "--ly")          o.ly = atoi(next());
    else if (a == "--n")           o.n = atoi(next());
    else if (a == "--obs-qubit")   o.obsQ = atoi(next());
    else if (a == "--seed")        o.seed = atoi(next());
    else if (a == "--theta")       o.theta = atof(next());
    else if (a == "--phi")         o.phi = atof(next());
    else if (a == "--jxy")         o.jxy = atof(next());
    else if (a == "--jz")          o.jz = atof(next());
    else if (a == "--noise")       o.noise = atof(next());
    else if (a == "--amp")         o.amp = atof(next());
    else if (a == "--coef")        o.coef = atof(next());
    else if (a == "--weight")      o.weight = atoi(next());
    else if (a == "--trunc-every") o.truncEvery = atoi(next());
    else if (a == "--vram-frac")   o.vramFrac = atof(next());
    else if (a == "--exp-share")   o.expShare = atof(next());
    else if (a == "--target")      o.target = atol(next());
    else if (a == "--gmax")        o.gmax = atoi(next());
    else if (a == "--cands")       o.cands = atoi(next());
    else if (a == "--ann-min")     o.annMin = atoi(next());
    else if (a == "--ckpt-window") o.ckptWindow = atoi(next());
    else if (a == "--ckpt-file")   o.ckptFile = next();
    else if (a == "--stop-at-ckpt") o.stopAtCkpt = atoi(next());
    else if (a == "--resume")      o.resume = next();
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }
  const bool ref = (o.mode == "ref");
  if (ref) { o.gpus = 1; o.gmax = 0; if (g_R > 1) die("reference mode runs on one node", __LINE__); }
  if (g_R > 1 && (!o.resume.empty() || !o.ckptFile.empty())) die("checkpoints are single-node for now", __LINE__);
  if (o.gmax > 8) die("--gmax is at most 8", __LINE__);
  if (o.weight > 0) o.frame = 0;      // weight truncation needs the true strings
  int nd = 0;
  CUDA(cudaGetDeviceCount(&nd));
  if (o.gpus > nd) die("fewer GPUs visible than requested", __LINE__);

  Problem P = buildProblem(o);
  g_L = P.L;
  int deferred = 0;
  g_ops = compileForEngine(P, o.frame != 0, &deferred);
  const int M = (int)g_ops.size(), words = g_L.words();
  int nBranch = 0, nMap = 0, nNoise = 0;
  for (auto& p : g_ops) { if (p.type == OP_ROT && p.branches) nBranch++; else if (p.type == OP_NOISE) nNoise++; else nMap++; }
  if (g_rank == 0)
    printf("circuit: %s; %d qubits; %zu gates -> %d operations on the data (%d branching rotations, %d Clifford, %d noise); "
           "%d Clifford gates deferred by the frame\n", P.name.c_str(), g_L.n, P.ops.size(), M, nBranch, nMap, nNoise, deferred);

  std::vector<Worker> workers(o.gpus);
  for (int t = 0; t < o.gpus; t++) workers[t].init(t, o);
  if (g_rank == 0) {
    printf("mode=%s nodes=%d gpus/node=%d coef=%g weight=%d truncEvery=%d GPU capacity=%ld terms/buffer bucket target=%ld gmax=%d\n",
           o.mode.c_str(), g_R, o.gpus, o.coef, o.weight, o.truncEvery, workers[0].capTerms, o.target, o.gmax);
    fflush(stdout);
  }

  // Initial pool: the observable, one bucket.
  std::vector<Bucket> pool(1);
  if (g_rank == 0)
    for (auto& t : P.observable) { pool[0].xz.insert(pool[0].xz.end(), t.first.begin(), t.first.end()); pool[0].cf.push_back(t.second); }
  int pos = 0, firstWindow = 0;

  auto writeCheckpoint = [&](int position, int windowIndex) {
    FILE* f = fopen(o.ckptFile.c_str(), "wb");
    if (!f) die("cannot open checkpoint for writing", __LINE__);
    int32_t hdr[5] = {0x50504B44, words, position, M, windowIndex};
    int64_t nbk = (int64_t)pool.size(), total = 0;
    fwrite(hdr, sizeof(int32_t), 5, f); fwrite(&nbk, sizeof(int64_t), 1, f);
    for (auto& b : pool) {
      int64_t n = (int64_t)b.cf.size(); total += n;
      fwrite(&n, sizeof(int64_t), 1, f);
      fwrite(b.xz.data(), sizeof(uint64_t), (size_t)n * words, f);
      fwrite(b.cf.data(), sizeof(double), (size_t)n, f);
    }
    fclose(f);
    printf("checkpoint written at operation %d: %ld buckets, %ld terms\n", position, (long)nbk, (long)total);
  };
  if (!o.resume.empty()) {
    FILE* f = fopen(o.resume.c_str(), "rb");
    if (!f) die("cannot open checkpoint for reading", __LINE__);
    int32_t hdr[5]; int64_t nbk = 0;
    if (fread(hdr, sizeof(int32_t), 5, f) != 5 || hdr[0] != 0x50504B44 || hdr[1] != words || hdr[3] != M) die("checkpoint does not match this circuit", __LINE__);
    if (fread(&nbk, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
    pos = hdr[2]; firstWindow = hdr[4] + 1;
    pool.assign(1, Bucket());      // merged into one bucket; the engine re-splits as needed
    long total = 0;
    for (int64_t i = 0; i < nbk; i++) {
      int64_t n = 0;
      if (fread(&n, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
      const size_t o1 = pool[0].xz.size(), o2 = pool[0].cf.size();
      pool[0].xz.resize(o1 + (size_t)n * words); pool[0].cf.resize(o2 + (size_t)n);
      if (fread(pool[0].xz.data() + o1, sizeof(uint64_t), (size_t)n * words, f) != (size_t)n * words) die("bad checkpoint", __LINE__);
      if (fread(pool[0].cf.data() + o2, sizeof(double), (size_t)n, f) != (size_t)n) die("bad checkpoint", __LINE__);
      total += n;
    }
    fclose(f);
    printf("resumed at operation %d with %ld terms\n", pos, total);
  }

  Shared sh;
  std::mt19937_64 rng(o.seed);
  double tPlan = 0, growth = 2.0;
  long maxBoundary = 0, movedBuckets = 0;
  int windows = 0, maxWindowOps = 0, minAnn = 2 * g_L.n;
  double tExchange = 0, bytesMoved = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double tBegin = now();

  std::vector<uint64_t> noSample;
  Window w = planShared(pos, 0, noSample, o, rng);   // the pool is one bucket at the start
  for (int wi = firstWindow; ; wi++) {
    long NinLocal = 0, Nin = 0;
    for (auto& b : pool) NinLocal += (long)b.cf.size();
    MPI_Allreduce(&NinLocal, &Nin, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    maxBoundary = std::max(maxBoundary, Nin);
    const bool last = (w.e >= M);

    // Plan the next window: its rows are scored on a sample of the current terms, moved through the
    // bit-changing operations of this window (branching is ignored for the estimate).
    Window nextW;
    if (!last) {
      const double tp = now();
      int gWanted = 0;
      if (!ref) { const double est = Nin * growth; while (gWanted < o.gmax && est / (double)(1L << gWanted) > (double)o.target) gWanted++; }
      std::vector<uint64_t> sample;
      if (gWanted > 0) {
        const long stride = std::max<long>(1, Nin / std::max<long>(1, o.sample));
        for (auto& b : pool)
          for (size_t i = 0; i < b.cf.size(); i += stride) sample.insert(sample.end(), b.xz.begin() + i * words, b.xz.begin() + (i + 1) * words);
        for (int k = w.b; k < w.e; k++)
          if (changesBits(g_ops[k]))
            for (size_t i = 0; i < sample.size() / words; i++) mapVec(g_ops[k], &sample[i * words], g_L);
      }
      nextW = planShared(w.e, gWanted, sample, o, rng);
      tPlan += now() - tp;
    }
    windows++;
    maxWindowOps = std::max(maxWindowOps, w.e - w.b);
    minAnn = std::min(minAnn, w.annDim);

    sh.nextB.assign(last ? 1 : ((size_t)1 << nextW.g), Bucket());
    sh.nout = 0;
    std::sort(pool.begin(), pool.end(), [](const Bucket& a, const Bucket& b) { return a.cf.size() > b.cf.size(); });
    for (auto& b : pool)
      if (!b.cf.empty()) { sh.queue.push_back(Work()); sh.queue.back().b.xz.swap(b.xz); sh.queue.back().b.cf.swap(b.cf); sh.queue.back().startOp = w.b; }
    sh.active = 0;
    std::vector<std::thread> th;
    for (int t = 0; t < o.gpus; t++) th.emplace_back([&, t] { workers[t].run(w, last ? nullptr : &nextW, sh, !ref); });
    for (auto& x : th) x.join();

    long NoutLocal = sh.nout.load(), Nout = 0;
    MPI_Allreduce(&NoutLocal, &Nout, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (Nin > 0 && Nout > 0) growth = std::max(1.0, std::min(8.0, (double)Nout / (double)Nin * 1.3));
    if (last) break;
    exchangeBuckets(sh.nextB, tExchange, bytesMoved);
    pool.swap(sh.nextB);
    if (nextW.g > 0) movedBuckets++;
    pos = w.e;
    if (wi == o.ckptWindow && !o.ckptFile.empty()) {
      writeCheckpoint(pos, wi);
      if (o.stopAtCkpt) { printf("stopped at checkpoint\n"); MPI_Finalize(); return 0; }
    }
    w = nextW;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = now() - tBegin;

  double loc[9] = {0, 0, 0, 0, 0, tExchange, bytesMoved, sh.expec, 0}, sum[9];
  for (auto& x : workers) { loc[0] += x.tOps; loc[1] += x.tPart; loc[2] += x.tH2D; loc[3] += x.tD2H; loc[4] += x.tIdle; }
  MPI_Allreduce(loc, sum, 9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  long lc[4] = {sh.runs.load(), sh.splits.load(), sh.hostSplits.load(), sh.peak.load()}, sc[4], mc[4];
  MPI_Allreduce(lc, sc, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(lc, mc, 4, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  if (g_rank == 0) {
    printf("expectation=%.12g\n", sum[7]);
    printf("windows=%d (longest %d operations, smallest row space %d of %d) partitioned boundaries=%ld\n",
           windows, maxWindowOps, minAnn, 2 * g_L.n, movedBuckets);
    printf("max_terms_at_window_boundary=%ld largest_bucket_on_a_gpu=%ld bucket_runs=%ld splits_on_gpu=%ld splits_on_host=%ld\n",
           maxBoundary, mc[3], sc[0], sc[1], sc[2]);
    printf("wall=%.3f s; summed over %d GPU workers: operations=%.3f partition=%.3f host->gpu=%.3f gpu->host=%.3f idle=%.3f; planning=%.3f\n",
           wall, o.gpus * g_R, sum[0], sum[1], sum[2], sum[3], sum[4], tPlan);
    if (g_R > 1)
      printf("between nodes: %.3f GB sent, %.3f s per node on average (%.2f GB/s per node while exchanging); MPI calls while a window runs: 0\n",
             sum[6] / 1e9, sum[5] / g_R, sum[5] > 0 ? sum[6] / 1e9 / (sum[5] / g_R) / g_R : 0.0);
  }
  MPI_Finalize();
  return 0;
}
