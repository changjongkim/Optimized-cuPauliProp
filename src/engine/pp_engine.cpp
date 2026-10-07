// Unified engine for partitioned Pauli propagation.
//
// One MPI process per node, one worker thread per GPU. The operator lives in host memory as
// buckets; a bucket is the set of terms that share the values of g parity rows. For the
// current window of operations the rows are chosen so that every branching rotation leaves
// them unchanged, which makes the buckets independent tasks: a worker loads a bucket, applies
// the whole window on its GPU with local deduplication and truncation, partitions the result
// by the next window's rows and appends the pieces to the next pool.
//
//   windows   found at run time. Rotations are added while the rows that survive them (the GF(2)
//             annihilator of their generators) can still split the terms: either a fixed number of
//             free dimensions is kept (--window ann) or the terms must stay spread over the cosets
//             of the generator span, measured on a sample (--window coset)
//   pilot     buckets of one window are independent copies of the same dynamics in miniature, so one
//             bucket is run ahead through the longest allowed window and the window is ended at the
//             truncation after which that bucket is smallest: every term is moved at a re-key, so
//             the re-key is placed where there are fewest terms (--pilot; without it --snap ends
//             the window at the last truncation)
//   rows      random elements of that annihilator, the best-balanced set on a sample of terms
//   overflow  a bucket that outgrows its GPU is split by one more row and requeued
//   frame     Clifford operations are deferred and folded into later rotations
//   storage   buckets are lists of fixed-size host chunks that are pinned once and reused, for
//             GPU copies and for messages between nodes
//   nodes     bucket d belongs to node d mod (number of nodes); after a window every node sends the
//             pieces it produced for other nodes' buckets chunk by chunk. GPUs never send to each
//             other and no MPI call happens while a window runs
//   restart   the pool at a window boundary, with the window's rows, is a checkpoint
//   verify    parities that the whole circuit conserves are checked on every bucket
//
// --mode ref   same operation list as one bucket on one GPU (plain cuPauliProp)
// --mode cupp  same operation list through cuPauliProp's own distributed mode (one rank per GPU)

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
int64_t pp_count_ne(const uint8_t* owners, int64_t n, int expected);
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
  fprintf(stderr, "[node %d] cuPauliProp status %d\n", g_rank, (int)s_); die(#x, __LINE__); } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "[node %d] CUDA: %s\n", g_rank, cudaGetErrorString(e_)); die(#x, __LINE__); } } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Opts {
  std::string circuit = "ki";
  std::string mode = "engine";   // engine | ref | cupp
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
  std::string window = "ann";    // ann | coset
  double kappa = 4.0;            // coset rule: the largest coset may hold 1/(kappa * buckets) of the sample
  long sample = 200000;
  int snap = 1;                  // end a window right after a truncation, where the operator is locally smallest
  int pilot = 1;                 // run one bucket ahead through the window and end the window where it is smallest
  long chunk = 262144;           // terms per host chunk
  int pinned = 1, verify = 1;
  std::string ckptFile, resume;
  int ckptWindow = -1, stopAtCkpt = 0;
  std::string profile;           // write the number of terms before every operation to this file
};

static Layout g_L;
static std::vector<Op> g_ops;       // operations in processing order

// ---------------------------------------------------------------- host storage
// Buckets are lists of fixed-size chunks. Chunks are pinned once and recycled, so copies to and
// from the GPUs and messages between nodes always use memory that is already locked.

struct Chunk { uint64_t* xz = nullptr; double* cf = nullptr; long n = 0; };
struct Seg { uint64_t* xz; double* cf; long n; };

struct ChunkPool {
  std::mutex m;
  std::vector<Chunk> freeList;
  long cap = 0, allocated = 0;
  bool pinned = true;
  Chunk get() {
    {
      std::lock_guard<std::mutex> lk(m);
      if (!freeList.empty()) { Chunk c = freeList.back(); freeList.pop_back(); c.n = 0; return c; }
      allocated++;
    }
    const size_t bytes = (size_t)cap * (g_L.words() * sizeof(uint64_t) + sizeof(double));
    void* p = nullptr;
    if (pinned) CUDA(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
    else if (posix_memalign(&p, 4096, bytes)) die("host allocation failed", __LINE__);
    Chunk c;
    c.xz = (uint64_t*)p; c.cf = (double*)((char*)p + (size_t)cap * g_L.words() * sizeof(uint64_t));
    return c;
  }
  void put(Chunk c) { std::lock_guard<std::mutex> lk(m); c.n = 0; freeList.push_back(c); }
};
static ChunkPool g_pool;

struct Bucket { std::vector<Chunk> ch; long n = 0; };
struct Work { Bucket b; int startOp; };

// Reserves room for c terms at the tail of a bucket (caller holds the bucket's lock if shared).
static void reserveTail(Bucket& b, long c, std::vector<Seg>& out) {
  const int W = g_L.words();
  while (c > 0) {
    if (b.ch.empty() || b.ch.back().n == g_pool.cap) b.ch.push_back(g_pool.get());
    Chunk& k = b.ch.back();
    const long take = std::min(c, g_pool.cap - k.n);
    out.push_back({k.xz + (size_t)k.n * W, k.cf + k.n, take});
    k.n += take; b.n += take; c -= take;
  }
}
static void appendHost(Bucket& b, const uint64_t* xz, const double* cf, long c) {
  std::vector<Seg> segs;
  reserveTail(b, c, segs);
  const int W = g_L.words();
  long off = 0;
  for (auto& s : segs) {
    memcpy(s.xz, xz + (size_t)off * W, (size_t)s.n * W * sizeof(uint64_t));
    memcpy(s.cf, cf + off, (size_t)s.n * sizeof(double));
    off += s.n;
  }
}
static void releaseBucket(Bucket& b) { for (auto& c : b.ch) g_pool.put(c); b.ch.clear(); b.n = 0; }

// ---------------------------------------------------------------- windows

struct Window {
  int b = 0, e = 0;                 // operations [b, e) in processing order
  int g = 0;                        // 2^g buckets
  std::vector<uint64_t> rows;       // g masks, valid for strings at position b
  std::vector<BV> spare;            // further rows of the same window, for splitting
  int annDim = 0, generators = 0;
  double cosetFrac = 0;             // share of the sample in the largest coset of the generator span
};

static int ownerCpu(const uint64_t* t, const std::vector<uint64_t>& rows, int g) {
  int ow = 0;
  for (int j = 0; j < g; j++) ow |= dotParity(t, &rows[(size_t)j * g_L.words()], g_L.words()) << j;
  return ow;
}

// Share of the sample that falls into the largest coset of V. Buckets are unions of cosets, so this
// bounds how evenly any choice of rows can split the terms.
static double largestCosetShare(const std::vector<uint64_t>& sample, const Span& V, size_t maxTerms) {
  const int W = g_L.words();
  const size_t ns = sample.size() / W;
  if (!ns) return 0;
  const size_t stride = std::max<size_t>(1, ns / maxTerms);
  std::vector<uint64_t> key;
  BV t(W);
  for (size_t i = 0; i < ns; i += stride) {
    std::copy(sample.begin() + i * W, sample.begin() + (i + 1) * W, t.begin());
    V.reduce(t);
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int w = 0; w < W; w++) { h ^= t[w] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 31; }
    key.push_back(h);
  }
  std::sort(key.begin(), key.end());
  size_t best = 0, run = 0;
  for (size_t i = 0; i < key.size(); i++) { run = (i && key[i] == key[i - 1]) ? run + 1 : 1; best = std::max(best, run); }
  return (double)best / (double)key.size();
}

// Finds the window that starts at b and the rows that survive it.
//   sample: strings as they will be at position b (may be empty); gWanted: desired log2(buckets).
static Window planWindow(int b, int gWanted, const std::vector<uint64_t>& sample, const Opts& o, std::mt19937_64& rng) {
  const int W = g_L.words(), M = (int)g_ops.size(), n2 = 2 * g_L.n;
  gWanted = std::min(gWanted, o.gmax);
  const bool cosetRule = (o.window == "coset") && gWanted > 0 && !sample.empty();
  const int annMin = std::min(n2 - 1, std::max(o.annMin < 0 ? g_L.n : o.annMin, 1));
  const int dimCap = cosetRule ? n2 - (gWanted + 4) : n2 - annMin;

  // Generators of the branching rotations ahead, as seen from position b, up to the dimension cap.
  struct Gen { int k; BV v; bool fresh; };
  std::vector<Gen> gens;
  std::vector<int> maps;
  Span V; V.L = g_L;
  int k = b;
  for (; k < M; k++) {
    const Op& op = g_ops[k];
    if (op.type == OP_ROT && op.branches) {
      BV v = op.gen;
      for (int j = (int)maps.size() - 1; j >= 0; --j) mapVec(g_ops[maps[j]], v.data(), g_L);
      BV t = v;
      const bool fresh = V.reduce(t);
      if (fresh) { if (V.dim() + 1 > dimCap) break; V.add(v); }
      gens.push_back({k, v, fresh});
    } else if (changesBits(op)) {
      maps.push_back(k);
    }
  }
  int e = k;

  Window w; w.b = b;
  if (cosetRule) {
    // Longest prefix whose generator span still leaves the sample spread over many cosets. The share
    // of the largest coset only grows as generators are added, so the last passing prefix is kept.
    const double limit = 1.0 / (o.kappa * (double)(1 << gWanted));
    Span P; P.L = g_L;
    int passEnd = -1;                       // index into gens after which the window may end
    size_t i = 0;
    for (; i < gens.size(); i++) {
      if (gens[i].fresh) {
        P.add(gens[i].v);
        if (P.dim() % 8 == 0 || P.dim() == V.dim()) {
          if (largestCosetShare(sample, P, 20000) > limit) break;
          passEnd = (int)i;
        }
      }
    }
    if (i < gens.size()) {
      // Cut back to the last passing point (at least the first generator), ending before the next one.
      const int keep = std::max(passEnd, 0);
      e = (keep + 1 < (int)gens.size()) ? gens[keep + 1].k : e;
      gens.resize(keep + 1);
    }
    V = Span(); V.L = g_L;
    for (auto& gn : gens) V.add(gn.v);
    w.cosetFrac = largestCosetShare(sample, V, 20000);
    while (gWanted > 0 && w.cosetFrac > 1.0 / (double)(1 << gWanted)) gWanted--;   // cannot split finer than the cosets
  }
  w.e = e;
  w.generators = (int)gens.size();
  std::vector<BV> ann = V.annihilator();
  w.annDim = (int)ann.size();
  w.g = std::max(0, std::min(gWanted, w.annDim));

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

// Without a pilot: end the window right after the last truncation in its second half.
static int snapEnd(const Window& w) {
  const int M = (int)g_ops.size();
  if (w.e >= M) return w.e;
  for (int k = w.e - 1; k >= w.b + (w.e - w.b) / 2; --k) if (g_ops[k].trunc) return k + 1;
  return w.e;
}

static void packWindow(const Window& w, int hdr[6], std::vector<uint64_t>& payload) {
  const int W = g_L.words();
  hdr[0] = w.b; hdr[1] = w.e; hdr[2] = w.g; hdr[3] = w.annDim; hdr[4] = w.generators; hdr[5] = (int)w.spare.size();
  payload = w.rows;
  payload.resize((size_t)std::max(w.g, 1) * W, 0);
  for (auto& r : w.spare) payload.insert(payload.end(), r.begin(), r.end());
}
static Window unpackWindow(const int hdr[6], const std::vector<uint64_t>& payload) {
  const int W = g_L.words();
  Window w;
  w.b = hdr[0]; w.e = hdr[1]; w.g = hdr[2]; w.annDim = hdr[3]; w.generators = hdr[4];
  const size_t nr = (size_t)std::max(w.g, 1) * W;
  w.rows.assign(payload.begin(), payload.begin() + nr);
  for (int i = 0; i < hdr[5]; i++) w.spare.emplace_back(payload.begin() + nr + (size_t)i * W, payload.begin() + nr + (size_t)(i + 1) * W);
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
  double frac = 0;
  if (g_rank == 0) { w = planWindow(b, gWanted, sample, o, rng); packWindow(w, hdr, payload); frac = w.cosetFrac; }
  if (g_R > 1) {
    MPI_Bcast(hdr, 6, MPI_INT, 0, MPI_COMM_WORLD);
    const int len = (std::max(hdr[2], 1) + hdr[5]) * W;
    payload.resize(len);
    MPI_Bcast(payload.data(), len, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&frac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (g_rank != 0) { w = unpackWindow(hdr, payload); w.cosetFrac = frac; }
  }
  return w;
}

// After a window: bucket d lives on node d mod g_R. Every node sends its pieces of other nodes'
// buckets chunk by chunk (all chunks full except the last), and the owner adopts each message as a
// chunk of its bucket.
static void exchangeBuckets(std::vector<Bucket>& nb, double& seconds, double& bytes) {
  if (g_R == 1) return;
  const double t0 = MPI_Wtime();
  const int B = (int)nb.size(), W = g_L.words();
  std::vector<int64_t> mine(B), all((size_t)B * g_R);
  for (int d = 0; d < B; d++) mine[d] = (int64_t)nb[d].n;
  MPI_Allgather(mine.data(), B, MPI_INT64_T, all.data(), B, MPI_INT64_T, MPI_COMM_WORLD);
  std::vector<MPI_Request> req;
  for (int d = 0; d < B; d++) {
    if (d % g_R != g_rank) continue;
    for (int r = 0; r < g_R; r++) {
      int64_t c = all[(size_t)r * B + d];
      if (r == g_rank) continue;
      while (c > 0) {
        const long take = (long)std::min<int64_t>(c, g_pool.cap);
        Chunk k = g_pool.get();
        k.n = take;
        req.emplace_back(); MPI_Irecv(k.xz, (int)(take * W), MPI_UINT64_T, r, 2 * d, MPI_COMM_WORLD, &req.back());
        req.emplace_back(); MPI_Irecv(k.cf, (int)take, MPI_DOUBLE, r, 2 * d + 1, MPI_COMM_WORLD, &req.back());
        nb[d].ch.push_back(k); nb[d].n += take;
        c -= take;
      }
    }
  }
  for (int d = 0; d < B; d++) {
    if (d % g_R == g_rank) continue;
    for (auto& k : nb[d].ch) {
      if (k.n == 0) continue;
      req.emplace_back(); MPI_Isend(k.xz, (int)(k.n * W), MPI_UINT64_T, d % g_R, 2 * d, MPI_COMM_WORLD, &req.back());
      req.emplace_back(); MPI_Isend(k.cf, (int)k.n, MPI_DOUBLE, d % g_R, 2 * d + 1, MPI_COMM_WORLD, &req.back());
      bytes += (double)k.n * (W * 8 + 8);
    }
  }
  if (!req.empty()) MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
  for (int d = 0; d < B; d++) if (d % g_R != g_rank) releaseBucket(nb[d]);
  seconds += MPI_Wtime() - t0;
}

// ---------------------------------------------------------------- workers

struct Shared {
  std::mutex qm;
  std::condition_variable qcv;
  std::deque<Work> queue;
  int active = 0;
  std::vector<Bucket> nextB;
  std::vector<std::mutex> bm{256};
  std::mutex em;
  double expec = 0;
  std::atomic<long> nout{0}, splits{0}, runs{0}, peak{0}, hostSplits{0}, verified{0}, violations{0};
  // conserved rows as masks for strings at the start of the current window, and their expected values
  std::vector<uint64_t> checkRows;
  int checkG = 0, checkExpected = 0;
  std::atomic<long>* popAt = nullptr;   // terms seen before each operation, summed over buckets
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
    rng.seed((uint64_t)o.seed * 7919ULL + dev + 131ULL * g_rank);
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
      sh.queue.front().b.ch.swap(piece.ch); sh.queue.front().b.n = piece.n; sh.queue.front().startOp = startOp; }
    piece.n = 0;
    sh.qcv.notify_all();
  }

  // Copies m terms from device arrays into a bucket's reserved tail.
  void toHost(const void* srcXz, const void* srcCf, long off, const std::vector<Seg>& segs) {
    for (auto& s : segs) {
      CUDA(cudaMemcpy(s.xz, (const char*)srcXz + off * xzPerTerm, (size_t)s.n * xzPerTerm, cudaMemcpyDeviceToHost));
      CUDA(cudaMemcpy(s.cf, (const char*)srcCf + off * sizeof(double), (size_t)s.n * sizeof(double), cudaMemcpyDeviceToHost));
      off += s.n;
    }
  }

  // Runs a copy of the bucket through the window and records the number of terms after every
  // operation. Returns how many operations it completed (it stops if the bucket outgrows the GPU).
  int pilot(const Window& w, const Bucket& b, std::vector<double>& after) {
    CUDA(cudaSetDevice(dev));
    if (b.n == 0 || b.n > capTerms / 2) return 0;
    long off = 0;
    for (auto& k : b.ch) {
      if (k.n == 0) continue;
      CUDA(cudaMemcpy((char*)xz[0] + off * xzPerTerm, k.xz, (size_t)k.n * xzPerTerm, cudaMemcpyHostToDevice));
      CUDA(cudaMemcpy((char*)cf[0] + off * sizeof(double), k.cf, (size_t)k.n * sizeof(double), cudaMemcpyHostToDevice));
      off += k.n;
    }
    makeExpansions(b.n);
    for (int k = w.b; k < w.e; k++) {
      const Op& op = g_ops[k];
      const long m = numTerms();
      if (m == 0) return w.e - w.b;
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
        CUPP(cupaulipropDestroyPauliExpansionView(v));
        return k - w.b;
      }
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, v, ex[1 - cur], oper[k], 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      cur ^= 1;
      after[k - w.b] = (double)numTerms();
    }
    return w.e - w.b;
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
        wk.b.ch.swap(sh.queue.front().b.ch); wk.b.n = sh.queue.front().b.n; wk.startOp = sh.queue.front().startOp;
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
    const long nb0 = b.n;
    if (nb0 > capTerms / 2) {
      // Too large to load with room to work: split on the host and requeue.
      if (!allowSplit) die("bucket does not fit the GPU (this mode cannot split)", __LINE__);
      BV row = splitRow(w, wk.startOp);
      Bucket h[2];
      for (auto& k : b.ch)
        for (long i = 0; i < k.n; i++)
          appendHost(h[dotParity(k.xz + (size_t)i * words, row.data(), words)], k.xz + (size_t)i * words, k.cf + i, 1);
      releaseBucket(b);
      for (int s = 0; s < 2; s++) if (h[s].n > 0) requeue(sh, h[s], wk.startOp);
      sh.hostSplits++;
      return;
    }
    sh.runs++;
    double t0 = now();
    long off = 0;
    for (auto& k : b.ch) {
      if (k.n == 0) continue;
      CUDA(cudaMemcpy((char*)xz[0] + off * xzPerTerm, k.xz, (size_t)k.n * xzPerTerm, cudaMemcpyHostToDevice));
      CUDA(cudaMemcpy((char*)cf[0] + off * sizeof(double), k.cf, (size_t)k.n * sizeof(double), cudaMemcpyHostToDevice));
      off += k.n;
    }
    releaseBucket(b);
    makeExpansions(nb0);
    tH2D += now() - t0;

    // Integrity: every term must carry the expected values of the parities the circuit conserves.
    if (sh.checkG > 0 && wk.startOp == w.b) {
      pp_owners((const uint64_t*)xz[0], nb0, words, sh.checkRows.data(), sh.checkG, dOwners);
      sh.violations += pp_count_ne(dOwners, nb0, sh.checkExpected);
      sh.verified += nb0;
    }

    t0 = now();
    long peak = nb0;
    for (int k = wk.startOp; k < w.e; k++) {
      const Op& op = g_ops[k];
      const long m = numTerms();
      if (m == 0) break;
      peak = std::max(peak, m);
      if (sh.popAt) sh.popAt[k] += m;
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
        if (!allowSplit) { fprintf(stderr, "operation %d: %ld terms do not fit the GPU\n", k, m); die("capacity (this mode cannot split)", __LINE__); }
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
          std::vector<Seg> segs;
          reserveTail(piece, c2[h], segs);
          toHost(xz[1 - cur], cf[1 - cur], off2, segs);
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
    long o2 = 0;
    for (size_t d = 0; d < cnt.size(); d++) {
      if (cnt[d] == 0) continue;
      std::vector<Seg> segs;
      { std::lock_guard<std::mutex> lk(sh.bm[d]); reserveTail(sh.nextB[d], cnt[d], segs); }
      toHost(srcXz, srcCf, o2, segs);       // the reserved ranges are private to this worker
      o2 += cnt[d];
    }
    tD2H += now() - t0;
  }
};

// ---------------------------------------------------------------- circuits

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

// ---------------------------------------------------------------- cuPauliProp's distributed mode
// The same operation list through the library's own multi-process execution: one rank per GPU,
// global deduplication inside every operator application. Needs CUPAULIPROP_COMM_LIB.
static int runCuppDistributed(const Opts& o, const Problem& P) {
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  CUPP(cupaulipropCreate(&handle));
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  if (!getenv("CUPAULIPROP_COMM_LIB")) die("CUPAULIPROP_COMM_LIB is not set", __LINE__);
  CUPP(cupaulipropResetDistributedConfiguration(handle, CUPAULIPROP_DISTRIBUTED_PROVIDER_MPI, &comm, sizeof(comm)));
  const size_t xzPerTerm = g_L.words() * sizeof(uint64_t), perTerm = xzPerTerm + sizeof(double);
  size_t freeMem = 0, totalMem = 0;
  CUDA(cudaMemGetInfo(&freeMem, &totalMem));
  const size_t usable = (size_t)(freeMem * o.vramFrac);
  const size_t capTerms = (size_t)(usable * o.expShare) / perTerm;
  const size_t xzMem = capTerms * xzPerTerm, cfMem = capTerms * sizeof(double), wsMem = usable - 2 * (xzMem + cfMem);
  void *xz[2], *cf[2], *dWs;
  for (int b = 0; b < 2; b++) { CUDA(cudaMalloc(&xz[b], xzMem)); CUDA(cudaMalloc(&cf[b], cfMem)); }
  CUDA(cudaMalloc(&dWs, wsMem));
  long n0 = 0;
  if (g_rank == 0) {
    std::vector<uint64_t> hx; std::vector<double> hc;
    for (auto& t : P.observable) { hx.insert(hx.end(), t.first.begin(), t.first.end()); hc.push_back(t.second); }
    n0 = (long)hc.size();
    CUDA(cudaMemcpy(xz[0], hx.data(), hx.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(cf[0], hc.data(), hc.size() * sizeof(double), cudaMemcpyHostToDevice));
  }
  cupaulipropPauliExpansion_t ex[2];
  CUPP(cupaulipropCreatePauliExpansion(handle, g_L.n, xz[0], xzMem, cf[0], cfMem, CUDA_R_64F, n0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[0]));
  CUPP(cupaulipropCreatePauliExpansion(handle, g_L.n, xz[1], xzMem, cf[1], cfMem, CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[1]));
  cupaulipropWorkspaceDescriptor_t ws;
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));
  cupaulipropCoefficientTruncationParams_t coefParams, junkParams; coefParams.cutoff = o.coef; junkParams.cutoff = o.junk;
  cupaulipropPauliWeightTruncationParams_t weightParams; weightParams.cutoff = o.weight;
  std::vector<cupaulipropTruncationStrategy_t> trunc, junk;
  trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
  if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});
  junk.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &junkParams});
  std::vector<cupaulipropQuantumOperator_t> oper(g_ops.size());
  for (size_t i = 0; i < g_ops.size(); i++) {
    const Op& p = g_ops[i];
    if (p.type == OP_ROT) CUPP(cupaulipropCreatePauliRotationGateOperator(handle, p.angle, (int32_t)p.qubits.size(), p.qubits.data(), p.paulis.data(), &oper[i]));
    else if (p.type == OP_CLIFF) CUPP(cupaulipropCreateCliffordGateOperator(handle, p.cliff, p.qubits.data(), &oper[i]));
    else CUPP(cupaulipropCreatePauliNoiseChannelOperator(handle, (int32_t)p.qubits.size(), p.qubits.data(), p.prob, &oper[i]));
  }
  if (g_rank == 0) { printf("mode=cupp ranks=%d (one GPU each) coef=%g weight=%d\n", g_R, o.coef, o.weight); fflush(stdout); }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  int cur = 0;
  long maxLocal = 0;
  for (size_t k = 0; k < g_ops.size(); k++) {
    const Op& op = g_ops[k];
    int64_t m = 0;
    CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &m));
    maxLocal = std::max<long>(maxLocal, m);
    const bool dust = (op.type == OP_ROT && !op.branches);
    const int32_t nT = op.trunc ? (int32_t)trunc.size() : (dust ? 1 : 0);
    const cupaulipropTruncationStrategy_t* pT = op.trunc ? trunc.data() : (dust ? junk.data() : nullptr);
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
    int64_t rp = 0, rc = 0;
    cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorApplication(
      handle, v, oper[k], CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, wsMem, &rp, &rc, ws);
    if (sp != CUPAULIPROP_STATUS_SUCCESS || rp > (int64_t)xzMem || rc > (int64_t)cfMem) {
      fprintf(stderr, "[rank %d] cuPauliProp distributed mode ran out of memory at operation %zu of %zu with %ld local terms (status %d)\n",
              g_rank, k, g_ops.size(), (long)m, (int)sp);
      die("cuPauliProp distributed capacity", __LINE__);
    }
    CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
    cupaulipropStatus_t sc = cupaulipropPauliExpansionViewComputeOperatorApplication(
      handle, v, ex[1 - cur], oper[k], 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, ws, stream);
    if (sc != CUPAULIPROP_STATUS_SUCCESS) {
      fprintf(stderr, "[rank %d] cuPauliProp distributed mode failed at operation %zu with %ld local terms (status %d)\n", g_rank, k, (long)m, (int)sc);
      die("cuPauliProp distributed application", __LINE__);
    }
    CUDA(cudaStreamSynchronize(stream));
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    cur ^= 1;
  }
  int64_t m = 0;
  CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &m));
  cupaulipropPauliExpansionView_t v;
  CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
  CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, wsMem, ws));
  CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
  double sig = 0, e2 = 0;
  CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &e2, ws, stream));
  CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = MPI_Wtime() - t0;
  long mx = 0;
  MPI_Allreduce(&maxLocal, &mx, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  if (g_rank == 0) {
    printf("expectation=%.12g\n", sig * std::pow(2.0, e2));
    printf("largest_local_expansion=%ld terms\nwall=%.3f s\n", mx, wall);
  }
  MPI_Finalize();
  return 0;
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
  // cuPauliProp's distributed mode hands GPU pointers to MPI: Cray MPICH must see this rank's device
  // before MPI_Init.
  bool cupp = false;
  for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--mode") && !strcmp(argv[i + 1], "cupp")) cupp = true;
  if (cupp) {
    const char* lid = getenv("SLURM_LOCALID");
    int nd0 = 0;
    if (lid && cudaGetDeviceCount(&nd0) == cudaSuccess && nd0 > 0) {
      cudaSetDevice(atoi(lid) % nd0);
      void* warm = nullptr;
      if (cudaMalloc(&warm, 1 << 20) == cudaSuccess) cudaFree(warm);
    }
  }
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
    else if (a == "--window")      o.window = next();
    else if (a == "--kappa")       o.kappa = atof(next());
    else if (a == "--snap")        o.snap = atoi(next());
    else if (a == "--pilot")       o.pilot = atoi(next());
    else if (a == "--chunk")       o.chunk = atol(next());
    else if (a == "--pinned")      o.pinned = atoi(next());
    else if (a == "--verify")      o.verify = atoi(next());
    else if (a == "--ckpt-window") o.ckptWindow = atoi(next());
    else if (a == "--ckpt-file")   o.ckptFile = next();
    else if (a == "--stop-at-ckpt") o.stopAtCkpt = atoi(next());
    else if (a == "--resume")      o.resume = next();
    else if (a == "--profile")     o.profile = next();
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }
  const bool ref = (o.mode == "ref");
  if (ref) { o.gpus = 1; o.gmax = 0; if (g_R > 1) die("reference mode runs on one node", __LINE__); }
  if (o.gmax > 8) die("--gmax is at most 8", __LINE__);
  if (o.weight > 0) o.frame = 0;      // weight truncation needs the true strings

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
  if (o.mode == "cupp") return runCuppDistributed(o, P);

  int nd = 0;
  CUDA(cudaGetDeviceCount(&nd));
  if (o.gpus > nd) die("fewer GPUs visible than requested", __LINE__);
  CUDA(cudaSetDevice(0));
  g_pool.cap = o.chunk; g_pool.pinned = (o.pinned != 0);

  std::vector<Worker> workers(o.gpus);
  for (int t = 0; t < o.gpus; t++) workers[t].init(t, o);
  CUDA(cudaSetDevice(0));
  if (g_rank == 0) {
    printf("mode=%s nodes=%d gpus/node=%d coef=%g weight=%d truncEvery=%d GPU capacity=%ld terms/buffer bucket target=%ld gmax=%d window=%s host=%s\n",
           o.mode.c_str(), g_R, o.gpus, o.coef, o.weight, o.truncEvery, workers[0].capTerms, o.target, o.gmax, o.window.c_str(),
           o.pinned ? "pinned chunks" : "pageable chunks");
    fflush(stdout);
  }

  Shared sh;
  std::mt19937_64 rng(o.seed);
  std::vector<std::atomic<long>> popAt(o.profile.empty() ? 0 : M);
  std::vector<int> windowStarts;
  if (!o.profile.empty()) { for (auto& x : popAt) x = 0; sh.popAt = popAt.data(); }

  // Parities conserved by every branching rotation of the whole run, as masks for strings at
  // position 0. With a single-term observable every term must keep the observable's values.
  std::vector<BV> conserved;
  if (o.verify && P.observable.size() == 1) {
    Span all; all.L = g_L;
    std::vector<int> maps;
    for (int k = 0; k < M; k++) {
      const Op& op = g_ops[k];
      if (op.type == OP_ROT && op.branches) {
        BV v = op.gen;
        for (int j = (int)maps.size() - 1; j >= 0; --j) mapVec(g_ops[maps[j]], v.data(), g_L);
        all.add(v);
      } else if (changesBits(op)) maps.push_back(k);
    }
    conserved = all.annihilator();
    if (conserved.size() > 8) conserved.resize(8);
    sh.checkExpected = 0;
    for (size_t j = 0; j < conserved.size(); j++)
      sh.checkExpected |= dotParity(P.observable[0].first.data(), conserved[j].data(), words) << j;
  }
  int conservedAt = 0;      // position the masks in `conserved` currently refer to
  auto setCheckRows = [&](int position) {
    for (; conservedAt < position; conservedAt++)
      for (auto& r : conserved) mapMask(g_ops[conservedAt], r.data(), g_L);
    sh.checkG = (int)conserved.size();
    sh.checkRows.assign((size_t)std::max(sh.checkG, 1) * words, 0);
    for (int j = 0; j < sh.checkG; j++) std::copy(conserved[j].begin(), conserved[j].end(), sh.checkRows.begin() + (size_t)j * words);
  };

  // Initial pool: the observable, one bucket on node 0.
  std::vector<Bucket> pool(1);
  if (g_rank == 0)
    for (auto& t : P.observable) appendHost(pool[0], t.first.data(), &t.second, 1);
  int pos = 0, firstWindow = 0;
  Window w;
  bool haveWindow = false;

  // Checkpoint: every node writes its buckets and the window they were partitioned for.
  auto ckptName = [&](const std::string& base, int r) { return base + ".r" + std::to_string(r); };
  auto writeCheckpoint = [&](int position, int windowIndex, const Window& nw) {
    FILE* f = fopen(ckptName(o.ckptFile, g_rank).c_str(), "wb");
    if (!f) die("cannot open checkpoint for writing", __LINE__);
    int32_t hdr[6] = {0x50504B45, words, position, M, windowIndex, g_R};
    int wh[6]; std::vector<uint64_t> wp;
    packWindow(nw, wh, wp);
    int64_t wl = (int64_t)wp.size(), nbk = (int64_t)pool.size(), total = 0;
    fwrite(hdr, sizeof(int32_t), 6, f); fwrite(wh, sizeof(int), 6, f); fwrite(&wl, sizeof(int64_t), 1, f);
    fwrite(wp.data(), sizeof(uint64_t), wp.size(), f); fwrite(&nbk, sizeof(int64_t), 1, f);
    for (auto& b : pool) {
      int64_t n = b.n; total += n;
      fwrite(&n, sizeof(int64_t), 1, f);
      for (auto& k : b.ch) { fwrite(k.xz, sizeof(uint64_t), (size_t)k.n * words, f); }
      for (auto& k : b.ch) { fwrite(k.cf, sizeof(double), (size_t)k.n, f); }
    }
    fclose(f);
    long tl = total, tg = 0;
    MPI_Allreduce(&tl, &tg, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) printf("checkpoint written at operation %d: %ld terms on %d node(s)\n", position, tg, g_R);
  };
  if (!o.resume.empty()) {
    // Same node count: every node reloads its own buckets and continues the stored window.
    // One node: all files are merged into one bucket and the window is planned again.
    int files = 1;
    { FILE* f0 = fopen(ckptName(o.resume, 0).c_str(), "rb");
      if (!f0) die("cannot open checkpoint for reading", __LINE__);
      int32_t h[6]; if (fread(h, sizeof(int32_t), 6, f0) != 6) die("bad checkpoint", __LINE__);
      files = h[5]; fclose(f0); }
    if (g_R != 1 && g_R != files) die("resume on one node or on the node count of the checkpoint", __LINE__);
    const bool merge = (g_R == 1);
    for (auto& b : pool) releaseBucket(b);
    pool.assign(1, Bucket());
    long total = 0;
    std::vector<uint64_t> bx; std::vector<double> bc;
    for (int r = (g_R == 1 ? 0 : g_rank); r < (g_R == 1 ? files : g_rank + 1); r++) {
      FILE* f = fopen(ckptName(o.resume, r).c_str(), "rb");
      if (!f) die("cannot open checkpoint for reading", __LINE__);
      int32_t hdr[6]; int wh[6]; int64_t wl = 0, nbk = 0;
      if (fread(hdr, sizeof(int32_t), 6, f) != 6 || hdr[0] != 0x50504B45 || hdr[1] != words || hdr[3] != M) die("checkpoint does not match this circuit", __LINE__);
      if (fread(wh, sizeof(int), 6, f) != 6 || fread(&wl, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
      std::vector<uint64_t> wp((size_t)wl);
      if (fread(wp.data(), sizeof(uint64_t), wp.size(), f) != wp.size() || fread(&nbk, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
      pos = hdr[2]; firstWindow = hdr[4] + 1;
      if (!merge) { w = unpackWindow(wh, wp); haveWindow = true; pool.assign((size_t)nbk, Bucket()); }
      for (int64_t i = 0; i < nbk; i++) {
        int64_t n = 0;
        if (fread(&n, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
        bx.resize((size_t)n * words); bc.resize((size_t)n);
        if (fread(bx.data(), sizeof(uint64_t), bx.size(), f) != bx.size() || fread(bc.data(), sizeof(double), bc.size(), f) != bc.size()) die("bad checkpoint", __LINE__);
        appendHost(pool[merge ? 0 : (size_t)i], bx.data(), bc.data(), (long)n);
        total += n;
      }
      fclose(f);
    }
    long tg = 0;
    MPI_Allreduce(&total, &tg, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) printf("resumed at operation %d with %ld terms on %d node(s) (checkpoint was written by %d)\n", pos, tg, g_R, files);
  }

  double tPlan = 0, growth = 2.0, tExchange = 0, bytesMoved = 0, worstCoset = 0, tPilot = 0, pilotGrowth = -1;
  long pilotEnds = 0;
  long maxBoundary = 0, movedBuckets = 0;
  int windows = 0, maxWindowOps = 0, minAnn = 2 * g_L.n;
  MPI_Barrier(MPI_COMM_WORLD);
  const double tBegin = now();

  std::vector<uint64_t> noSample;
  if (!haveWindow) w = planShared(pos, 0, noSample, o, rng);   // the pool is one bucket at the start
  for (int wi = firstWindow; ; wi++) {
    long NinLocal = 0, Nin = 0;
    for (auto& b : pool) NinLocal += b.n;
    MPI_Allreduce(&NinLocal, &Nin, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    maxBoundary = std::max(maxBoundary, Nin);

    // Where should this window end? Its rows stay valid for any shorter window. With several buckets,
    // every node runs one of its buckets ahead and the window ends at the truncation after which those
    // pilots are smallest (in the second half of the window). Otherwise at the last truncation.
    if (!ref && w.e - w.b > 1) {
      const double tq = now();
      int newEnd = o.snap ? snapEnd(w) : w.e;
      if (o.pilot && w.g > 0 && Nin >= o.target) {
        const int len = w.e - w.b;
        std::vector<double> after(len, 0.0), tot(len, 0.0);
        std::vector<const Bucket*> cand;
        for (auto& b : pool) if (b.n > 0) cand.push_back(&b);
        int done = len;
        double scale = 0;
        if (!cand.empty()) {
          std::sort(cand.begin(), cand.end(), [](const Bucket* a, const Bucket* b) { return a->n < b->n; });
          const Bucket* pb = cand[cand.size() / 2];
          done = workers[0].pilot(w, *pb, after);
          CUDA(cudaSetDevice(0));
          scale = (double)pb->n;
        }
        int doneMin = len; double scaleTot = 0;
        MPI_Allreduce(&done, &doneMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(after.data(), tot.data(), len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&scale, &scaleTot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (scaleTot > 0 && doneMin > len / 2) {
          double best = 1e300; int bestEnd = -1;
          for (int i = len / 2; i < doneMin; i++) {
            const int k = w.b + i;
            const bool candidate = g_ops[k].trunc || (k + 1 == w.e && doneMin == len);
            if (candidate && tot[i] < best) { best = tot[i]; bestEnd = k + 1; }
          }
          if (bestEnd > 0) { newEnd = bestEnd; pilotEnds++; pilotGrowth = best / scaleTot; }
        }
      }
      w.e = newEnd;
      tPilot += now() - tq;
    }
    const bool last = (w.e >= M);

    // Plan the next window: its rows are scored on a sample of the current terms, moved through the
    // bit-changing operations of this window (branching is ignored for the estimate).
    Window nextW;
    if (!last) {
      const double tp = now();
      int gWanted = 0;
      // Terms expected at the end of this window: from the pilot if there was one, else from the last ratio.
      if (!ref) { const double est = Nin * (pilotGrowth > 0 ? pilotGrowth * 1.3 : growth); while (gWanted < o.gmax && est / (double)(1L << gWanted) > (double)o.target) gWanted++; }
      pilotGrowth = -1;
      std::vector<uint64_t> sample;
      if (gWanted > 0) {
        const long stride = std::max<long>(1, Nin / std::max<long>(1, o.sample));
        for (auto& b : pool)
          for (auto& k : b.ch)
            for (long i = 0; i < k.n; i += stride) sample.insert(sample.end(), k.xz + (size_t)i * words, k.xz + (size_t)(i + 1) * words);
        for (int k = w.b; k < w.e; k++)
          if (changesBits(g_ops[k]))
            for (size_t i = 0; i < sample.size() / words; i++) mapVec(g_ops[k], &sample[i * words], g_L);
      }
      nextW = planShared(w.e, gWanted, sample, o, rng);
      tPlan += now() - tp;
    }
    windows++;
    windowStarts.push_back(w.b);
    maxWindowOps = std::max(maxWindowOps, w.e - w.b);
    minAnn = std::min(minAnn, w.annDim);
    worstCoset = std::max(worstCoset, w.cosetFrac);
    if (!conserved.empty()) setCheckRows(w.b);

    sh.nextB.assign(last ? 1 : ((size_t)1 << nextW.g), Bucket());
    sh.nout = 0;
    std::sort(pool.begin(), pool.end(), [](const Bucket& a, const Bucket& b) { return a.n > b.n; });
    for (auto& b : pool)
      if (b.n > 0) { sh.queue.push_back(Work()); sh.queue.back().b.ch.swap(b.ch); sh.queue.back().b.n = b.n; sh.queue.back().startOp = w.b; b.n = 0; }
      else releaseBucket(b);
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
      writeCheckpoint(pos, wi, nextW);
      if (o.stopAtCkpt) { if (g_rank == 0) printf("stopped at checkpoint\n"); MPI_Finalize(); return 0; }
    }
    w = nextW;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = now() - tBegin;

  double loc[9] = {0, 0, 0, 0, 0, tExchange, bytesMoved, sh.expec, 0}, sum[9];
  for (auto& x : workers) { loc[0] += x.tOps; loc[1] += x.tPart; loc[2] += x.tH2D; loc[3] += x.tD2H; loc[4] += x.tIdle; }
  MPI_Allreduce(loc, sum, 9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  long lc[7] = {sh.runs.load(), sh.splits.load(), sh.hostSplits.load(), sh.peak.load(), sh.verified.load(), sh.violations.load(), g_pool.allocated}, sc[7], mc[7];
  MPI_Allreduce(lc, sc, 7, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(lc, mc, 7, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  if (g_rank == 0) {
    printf("expectation=%.12g\n", sum[7]);
    printf("windows=%d (longest %d operations, smallest row space %d of %d, largest coset share %.4f) partitioned boundaries=%ld\n",
           windows, maxWindowOps, minAnn, 2 * g_L.n, worstCoset, movedBuckets);
    printf("max_terms_at_window_boundary=%ld largest_bucket_on_a_gpu=%ld bucket_runs=%ld splits_on_gpu=%ld splits_on_host=%ld\n",
           maxBoundary, mc[3], sc[0], sc[1], sc[2]);
    if (!conserved.empty())
      printf("integrity: %ld term checks against %zu conserved parities, %ld violations\n", sc[4], conserved.size(), sc[5]);
    printf("wall=%.3f s; summed over %d GPU workers: operations=%.3f partition=%.3f host->gpu=%.3f gpu->host=%.3f idle=%.3f; planning=%.3f; pilot=%.3f (%ld window ends chosen)\n",
           wall, o.gpus * g_R, sum[0], sum[1], sum[2], sum[3], sum[4], tPlan, tPilot, pilotEnds);
    printf("host memory in chunks: %.2f GiB on the busiest node\n",
           mc[6] * (double)o.chunk * (words * 8 + 8) / 1073741824.0);
    if (!o.profile.empty() && g_R == 1) {
      FILE* f = fopen(o.profile.c_str(), "w");
      if (f) {
        fprintf(f, "op,branching,trunc,window_start,terms_before\n");
        size_t wi2 = 0;
        for (int k = 0; k < M; k++) {
          int ws = 0;
          while (wi2 < windowStarts.size() && windowStarts[wi2] < k) wi2++;
          if (wi2 < windowStarts.size() && windowStarts[wi2] == k) ws = 1;
          fprintf(f, "%d,%d,%d,%d,%ld\n", k, (int)(g_ops[k].type == OP_ROT && g_ops[k].branches), (int)g_ops[k].trunc, ws, popAt[k].load());
        }
        fclose(f);
      }
    }
    if (g_R > 1)
      printf("between nodes: %.3f GB sent, %.3f s per node on average (%.2f GB/s per node while exchanging); MPI calls while a window runs: 0\n",
             sum[6] / 1e9, sum[5] / g_R, sum[5] > 0 ? sum[6] / 1e9 / (sum[5] / g_R) / g_R : 0.0);
  }
  MPI_Finalize();
  return 0;
}
