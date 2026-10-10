// Times the MPI calls of an unmodified MPI program (LD_PRELOAD): used for the distributed CPU baselines.
// Every wrapped call adds its time, its count and the bytes it sends to a counter of its kind. At MPI_Finalize
// the counters are reduced over the ranks and rank 0 writes them to the file named by PP_MPIPROF_OUT:
//   mpi <call>: class=<collective|point_to_point|one_sided|wait> calls/rank=... seconds/rank=... max_seconds=... GB/rank=...
//   mpi total: seconds/rank=... (collective=... point_to_point=... one_sided=... wait=...) lifetime/rank=... ranks=...
// "wait" is the completion of nonblocking calls (MPI_Wait*, MPI_Test*). lifetime is MPI_Init to MPI_Finalize.
#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { K_ALLREDUCE, K_REDUCE, K_BCAST, K_BARRIER, K_ALLTOALL, K_ALLTOALLV, K_ALLGATHER, K_ALLGATHERV, K_GATHER, K_GATHERV,
       K_SCAN, K_IALLTOALLV, K_SEND, K_RECV, K_ISEND, K_IRECV, K_SENDRECV, K_PROBE, K_WAIT, K_TEST, K_PUT, K_GET, K_ACCUMULATE,
       K_WIN_SYNC, K_WIN_SETUP, K_COUNT };
static const char* const kName[K_COUNT] = {
  "allreduce", "reduce", "bcast", "barrier", "alltoall", "alltoallv", "allgather", "allgatherv", "gather", "gatherv",
  "scan", "ialltoallv", "send", "recv", "isend", "irecv", "sendrecv", "probe", "wait", "test", "put", "get", "accumulate",
  "win_sync", "win_setup" };
// 0 collective, 1 point to point, 2 one sided, 3 wait
static const int kClass[K_COUNT] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2};
static const char* const cName[4] = {"collective", "point_to_point", "one_sided", "wait"};
static long long gNs[K_COUNT], gCalls[K_COUNT], gBytes[K_COUNT];
static double gStart;

static inline long long nowNs(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}
static inline void add(int k, long long t0, long long bytes) {
  __sync_fetch_and_add(&gNs[k], nowNs() - t0);
  __sync_fetch_and_add(&gCalls[k], 1);
  if (bytes) __sync_fetch_and_add(&gBytes[k], bytes);
}
static inline long long size(int count, MPI_Datatype type) {
  int s = 0;
  if (type != MPI_DATATYPE_NULL) PMPI_Type_size(type, &s);
  return (long long)count * s;
}
static long long sizeV(const int* counts, MPI_Datatype type, MPI_Comm comm) {
  int n = 0, s = 0;
  long long total = 0;
  if (!counts) return 0;
  PMPI_Comm_size(comm, &n);
  PMPI_Type_size(type, &s);
  for (int i = 0; i < n; i++) total += counts[i];
  return total * s;
}
#define TIMED(k, bytes, call) long long t0_ = nowNs(); int r_ = (call); add((k), t0_, (bytes)); return r_

int MPI_Init(int* argc, char*** argv) { int r = PMPI_Init(argc, argv); gStart = PMPI_Wtime(); return r; }
int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) {
  int r = PMPI_Init_thread(argc, argv, required, provided);
  gStart = PMPI_Wtime();
  return r;
}

int MPI_Allreduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op op, MPI_Comm c) { TIMED(K_ALLREDUCE, size(n, t), PMPI_Allreduce(s, r, n, t, op, c)); }
int MPI_Reduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op op, int root, MPI_Comm c) { TIMED(K_REDUCE, size(n, t), PMPI_Reduce(s, r, n, t, op, root, c)); }
int MPI_Bcast(void* b, int n, MPI_Datatype t, int root, MPI_Comm c) { TIMED(K_BCAST, size(n, t), PMPI_Bcast(b, n, t, root, c)); }
int MPI_Barrier(MPI_Comm c) { TIMED(K_BARRIER, 0, PMPI_Barrier(c)); }
int MPI_Alltoall(const void* s, int sn, MPI_Datatype st, void* r, int rn, MPI_Datatype rt, MPI_Comm c) {
  int p = 1;
  PMPI_Comm_size(c, &p);
  TIMED(K_ALLTOALL, size(sn, st) * p, PMPI_Alltoall(s, sn, st, r, rn, rt, c));
}
int MPI_Alltoallv(const void* s, const int* sn, const int* sd, MPI_Datatype st, void* r, const int* rn, const int* rd, MPI_Datatype rt, MPI_Comm c) {
  TIMED(K_ALLTOALLV, sizeV(sn, st, c), PMPI_Alltoallv(s, sn, sd, st, r, rn, rd, rt, c));
}
int MPI_Ialltoallv(const void* s, const int* sn, const int* sd, MPI_Datatype st, void* r, const int* rn, const int* rd, MPI_Datatype rt, MPI_Comm c, MPI_Request* q) {
  TIMED(K_IALLTOALLV, sizeV(sn, st, c), PMPI_Ialltoallv(s, sn, sd, st, r, rn, rd, rt, c, q));
}
int MPI_Allgather(const void* s, int sn, MPI_Datatype st, void* r, int rn, MPI_Datatype rt, MPI_Comm c) { TIMED(K_ALLGATHER, size(sn, st), PMPI_Allgather(s, sn, st, r, rn, rt, c)); }
int MPI_Allgatherv(const void* s, int sn, MPI_Datatype st, void* r, const int* rn, const int* rd, MPI_Datatype rt, MPI_Comm c) {
  TIMED(K_ALLGATHERV, size(sn, st), PMPI_Allgatherv(s, sn, st, r, rn, rd, rt, c));
}
int MPI_Gather(const void* s, int sn, MPI_Datatype st, void* r, int rn, MPI_Datatype rt, int root, MPI_Comm c) { TIMED(K_GATHER, size(sn, st), PMPI_Gather(s, sn, st, r, rn, rt, root, c)); }
int MPI_Gatherv(const void* s, int sn, MPI_Datatype st, void* r, const int* rn, const int* rd, MPI_Datatype rt, int root, MPI_Comm c) {
  TIMED(K_GATHERV, size(sn, st), PMPI_Gatherv(s, sn, st, r, rn, rd, rt, root, c));
}
int MPI_Scan(const void* s, void* r, int n, MPI_Datatype t, MPI_Op op, MPI_Comm c) { TIMED(K_SCAN, size(n, t), PMPI_Scan(s, r, n, t, op, c)); }
int MPI_Exscan(const void* s, void* r, int n, MPI_Datatype t, MPI_Op op, MPI_Comm c) { TIMED(K_SCAN, size(n, t), PMPI_Exscan(s, r, n, t, op, c)); }

int MPI_Send(const void* b, int n, MPI_Datatype t, int d, int tag, MPI_Comm c) { TIMED(K_SEND, size(n, t), PMPI_Send(b, n, t, d, tag, c)); }
int MPI_Recv(void* b, int n, MPI_Datatype t, int s, int tag, MPI_Comm c, MPI_Status* st) { TIMED(K_RECV, 0, PMPI_Recv(b, n, t, s, tag, c, st)); }
int MPI_Isend(const void* b, int n, MPI_Datatype t, int d, int tag, MPI_Comm c, MPI_Request* q) { TIMED(K_ISEND, size(n, t), PMPI_Isend(b, n, t, d, tag, c, q)); }
int MPI_Irecv(void* b, int n, MPI_Datatype t, int s, int tag, MPI_Comm c, MPI_Request* q) { TIMED(K_IRECV, 0, PMPI_Irecv(b, n, t, s, tag, c, q)); }
int MPI_Sendrecv(const void* sb, int sn, MPI_Datatype st, int d, int stag, void* rb, int rn, MPI_Datatype rt, int s, int rtag, MPI_Comm c, MPI_Status* status) {
  TIMED(K_SENDRECV, size(sn, st), PMPI_Sendrecv(sb, sn, st, d, stag, rb, rn, rt, s, rtag, c, status));
}
int MPI_Probe(int s, int tag, MPI_Comm c, MPI_Status* st) { TIMED(K_PROBE, 0, PMPI_Probe(s, tag, c, st)); }
int MPI_Iprobe(int s, int tag, MPI_Comm c, int* flag, MPI_Status* st) { TIMED(K_PROBE, 0, PMPI_Iprobe(s, tag, c, flag, st)); }
int MPI_Wait(MPI_Request* q, MPI_Status* st) { TIMED(K_WAIT, 0, PMPI_Wait(q, st)); }
int MPI_Waitall(int n, MPI_Request q[], MPI_Status st[]) { TIMED(K_WAIT, 0, PMPI_Waitall(n, q, st)); }
int MPI_Waitany(int n, MPI_Request q[], int* i, MPI_Status* st) { TIMED(K_WAIT, 0, PMPI_Waitany(n, q, i, st)); }
int MPI_Waitsome(int n, MPI_Request q[], int* m, int idx[], MPI_Status st[]) { TIMED(K_WAIT, 0, PMPI_Waitsome(n, q, m, idx, st)); }
int MPI_Test(MPI_Request* q, int* flag, MPI_Status* st) { TIMED(K_TEST, 0, PMPI_Test(q, flag, st)); }
int MPI_Testall(int n, MPI_Request q[], int* flag, MPI_Status st[]) { TIMED(K_TEST, 0, PMPI_Testall(n, q, flag, st)); }

int MPI_Put(const void* o, int on, MPI_Datatype ot, int rank, MPI_Aint disp, int tn, MPI_Datatype tt, MPI_Win w) { TIMED(K_PUT, size(on, ot), PMPI_Put(o, on, ot, rank, disp, tn, tt, w)); }
int MPI_Get(void* o, int on, MPI_Datatype ot, int rank, MPI_Aint disp, int tn, MPI_Datatype tt, MPI_Win w) { TIMED(K_GET, size(on, ot), PMPI_Get(o, on, ot, rank, disp, tn, tt, w)); }
int MPI_Accumulate(const void* o, int on, MPI_Datatype ot, int rank, MPI_Aint disp, int tn, MPI_Datatype tt, MPI_Op op, MPI_Win w) {
  TIMED(K_ACCUMULATE, size(on, ot), PMPI_Accumulate(o, on, ot, rank, disp, tn, tt, op, w));
}
int MPI_Win_fence(int a, MPI_Win w) { TIMED(K_WIN_SYNC, 0, PMPI_Win_fence(a, w)); }
int MPI_Win_lock(int type, int rank, int a, MPI_Win w) { TIMED(K_WIN_SYNC, 0, PMPI_Win_lock(type, rank, a, w)); }
int MPI_Win_unlock(int rank, MPI_Win w) { TIMED(K_WIN_SYNC, 0, PMPI_Win_unlock(rank, w)); }
int MPI_Win_lock_all(int a, MPI_Win w) { TIMED(K_WIN_SYNC, 0, PMPI_Win_lock_all(a, w)); }
int MPI_Win_unlock_all(MPI_Win w) { TIMED(K_WIN_SYNC, 0, PMPI_Win_unlock_all(w)); }
int MPI_Win_flush(int rank, MPI_Win w) { TIMED(K_WIN_SYNC, 0, PMPI_Win_flush(rank, w)); }
int MPI_Win_flush_all(MPI_Win w) { TIMED(K_WIN_SYNC, 0, PMPI_Win_flush_all(w)); }
int MPI_Win_create(void* b, MPI_Aint s, int du, MPI_Info info, MPI_Comm c, MPI_Win* w) { TIMED(K_WIN_SETUP, 0, PMPI_Win_create(b, s, du, info, c, w)); }
int MPI_Win_allocate(MPI_Aint s, int du, MPI_Info info, MPI_Comm c, void* b, MPI_Win* w) { TIMED(K_WIN_SETUP, 0, PMPI_Win_allocate(s, du, info, c, b, w)); }
int MPI_Win_free(MPI_Win* w) { TIMED(K_WIN_SETUP, 0, PMPI_Win_free(w)); }

int MPI_Finalize(void) {
  int rank = 0, ranks = 1;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &ranks);
  double loc[3 * K_COUNT + 1], sum[3 * K_COUNT + 1], mx[3 * K_COUNT + 1];
  for (int k = 0; k < K_COUNT; k++) {
    loc[3 * k] = gNs[k] * 1e-9;
    loc[3 * k + 1] = (double)gCalls[k];
    loc[3 * k + 2] = (double)gBytes[k];
  }
  loc[3 * K_COUNT] = PMPI_Wtime() - gStart;
  PMPI_Reduce(loc, sum, 3 * K_COUNT + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  PMPI_Reduce(loc, mx, 3 * K_COUNT + 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  const char* path = getenv("PP_MPIPROF_OUT");
  if (rank == 0 && path) {
    FILE* f = fopen(path, "w");
    if (f) {
      double cls[4] = {0, 0, 0, 0}, total = 0;
      for (int k = 0; k < K_COUNT; k++) {
        if (sum[3 * k + 1] == 0) continue;
        fprintf(f, "mpi %s: class=%s calls/rank=%.1f seconds/rank=%.4f max_seconds=%.4f GB/rank=%.4f\n", kName[k], cName[kClass[k]],
                sum[3 * k + 1] / ranks, sum[3 * k] / ranks, mx[3 * k], sum[3 * k + 2] / ranks / 1e9);
        cls[kClass[k]] += sum[3 * k] / ranks;
        total += sum[3 * k] / ranks;
      }
      fprintf(f, "mpi total: seconds/rank=%.4f (collective=%.4f point_to_point=%.4f one_sided=%.4f wait=%.4f) lifetime/rank=%.4f ranks=%d\n",
              total, cls[0], cls[1], cls[2], cls[3], sum[3 * K_COUNT] / ranks, ranks);
      fclose(f);
    }
  }
  return PMPI_Finalize();
}
