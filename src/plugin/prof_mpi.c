/*
 * Profiling shim around NVIDIA's cuPauliProp MPI distributed interface.
 *
 * The original plugin (orig_mpi.c, BSD-3-Clause, shipped in the cupauliprop
 * wheel) is included unchanged; every interface entry that communicates is
 * wrapped with a timer and call/byte counters. The driver reads the counters
 * through cuppprof_snapshot() after each gate.
 */

#define cupaulipropCommInterface cupaulipropCommInterface_orig
#include "orig_mpi.c"
#undef cupaulipropCommInterface

#include <string.h>

enum {
    K_BARRIER = 0, K_WAIT, K_TEST, K_SEND, K_ISEND, K_RECV, K_IRECV,
    K_BCAST, K_ALLREDUCE, K_ALLREDUCE_MIN, K_MINLOC, K_ALLGATHER, NK
};

#define MAX_RANKS 4096

static double    g_calls[NK];
static double    g_time[NK];
static double    g_bytes[NK];
static unsigned  g_gen = 1;
static unsigned  g_destGen[MAX_RANKS];
static double    g_destPairs;            /* cumulative distinct (mark, destination) pairs */
static long long g_destBytes[MAX_RANKS]; /* cumulative bytes sent per destination */
static double    g_maxMsgBytes;

static double elemSize(cudaDataType_t t)
{
    switch (t) {
    case CUDA_R_8I:  return 1;
    case CUDA_R_16I: return 2;
    case CUDA_R_32I: return 4;
    case CUDA_R_64I: return 8;
    case CUDA_R_32F: return 4;
    case CUDA_R_64F: return 8;
    case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default:         return 0;
    }
}

#define ACC(k, t0, nbytes) \
    do { g_calls[k] += 1; g_time[k] += MPI_Wtime() - (t0); g_bytes[k] += (nbytes); } while (0)

/* ---- exported to the driver ---- */

int cuppprof_num_kinds(void) { return NK; }

/* Starts a new attribution window (one per gate). */
void cuppprof_mark(void) { g_gen++; }

/* out must hold 3*NK + 2 doubles: calls[NK], time[NK], bytes[NK], destPairs, maxMsgBytes */
void cuppprof_snapshot(double* out)
{
    memcpy(out,          g_calls, sizeof(g_calls));
    memcpy(out + NK,     g_time,  sizeof(g_time));
    memcpy(out + 2 * NK, g_bytes, sizeof(g_bytes));
    out[3 * NK]     = g_destPairs;
    out[3 * NK + 1] = g_maxMsgBytes;
}

void cuppprof_dest_bytes(long long* out, int n)
{
    for (int i = 0; i < n && i < MAX_RANKS; i++) out[i] = g_destBytes[i];
}

/* ---- wrappers ---- */

static int profBarrier(const cupaulipropDistributedCommunicator_t* comm, void* buf)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiBarrier(comm, buf);
    ACC(K_BARRIER, t0, 0);
    return r;
}

static int profWait(cupaulipropDistributedRequest_t req)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiWaitRequest(req);
    ACC(K_WAIT, t0, 0);
    return r;
}

static int profTest(cupaulipropDistributedRequest_t req, int32_t* completed)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiTestRequest(req, completed);
    ACC(K_TEST, t0, 0);
    return r;
}

static int profSend(const cupaulipropDistributedCommunicator_t* comm, const void* buf,
                    int32_t count, cudaDataType_t dt, int32_t dest, int32_t tag)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiSend(comm, buf, count, dt, dest, tag);
    double nb = count * elemSize(dt);
    ACC(K_SEND, t0, nb);
    if (dest >= 0 && dest < MAX_RANKS) {
        g_destBytes[dest] += (long long)nb;
        if (g_destGen[dest] != g_gen) { g_destGen[dest] = g_gen; g_destPairs += 1; }
    }
    if (nb > g_maxMsgBytes) g_maxMsgBytes = nb;
    return r;
}

static int profIsend(const cupaulipropDistributedCommunicator_t* comm, const void* buf,
                     int32_t count, cudaDataType_t dt, int32_t dest, int32_t tag,
                     cupaulipropDistributedRequest_t req)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiSendAsync(comm, buf, count, dt, dest, tag, req);
    double nb = count * elemSize(dt);
    ACC(K_ISEND, t0, nb);
    if (dest >= 0 && dest < MAX_RANKS) {
        g_destBytes[dest] += (long long)nb;
        if (g_destGen[dest] != g_gen) { g_destGen[dest] = g_gen; g_destPairs += 1; }
    }
    if (nb > g_maxMsgBytes) g_maxMsgBytes = nb;
    return r;
}

static int profRecv(const cupaulipropDistributedCommunicator_t* comm, void* buf,
                    int32_t count, cudaDataType_t dt, int32_t src, int32_t tag)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiRecv(comm, buf, count, dt, src, tag);
    ACC(K_RECV, t0, count * elemSize(dt));
    return r;
}

static int profIrecv(const cupaulipropDistributedCommunicator_t* comm, void* buf,
                     int32_t count, cudaDataType_t dt, int32_t src, int32_t tag,
                     cupaulipropDistributedRequest_t req)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiRecvAsync(comm, buf, count, dt, src, tag, req);
    ACC(K_IRECV, t0, count * elemSize(dt));
    return r;
}

static int profBcast(const cupaulipropDistributedCommunicator_t* comm, void* buf,
                     int32_t count, cudaDataType_t dt, int32_t root)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiBcast(comm, buf, count, dt, root);
    ACC(K_BCAST, t0, count * elemSize(dt));
    return r;
}

static int profAllreduce(const cupaulipropDistributedCommunicator_t* comm, const void* in,
                         void* out, int32_t count, cudaDataType_t dt)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiAllreduce(comm, in, out, count, dt);
    ACC(K_ALLREDUCE, t0, count * elemSize(dt));
    return r;
}

static int profAllreduceInPlace(const cupaulipropDistributedCommunicator_t* comm, void* buf,
                                int32_t count, cudaDataType_t dt)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiAllreduceInPlace(comm, buf, count, dt);
    ACC(K_ALLREDUCE, t0, count * elemSize(dt));
    return r;
}

static int profAllreduceInPlaceMin(const cupaulipropDistributedCommunicator_t* comm, void* buf,
                                   int32_t count, cudaDataType_t dt)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiAllreduceInPlaceMin(comm, buf, count, dt);
    ACC(K_ALLREDUCE_MIN, t0, count * elemSize(dt));
    return r;
}

static int profMinloc(const cupaulipropDistributedCommunicator_t* comm, const void* in, void* out)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiAllreduceDoubleIntMinloc(comm, in, out);
    ACC(K_MINLOC, t0, 12);
    return r;
}

static int profAllgather(const cupaulipropDistributedCommunicator_t* comm, const void* in,
                         void* out, int32_t count, cudaDataType_t dt)
{
    double t0 = MPI_Wtime();
    int r = cupaulipropMpiAllgather(comm, in, out, count, dt);
    ACC(K_ALLGATHER, t0, count * elemSize(dt));
    return r;
}

cupaulipropDistributedInterface_t cupaulipropCommInterface = {
    CUPAULIPROP_DISTRIBUTED_INTERFACE_VERSION,
    CUPAULIPROP_DISTRIBUTED_PROVIDER_MPI,
    cupaulipropMpiCommSize,
    cupaulipropMpiCommSizeShared,
    cupaulipropMpiCommRank,
    profBarrier,
    cupaulipropMpiCreateRequest,
    cupaulipropMpiDestroyRequest,
    profWait,
    profTest,
    cupaulipropMpiGroupStart,
    cupaulipropMpiGroupEnd,
    profSend,
    profIsend,
    profRecv,
    profIrecv,
    profBcast,
    profAllreduce,
    profAllreduceInPlace,
    profAllreduceInPlaceMin,
    profMinloc,
    profAllgather
};
