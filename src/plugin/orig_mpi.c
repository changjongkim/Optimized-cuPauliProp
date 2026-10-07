/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cupauliprop.h> // cuPauliProp library header

#include <mpi.h> // MPI library header
#if defined(OPEN_MPI)
#include <mpi-ext.h> // MPIX_Query_cuda_support (Open MPI extension)
#endif

#include <stdlib.h>
#include <stdio.h>

/*
 * MPI Distributed Interface for cuPauliProp
 *
 * Caller must ensure stream ordering before calling these functions
 * (e.g., via ensureStreamOrderingForDistributedBackend which synchronizes the stream).
 */

/** Converts CUDA data type to the corresponding MPI data type */
static int convertCudaToMpiDataType(const cudaDataType_t cudaDataType,
                                    MPI_Datatype* mpiDataType)
{
    switch (cudaDataType)
    {
    case CUDA_R_8I:  *mpiDataType = MPI_INT8_T; return 0;
    case CUDA_R_16I: *mpiDataType = MPI_INT16_T; return 0;
    case CUDA_R_32I: *mpiDataType = MPI_INT32_T; return 0;
    case CUDA_R_64I: *mpiDataType = MPI_INT64_T; return 0;
    case CUDA_R_32F: *mpiDataType = MPI_FLOAT; return 0;
    case CUDA_R_64F: *mpiDataType = MPI_DOUBLE; return 0;
    case CUDA_C_32F: *mpiDataType = MPI_C_FLOAT_COMPLEX; return 0;
    case CUDA_C_64F: *mpiDataType = MPI_C_DOUBLE_COMPLEX; return 0;
    default:
        printf("#ERROR(cupauliprop:mpi-interface): Unknown CUDA data type: %d\n", (int)(cudaDataType));
        return -1;
    }
}

/** Unpacks the MPI_Comm object */
static int unpackMpiCommunicator(const cupaulipropDistributedCommunicator_t* comm,
                                 MPI_Comm* mpiComm)
{
    if (comm->commPtr == NULL)
    {
        printf("#ERROR(cupauliprop:mpi-interface): MPI_Comm object is NULL\n");
        return -1;
    }
    if (sizeof(MPI_Comm) != comm->commSize)
    {
        printf("#ERROR(cupauliprop:mpi-interface): MPI_Comm object has unexpected size\n");
        return -1;
    }
    *mpiComm = *((const MPI_Comm *)(comm->commPtr));
    return 0;
}

/** Frees MPI_Info while preserving an earlier MPI error, if any. */
static int freeMpiInfo(MPI_Info* info, int mpiErr)
{
    const int infoErr = MPI_Info_free(info);
    return mpiErr == MPI_SUCCESS ? infoErr : mpiErr;
}

#ifdef __cplusplus
extern "C" {
#endif

/** MPI_Comm_size wrapper */
/**
 * cuPauliProp passes raw device pointers into MPI_Isend/Irecv, so CUDA-aware
 * MPI is a hard runtime requirement of the distributed data path. A
 * non-CUDA-aware Open MPI would segfault or silently corrupt instead of
 * failing with a message; probe once and warn loudly when detectable.
 */
static void warnIfMpiNotCudaAware(void)
{
    static int probed = 0;
    if (probed) return;
    probed = 1;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (!MPIX_Query_cuda_support()) {
        fprintf(stderr,
            "#WARNING(cuPauliProp MPI shim): this Open MPI build reports NO "
            "CUDA support (MPIX_Query_cuda_support() == 0). cuPauliProp "
            "passes device pointers directly to MPI; expect crashes or data "
            "corruption. Use a CUDA-aware MPI build.\n");
    }
#endif
}

int cupaulipropMpiCommSize(const cupaulipropDistributedCommunicator_t * comm,
                         int32_t * numRanks)
{
    MPI_Comm mpiComm;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0) return -1;
    warnIfMpiNotCudaAware();

    int nranks = 0;
    int mpiErr = MPI_Comm_size(mpiComm, &nranks);
    *numRanks = nranks;
    return mpiErr;
}

/** Returns the size of the local subgroup of processes sharing node memory */
int cupaulipropMpiCommSizeShared(const cupaulipropDistributedCommunicator_t * comm,
                               int32_t * numRanks)
{
    *numRanks = 0;
    MPI_Comm mpiComm;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0) return -1;

    MPI_Info info = MPI_INFO_NULL;
    int mpiErr = MPI_Info_create(&info);
    if (mpiErr != MPI_SUCCESS) return mpiErr;

    mpiErr = MPI_Info_set(info, "mpi_hw_resource_type", "mpi_shared_memory");
    if (mpiErr != MPI_SUCCESS) return freeMpiInfo(&info, mpiErr);

    int procRank = -1;
    mpiErr = MPI_Comm_rank(mpiComm, &procRank);
    if (mpiErr != MPI_SUCCESS) return freeMpiInfo(&info, mpiErr);

    MPI_Comm localComm = MPI_COMM_NULL;
    mpiErr = MPI_Comm_split_type(mpiComm, MPI_COMM_TYPE_SHARED,
                                 procRank, info, &localComm);
    if (mpiErr != MPI_SUCCESS) return freeMpiInfo(&info, mpiErr);

    int numLocalRanks = 0;
    mpiErr = MPI_Comm_size(localComm, &numLocalRanks);
    if (mpiErr == MPI_SUCCESS) *numRanks = numLocalRanks;

    const int localCommErr = MPI_Comm_free(&localComm);
    if (mpiErr == MPI_SUCCESS) mpiErr = localCommErr;
    return freeMpiInfo(&info, mpiErr);
}

/** MPI_Comm_rank wrapper */
int cupaulipropMpiCommRank(const cupaulipropDistributedCommunicator_t * comm,
                         int32_t *procRank)
{
    MPI_Comm mpiComm;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0) return -1;

    int prank = -1;
    int mpiErr = MPI_Comm_rank(mpiComm, &prank);
    *procRank = prank;
    return mpiErr;
}

/** MPI_Barrier wrapper
 *
 * @param comm The distributed communicator
 * @param barrierBuffer Ignored for MPI (can be NULL)
 */
int cupaulipropMpiBarrier(const cupaulipropDistributedCommunicator_t * comm,
                           void* barrierBuffer)
{
    (void)barrierBuffer; /* Not used by MPI */
    MPI_Comm mpiComm;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0) return -1;
    return MPI_Barrier(mpiComm);
}

/** Allocates an MPI_Request */
int cupaulipropMpiCreateRequest(cupaulipropDistributedRequest_t * request)
{
    if (request == NULL) return -1;
    *request = NULL;
    *request = malloc(sizeof(MPI_Request));
    if (*request == NULL) return -1;
    *((MPI_Request*)*request) = MPI_REQUEST_NULL;
    return 0;
}

/** Frees a previously allocated MPI_Request */
int cupaulipropMpiDestroyRequest(cupaulipropDistributedRequest_t request)
{
    if (request == NULL) return -1;
    free(request);
    return 0;
}

/** MPI_Wait wrapper (waits for completion of MPI_Request) */
int cupaulipropMpiWaitRequest(cupaulipropDistributedRequest_t request)
{
    if (request == NULL) return -1;
    MPI_Status waitStatus;
    return MPI_Wait((MPI_Request*)request,
                    &waitStatus);
}

/** MPI_Test wrapper (tests for completion of MPI_Request) */
int cupaulipropMpiTestRequest(cupaulipropDistributedRequest_t request,
                               int32_t * completed)
{
    if (request == NULL || completed == NULL) return -1;
    MPI_Status testStatus;
    return MPI_Test((MPI_Request*)request,
                    completed,
                    &testStatus);
}

/** Begin a group of communication operations (no-op for MPI).
 *
 * MPI handles send/recv pairing internally, so this is a no-op.
 * This exists for compatibility with NCCL which requires grouping.
 */
int cupaulipropMpiGroupStart(void)
{
    return 0;
}

/** End a group of communication operations (no-op for MPI).
 *
 * MPI handles send/recv pairing internally, so this is a no-op.
 * This exists for compatibility with NCCL which requires grouping.
 */
int cupaulipropMpiGroupEnd(void)
{
    return 0;
}

/** MPI_Send wrapper */
int cupaulipropMpiSend(const cupaulipropDistributedCommunicator_t * comm,
                     const void * buffer,
                     int32_t count,
                     cudaDataType_t datatype,
                     int32_t destination,
                     int32_t tag)
{
    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    return MPI_Send(buffer,
                    (int)count,
                    mpiDataType,
                    (int)destination,
                    (int)tag,
                    mpiComm);
}

/** MPI_Isend wrapper */
int cupaulipropMpiSendAsync(const cupaulipropDistributedCommunicator_t * comm,
                          const void * buffer,
                          int32_t count,
                          cudaDataType_t datatype,
                          int32_t destination,
                          int32_t tag,
                          cupaulipropDistributedRequest_t request)
{
    if (request == NULL) return -1;

    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    int mpiErr = MPI_Isend(buffer,
                           (int)count,
                           mpiDataType,
                           (int)destination,
                           (int)tag,
                           mpiComm,
                           (MPI_Request*)request);
    if (mpiErr != MPI_SUCCESS) *((MPI_Request*)request) = MPI_REQUEST_NULL;
    return mpiErr;
}

/** MPI_Recv wrapper */
int cupaulipropMpiRecv(const cupaulipropDistributedCommunicator_t * comm,
                     void * buffer,
                     int32_t count,
                     cudaDataType_t datatype,
                     int32_t source,
                     int32_t tag)
{
    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    MPI_Status recvStatus;
    return MPI_Recv(buffer,
                    (int)count,
                    mpiDataType,
                    (int)source,
                    (int)tag,
                    mpiComm,
                    &recvStatus);
}

/** MPI_Irecv wrapper */
int cupaulipropMpiRecvAsync(const cupaulipropDistributedCommunicator_t * comm,
                          void * buffer,
                          int32_t count,
                          cudaDataType_t datatype,
                          int32_t source,
                          int32_t tag,
                          cupaulipropDistributedRequest_t request)
{
    if (request == NULL) return -1;

    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    int mpiErr = MPI_Irecv(buffer,
                           (int)count,
                           mpiDataType,
                           (int)source,
                           (int)tag,
                           mpiComm,
                           (MPI_Request*)request);
    if (mpiErr != MPI_SUCCESS) *((MPI_Request*)request) = MPI_REQUEST_NULL;
    return mpiErr;
}

/** MPI_Bcast wrapper */
int cupaulipropMpiBcast(const cupaulipropDistributedCommunicator_t * comm,
                      void * buffer,
                      int32_t count,
                      cudaDataType_t datatype,
                      int32_t root)
{
    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    return MPI_Bcast(buffer,
                     (int)count,
                     mpiDataType,
                     (int)root,
                     mpiComm);
}

/** MPI_Allreduce wrapper */
int cupaulipropMpiAllreduce(const cupaulipropDistributedCommunicator_t * comm,
                          const void * bufferIn,
                          void * bufferOut,
                          int32_t count,
                          cudaDataType_t datatype)
{
    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    return MPI_Allreduce(bufferIn,
                         bufferOut,
                         (int)count,
                         mpiDataType,
                         MPI_SUM,
                         mpiComm);
}

/** MPI_Allreduce IN_PLACE wrapper */
int cupaulipropMpiAllreduceInPlace(const cupaulipropDistributedCommunicator_t * comm,
                                 void * buffer,
                                 int32_t count,
                                 cudaDataType_t datatype)
{
    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    return MPI_Allreduce(MPI_IN_PLACE,
                         buffer,
                         (int)count,
                         mpiDataType,
                         MPI_SUM,
                         mpiComm);
}

/** MPI_Allreduce IN_PLACE MIN wrapper */
int cupaulipropMpiAllreduceInPlaceMin(const cupaulipropDistributedCommunicator_t * comm,
                                    void * buffer,
                                    int32_t count,
                                    cudaDataType_t datatype)
{
    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    return MPI_Allreduce(MPI_IN_PLACE,
                         buffer,
                         (int)count,
                         mpiDataType,
                         MPI_MIN,
                         mpiComm);
}

/** MPI_Allreduce DOUBLE_INT MINLOC wrapper */
int cupaulipropMpiAllreduceDoubleIntMinloc(const cupaulipropDistributedCommunicator_t * comm,
                                         const void * bufferIn, // *struct {double; int;}
                                         void * bufferOut)      // *struct {double; int;}
{
    MPI_Comm mpiComm;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0) return -1;

    return MPI_Allreduce(bufferIn,
                         bufferOut,
                         1,
                         MPI_DOUBLE_INT,
                         MPI_MINLOC,
                         mpiComm);
}

/** MPI_Allgather wrapper */
int cupaulipropMpiAllgather(const cupaulipropDistributedCommunicator_t * comm,
                          const void * bufferIn,
                          void * bufferOut,
                          int32_t count,
                          cudaDataType_t datatype)
{
    MPI_Comm mpiComm;
    MPI_Datatype mpiDataType;
    if (unpackMpiCommunicator(comm, &mpiComm) != 0 ||
        convertCudaToMpiDataType(datatype, &mpiDataType) != 0) return -1;

    return MPI_Allgather(bufferIn,
                         (int)count,
                         mpiDataType,
                         bufferOut,
                         (int)count,
                         mpiDataType,
                         mpiComm);
}

/**
 * Distributed communication service API wrapper binding table (imported by cuPauliProp).
 * The exposed C symbol must be named as "cupaulipropCommInterface".
 */
cupaulipropDistributedInterface_t cupaulipropCommInterface = {
    CUPAULIPROP_DISTRIBUTED_INTERFACE_VERSION,
    CUPAULIPROP_DISTRIBUTED_PROVIDER_MPI,
    cupaulipropMpiCommSize,
    cupaulipropMpiCommSizeShared,
    cupaulipropMpiCommRank,
    cupaulipropMpiBarrier,
    cupaulipropMpiCreateRequest,
    cupaulipropMpiDestroyRequest,
    cupaulipropMpiWaitRequest,
    cupaulipropMpiTestRequest,
    cupaulipropMpiGroupStart,
    cupaulipropMpiGroupEnd,
    cupaulipropMpiSend,
    cupaulipropMpiSendAsync,
    cupaulipropMpiRecv,
    cupaulipropMpiRecvAsync,
    cupaulipropMpiBcast,
    cupaulipropMpiAllreduce,
    cupaulipropMpiAllreduceInPlace,
    cupaulipropMpiAllreduceInPlaceMin,
    cupaulipropMpiAllreduceDoubleIntMinloc,
    cupaulipropMpiAllgather
};

#ifdef __cplusplus
} // extern "C"
#endif
