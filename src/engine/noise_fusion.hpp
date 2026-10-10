#pragma once

// Consecutive Pauli noise channels in one library call (cuPauliProp 0.5.0 fused application). Noise channels do
// not branch and do not merge terms, so a group gives the same terms and coefficients as the channels applied one
// by one, provided the input holds no equal strings (the caller checks). A truncation always ends a group; the
// caller ends groups at window boundaries.
static int noiseFusionCount(int begin, int end, int limit) {
  if (limit < 2 || begin >= end || g_ops[begin].type != OP_NOISE) return 1;
  int n = 1;
  while (n < limit && begin + n < end && g_ops[begin + n].type == OP_NOISE && !g_ops[begin + n - 1].trunc) n++;
  return n;
}

struct NoiseFusion {
  cupaulipropHandle_t handle;
  cupaulipropWorkspaceDescriptor_t wsMin, wsAvg, wsMax;
  std::vector<int32_t> adjoints;
  bool distributed = false;
  long calls = 0, operations = 0, fallbacks = 0;

  void init(cupaulipropHandle_t h, int limit, bool mpi) {
    handle = h; distributed = mpi;
    adjoints.assign(limit, 1);
    CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &wsMin));
    CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &wsAvg));
    CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &wsMax));
  }

  // Applies operations [begin, begin + n) to `input` and returns how many it applied: n, a smaller group, or 0
  // when the caller has to apply operation `begin` in the ordinary way. A worker of the engine is never
  // distributed: this makes no MPI call inside a window. The cuPauliProp distributed baseline agrees between
  // the ranks on whether the group fits before it enters the collective call.
  int apply(cupaulipropPauliExpansionView_t input, cupaulipropPauliExpansion_t output,
            const std::vector<cupaulipropQuantumOperator_t>& native, int begin, int n,
            bool allowCut, const std::vector<cupaulipropTruncationStrategy_t>& trunc,
            long capacity, size_t workspaceBytes, void* workspace, cudaStream_t stream) {
    while (n > 1) {
      const bool cut = allowCut && g_ops[begin + n - 1].trunc;
      const int32_t nT = cut ? (int32_t)trunc.size() : 0;
      const auto* pT = cut ? trunc.data() : nullptr;
      int64_t lo = 0, avg = 0, hi = 0, need = 0;
      const auto status = cupaulipropPauliExpansionViewPrepareOperatorFusedApplication(
        handle, input, n, native.data() + begin, adjoints.data(), nT, pT,
        (int64_t)workspaceBytes, &lo, wsMin, &avg, wsAvg, &hi, wsMax);
      int decision = 0;  // 0: ready, 1: ordinary application, 2: smaller group
      if (status == CUPAULIPROP_STATUS_INSUFFICIENT_DEVICE_PROPERTY) decision = 2;
      else if (status == CUPAULIPROP_STATUS_INSUFFICIENT_WORKSPACE) decision = 1;
      else {
        CUPP(status);
        CUPP(cupaulipropWorkspaceGetMemorySize(handle, wsMax, CUPAULIPROP_MEMSPACE_DEVICE,
                                              CUPAULIPROP_WORKSPACE_SCRATCH, &need));
        if (hi < 0 || hi > capacity || need < 0 || need > (int64_t)workspaceBytes) decision = 1;
      }
      if (distributed) {
        int global = 0;
        MPI_Allreduce(&decision, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        decision = global;
      }
      if (decision) {
        fallbacks++;
        if (decision == 1) return 0;
        n /= 2; continue;
      }
      CUPP(cupaulipropWorkspaceSetMemory(handle, wsMax, CUPAULIPROP_MEMSPACE_DEVICE,
                                        CUPAULIPROP_WORKSPACE_SCRATCH, workspace, workspaceBytes));
      const auto applied = cupaulipropPauliExpansionViewComputeOperatorFusedApplication(
        handle, input, output, n, native.data() + begin, adjoints.data(), nT, pT, wsMax, stream);
      // With this status the output is untouched. Any other error is fatal: an expansion that was written in
      // part is never retried.
      if (applied == CUPAULIPROP_STATUS_INSUFFICIENT_DEVICE_PROPERTY) { fallbacks++; n /= 2; continue; }
      CUPP(applied);
      CUDA(cudaStreamSynchronize(stream));
      calls++; operations += n;
      return n;
    }
    return 0;
  }
};
