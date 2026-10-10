// Cost of host memory for the chunk pool: how fast is new pinned memory, and what are the alternatives.
//   pin_test [GiB]      (one GPU; prints seconds and GB/s)
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA: %s (line %d)\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static void touch(char* p, size_t n, int threads) {
  std::vector<std::thread> th;
  for (int t = 0; t < threads; t++) th.emplace_back([=] { memset(p + n / threads * t, 1, n / threads); });
  for (auto& x : th) x.join();
}

int main(int argc, char** argv) {
  const double gib = argc > 1 ? atof(argv[1]) : 2.0;
  const size_t n = (size_t)(gib * 1073741824.0) & ~((size_t)(1 << 21) - 1);
  CUDA(cudaSetDevice(0));
  void* dev = nullptr;
  CUDA(cudaMalloc(&dev, n));
  CUDA(cudaMemset(dev, 3, n));
  CUDA(cudaDeviceSynchronize());
  auto rate = [&](double s) { return (double)n / 1e9 / s; };
  auto copies = [&](void* host, const char* what) {
    double t0 = now(); CUDA(cudaMemcpy(host, dev, n, cudaMemcpyDeviceToHost)); double t1 = now();
    CUDA(cudaMemcpy(dev, host, n, cudaMemcpyHostToDevice)); double t2 = now();
    CUDA(cudaMemcpy(host, dev, n, cudaMemcpyDeviceToHost)); double t3 = now();
    printf("  copies with %s: gpu->host %.1f GB/s (again %.1f), host->gpu %.1f GB/s\n", what, rate(t1 - t0), rate(t3 - t2), rate(t2 - t1));
  };
  printf("%.2f GiB per test\n", (double)n / 1073741824.0);
  { double t0 = now(); void* p = nullptr; CUDA(cudaHostAlloc(&p, n, cudaHostAllocPortable)); double t1 = now();
    printf("cudaHostAlloc (portable): %.3f s, %.2f GB/s\n", t1 - t0, rate(t1 - t0)); copies(p, "it");
    t0 = now(); CUDA(cudaFreeHost(p)); printf("  cudaFreeHost: %.3f s\n", now() - t0); }
  { double t0 = now(); void* p = nullptr; CUDA(cudaHostAlloc(&p, n, cudaHostAllocDefault)); double t1 = now();
    printf("cudaHostAlloc (default): %.3f s, %.2f GB/s\n", t1 - t0, rate(t1 - t0)); CUDA(cudaFreeHost(p)); }
  for (int huge = 0; huge < 2; huge++)
    for (int threads : {1, 4}) {
      void* p = nullptr;
      if (posix_memalign(&p, 1 << 21, n)) return 1;
      if (huge) madvise(p, n, MADV_HUGEPAGE);
      double t0 = now(); touch((char*)p, n, threads); double t1 = now();
      printf("first touch, %s pages, %d thread(s): %.3f s, %.2f GB/s\n", huge ? "2 MB (madvise)" : "4 KB", threads, t1 - t0, rate(t1 - t0));
      if (threads == 1) copies(p, "pageable memory");
      t0 = now(); cudaError_t e = cudaHostRegister(p, n, cudaHostRegisterPortable); t1 = now();
      if (e == cudaSuccess) {
        printf("  cudaHostRegister (portable) of touched memory: %.3f s, %.2f GB/s\n", t1 - t0, rate(t1 - t0));
        if (threads == 1) copies(p, "registered memory");
        t0 = now(); CUDA(cudaHostUnregister(p)); printf("  cudaHostUnregister: %.3f s\n", now() - t0);
      } else printf("  cudaHostRegister failed: %s\n", cudaGetErrorString(e));
      free(p);
    }
  { // registering in pieces of 10.5 MB (one chunk) and of 256 MB
    for (size_t piece : {(size_t)10485760, (size_t)268435456}) {
      void* p = nullptr;
      if (posix_memalign(&p, 1 << 21, n)) return 1;
      madvise(p, n, MADV_HUGEPAGE); touch((char*)p, n, 4);
      double t0 = now(); size_t k = 0;
      for (size_t off = 0; off + piece <= n; off += piece, k++) CUDA(cudaHostRegister((char*)p + off, piece, cudaHostRegisterPortable));
      double t1 = now();
      printf("cudaHostRegister in %zu pieces of %.1f MB: %.3f s, %.2f GB/s\n", k, piece / 1048576.0, t1 - t0, (double)(k * piece) / 1e9 / (t1 - t0));
      for (size_t off = 0, j = 0; j < k; off += piece, j++) CUDA(cudaHostUnregister((char*)p + off));
      free(p);
    }
  }
  return 0;
}
