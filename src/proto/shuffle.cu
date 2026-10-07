// GPU kernels for parity-based (GF(2)-linear) ownership of Pauli strings.
//
// A term is stored as 2*nw packed 64-bit words: X words then Z words. Ownership is
// defined by g rows; row j is a bit mask over all 2*nw words, and its value for a term
// is the parity of (term AND mask). The owner id is the g-bit number formed by the rows.

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cstdint>

namespace {

const int MAX_ROWS = 8;
const int MAX_WORDS = 8;

struct OwnerFn {
  const uint64_t* xz;
  int words;      // 2*nw
  int g;
  int64_t stride;
  uint64_t mask[MAX_ROWS][MAX_WORDS];
  __device__ uint8_t operator()(int64_t i) const {
    const uint64_t* t = xz + i * stride * words;
    uint8_t o = 0;
    for (int j = 0; j < g; j++) {
      int par = 0;
      for (int w = 0; w < words; w++) par ^= __popcll(t[w] & mask[j][w]) & 1;
      o |= (uint8_t)(par << j);
    }
    return o;
  }
};

template <int W>
struct Term { uint64_t w[W]; };

}  // namespace

extern "C" {

// owners[i] = owner id of term i. masks: g rows of `words` 64-bit words (host memory).
void pp_owners(const uint64_t* xz, int64_t n, int words, const uint64_t* masks, int g, uint8_t* owners) {
  OwnerFn f;
  f.xz = xz; f.words = words; f.g = g; f.stride = 1;
  for (int j = 0; j < g; j++)
    for (int w = 0; w < words; w++) f.mask[j][w] = masks[j * words + w];
  thrust::transform(thrust::device, thrust::counting_iterator<int64_t>(0), thrust::counting_iterator<int64_t>(n),
                    thrust::device_pointer_cast(owners), f);
}

// Owner ids of every stride-th term (terms 0, stride, 2*stride, ...); writes ceil(n/stride) ids.
void pp_owners_strided(const uint64_t* xz, int64_t n, int64_t stride, int words, const uint64_t* masks, int g, uint8_t* owners) {
  OwnerFn f;
  f.xz = xz; f.words = words; f.g = g; f.stride = stride;
  for (int j = 0; j < g; j++)
    for (int w = 0; w < words; w++) f.mask[j][w] = masks[j * words + w];
  const int64_t m = (n + stride - 1) / stride;
  thrust::transform(thrust::device, thrust::counting_iterator<int64_t>(0), thrust::counting_iterator<int64_t>(m),
                    thrust::device_pointer_cast(owners), f);
}

// Counts per owner id from owner ids that are already sorted (after pp_partition).
void pp_counts_sorted(const uint8_t* owners, int64_t n, int R, int64_t* cnt) {
  auto o = thrust::device_pointer_cast(owners);
  int64_t prev = 0;
  for (int r = 0; r < R; r++) {
    const int64_t ub = thrust::upper_bound(thrust::device, o, o + n, (uint8_t)r) - o;
    cnt[r] = ub - prev; prev = ub;
  }
}

// hist[r] = number of terms owned by r, for r < R (host memory).
void pp_hist(const uint8_t* owners, int64_t n, int R, int64_t* hist) {
  auto o = thrust::device_pointer_cast(owners);
  for (int r = 0; r < R; r++) hist[r] = thrust::count(thrust::device, o, o + n, (uint8_t)r);
}

// Reorders terms so that owner ids are non-decreasing. owners is overwritten (sorted);
// idx is scratch for n int32 values. Supports words == 2 or 4.
int pp_partition(const uint64_t* xz, const double* coef, uint8_t* owners, int64_t n, int words,
                 uint64_t* xzOut, double* coefOut, int32_t* idx) {
  auto o = thrust::device_pointer_cast(owners);
  auto ix = thrust::device_pointer_cast(idx);
  thrust::sequence(thrust::device, ix, ix + n);
  thrust::stable_sort_by_key(thrust::device, o, o + n, ix);
  thrust::gather(thrust::device, ix, ix + n, thrust::device_pointer_cast(coef), thrust::device_pointer_cast(coefOut));
  if (words == 4) {
    thrust::gather(thrust::device, ix, ix + n, thrust::device_pointer_cast((const Term<4>*)xz),
                   thrust::device_pointer_cast((Term<4>*)xzOut));
  } else if (words == 2) {
    thrust::gather(thrust::device, ix, ix + n, thrust::device_pointer_cast((const Term<2>*)xz),
                   thrust::device_pointer_cast((Term<2>*)xzOut));
  } else if (words == 6) {
    thrust::gather(thrust::device, ix, ix + n, thrust::device_pointer_cast((const Term<6>*)xz),
                   thrust::device_pointer_cast((Term<6>*)xzOut));
  } else if (words == 8) {
    thrust::gather(thrust::device, ix, ix + n, thrust::device_pointer_cast((const Term<8>*)xz),
                   thrust::device_pointer_cast((Term<8>*)xzOut));
  } else {
    return -1;
  }
  return 0;
}

}  // extern "C"
