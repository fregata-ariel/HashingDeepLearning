#pragma once
// Restricted test adapter. CUDA allocation, scheduling and warp instructions
// are not emulated. Shared-memory bodies must run with exactly one thread.
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>
#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__
struct CpuDim { int x = 1; };
static CpuDim blockDim, gridDim;
static CpuDim threadIdx{0}, blockIdx{0};
inline void __syncthreads() { assert(blockDim.x == 1); }
inline int __syncthreads_and(int value) { __syncthreads(); return value; }
inline void __threadfence_block() { __syncthreads(); }
template<class T, class U> T atomicAdd(T* ptr, U value) { T old = *ptr; *ptr += value; return old; }
template<class T, class U> T atomicExch(T* ptr, U value) { T old = *ptr; *ptr = value; return old; }
inline int atomicCAS(int* ptr, int compare, int value) { int old = *ptr; if (old == compare) *ptr = value; return old; }
template<class T, class U> typename std::common_type<T,U>::type max(T a, U b) { return std::max<typename std::common_type<T,U>::type>(a,b); }
template<class T, class U> typename std::common_type<T,U>::type min(T a, U b) { return std::min<typename std::common_type<T,U>::type>(a,b); }
inline float __expf(float x) { return std::exp(x); }
inline float __fdividef(float x, float y) { return x / y; }
inline float block_reduce(float x) { __syncthreads(); return x; }
inline float block_max(float x) { __syncthreads(); return x; }
alignas(64) char smem[65536];
int s_tile_bins[4096];
#include "CscActNodes.h"
#include "GPUMultiLinkedHashTable.h"

// Host storage substitutes for cudaMalloc/cudaFree. Test fixture capacities
// are small; these definitions are not production constructors.
CscActNodes::CscActNodes(int batch, int capacity, bool values, bool)
    : max_batch_size(batch), node_capacity(capacity), d_nodes(new int[capacity]{}),
      d_vals(values ? new float[capacity]{} : nullptr), d_offsets(new int[batch+1]{}), val_enabled(values) {}
void CscActNodes::free() { delete[] d_nodes; delete[] d_vals; delete[] d_offsets; }
struct OwnedCsc : CscActNodes {
  OwnedCsc(std::vector<int> nodes, std::vector<float> values, std::vector<int> offsets)
      : CscActNodes(static_cast<int>(offsets.size())-1, static_cast<int>(nodes.size()), !values.empty()) {
    assert(values.empty() || values.size() == nodes.size());
    std::copy(nodes.begin(), nodes.end(), d_nodes);
    std::copy(values.begin(), values.end(), d_vals);
    std::copy(offsets.begin(), offsets.end(), d_offsets);
  }
  ~OwnedCsc() { free(); }
  OwnedCsc(const OwnedCsc&) = delete;
  OwnedCsc& operator=(const OwnedCsc&) = delete;
};
GPUMultiLinkedHashTable::GPUMultiLinkedHashTable(int batch, size_t buckets, size_t pool, int cutoff)
    : max_tbl_num(batch), bucket_num_per_tbl(buckets), pool_size(pool),
      d_multi_tbl_keys(new int[(buckets+pool)*batch]),
      d_multi_tbl_vals(new int[(buckets+pool)*batch]{}),
      d_multi_tbl_nexts(new int[(buckets+pool)*batch]),
      d_multi_tbl_locks(new int[buckets*batch]{}), d_multi_tbl_sizes(new int[batch]{}),
      d_multi_tbl_pool_used_sizes(new int[batch]{}), threshold(cutoff) {
  std::fill_n(d_multi_tbl_keys, (buckets+pool)*batch, -1);
  std::fill_n(d_multi_tbl_nexts, (buckets+pool)*batch, -1);
}
void GPUMultiLinkedHashTable::free() {
  delete[] d_multi_tbl_keys; delete[] d_multi_tbl_vals; delete[] d_multi_tbl_nexts;
  delete[] d_multi_tbl_locks; delete[] d_multi_tbl_sizes; delete[] d_multi_tbl_pool_used_sizes;
}
struct OwnedTable : GPUMultiLinkedHashTable {
  using GPUMultiLinkedHashTable::GPUMultiLinkedHashTable;
  ~OwnedTable() { free(); }
  OwnedTable(const OwnedTable&) = delete;
  OwnedTable& operator=(const OwnedTable&) = delete;
};
inline void cpu_block(int block, int count) {
  blockDim.x = 1; threadIdx.x = 0; blockIdx.x = block; gridDim.x = count;
  std::fill_n(smem, sizeof(smem), 0); std::fill_n(s_tile_bins, 4096, 0);
}
inline void near(float actual, double expected, double tolerance = 2e-6) {
  if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance) {
    std::cerr << "mismatch actual=" << actual << " expected=" << expected << '\n';
    std::abort();
  }
}
