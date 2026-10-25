#pragma once
#include "device_policy.hpp"

// WTA natural-log packing variant; caller clears bucket sizes. Bounded atomic
// ring, unordered survivors.
namespace gslide::sycl_port {
struct hash_shape {
  int previous, nodes, tables, k, bin, tile, per_thread, buckets, capacity;
};
inline int address(const int *bins, const float *row, int k, int bin,
                   int shift) {
  int result = 0;
  for (int j = 0; j < k; ++j) {
    float maximum = -std::numeric_limits<float>::max();
    int winner = 0;
    for (int p = 0; p < bin; ++p)
      if (row[bins[j * bin + p]] > maximum) {
        maximum = row[bins[j * bin + p]];
        winner = p;
      }
    result += winner << ((k - 1 - j) * shift);
  }
  return result;
}
inline void build_hash(sycl::queue &q, const int *bins, const float *weights,
                       hash_shape z, int *buckets, int *sizes, int local = 64) {
  require(z.previous > 0 && z.nodes >= 0 && z.tables > 0 && z.k > 0 &&
              z.bin > 0 && z.tile > 0 && z.per_thread > 0 &&
              z.tile % z.per_thread == 0 && z.capacity > 0,
          "hash shape");
  int lanes = z.tile / z.per_thread, shift = int(std::log(float(z.bin)));
  require(local % lanes == 0 && (z.k - 1) * shift < 30, "hash lanes/shift");
  launch_policy(q, local);
  std::uint64_t bound = 0;
  for (int j = 0; j < z.k; ++j)
    bound += std::uint64_t(z.bin - 1) << ((z.k - 1 - j) * shift);
  require(bound < unsigned(z.buckets), "bucket address capacity");
  if (!z.nodes)
    return;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<int, 1> tile(z.tile * z.k * z.bin, h);
     sycl::local_accessor<float, 1> rows((local / lanes) * z.previous, h);
     h.parallel_for(
         sycl::nd_range<1>(padded(z.nodes * lanes, local), local),
         [=](sycl::nd_item<1> it) {
           int lane = it.get_local_linear_id(),
               node = it.get_global_linear_id() / lanes, slot = lane / lanes,
               tile_lane = lane % lanes;
           int node_begin = it.get_group_linear_id() * (local / lanes);
           for (int i = lane; i < (local / lanes) * z.previous; i += local) {
             int global = node_begin * z.previous + i;
             rows[i] = global < z.nodes * z.previous ? weights[global] : 0;
           }
           for (int t = 0; t < z.tables; t += z.tile) {
             for (int i = lane; i < z.tile * z.k * z.bin; i += local) {
               int global = t * z.k * z.bin + i;
               tile[i] = global < z.tables * z.k * z.bin ? bins[global] : 0;
             }
             it.barrier(sycl::access::fence_space::local_space);
             if (node < z.nodes)
               for (int j = tile_lane * z.per_thread;
                    j < (tile_lane + 1) * z.per_thread && t + j < z.tables;
                    ++j) {
                 int a =
                     address(tile.get_multi_ptr<sycl::access::decorated::no>()
                                     .get() +
                                 j * z.k * z.bin,
                             rows.get_multi_ptr<sycl::access::decorated::no>()
                                     .get() +
                                 slot * z.previous,
                             z.k, z.bin, shift) +
                     (t + j) * z.buckets;
                 sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                     count(sizes[a]);
                 int pos = count.fetch_add(1) % z.capacity;
                 sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                     entry(buckets[a * z.capacity + pos]);
                 entry.store(node);
               }
             it.barrier(sycl::access::fence_space::local_space);
           }
         });
   }).wait_and_throw();
}
} // namespace gslide::sycl_port
