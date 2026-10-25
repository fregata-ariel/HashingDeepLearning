#pragma once
#include "random.hpp"

// Direct-address frequency variant, not CUDA linked-pool progress. Caller-owned
// counts/permutation/offsets; synchronous.
namespace gslide::sycl_port {
inline void count_candidates(sycl::queue &q, const int *raw, const int *offsets,
                             int tables, int batch, int nodes, int capacity,
                             int *counts, int local = 64) {
  require(tables > 0 && batch >= 0 && nodes > 0 &&
              std::int64_t(batch) * nodes <= capacity,
          "candidate capacity");
  launch_policy(q, local);
  if (!batch)
    return;
  q.parallel_for(sycl::nd_range<1>(batch * local, local),
                 [=](sycl::nd_item<1> it) {
                   int b = it.get_group_linear_id();
                   for (int i = offsets[b * tables] + it.get_local_linear_id();
                        i < offsets[(b + 1) * tables]; i += local) {
                     sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>
                         c(counts[b * nodes + raw[i]]);
                     c.fetch_add(1);
                   }
                 })
      .wait_and_throw();
}
inline void activate_labels(sycl::queue &q, const int *labels,
                            const int *offsets, const int *permutation,
                            int batch, int nodes, int threshold, int minimum,
                            unsigned seed, int *counts, int *sizes,
                            int local = 64) {
  require(batch >= 0 && nodes > 0 && threshold > 0 && minimum >= 0 &&
              minimum <= nodes,
          "activation shape/minimum");
  launch_policy(q, local);
  if (!batch)
    return;
  q.parallel_for(sycl::nd_range<1>(padded(batch, local), local),
                 [=](sycl::nd_item<1> it) {
                   int b = it.get_global_linear_id();
                   if (b >= batch)
                     return;
                   int *row = counts + b * nodes;
                   int size = 0;
                   for (int n = 0; n < nodes; ++n)
                     size += row[n] >= threshold;
                   for (int i = offsets[b]; i < offsets[b + 1]; ++i) {
                     int n = labels[i];
                     if (row[n] < threshold)
                       ++size;
                     row[n] += threshold;
                   }
                   if (permutation) {
                     int i = random_at(seed, b) % nodes;
                     for (int visited = 0; size < minimum && visited < nodes;
                          ++visited) {
                       int n = permutation[i];
                       if (row[n] < threshold)
                         ++size;
                       row[n] += threshold;
                       i = (i + 1) % nodes;
                     }
                   }
                   sizes[b] = size;
                 })
      .wait_and_throw();
}
} // namespace gslide::sycl_port
