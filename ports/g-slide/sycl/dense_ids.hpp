#pragma once
#include "device_policy.hpp"

// Emit dense IDs and batch+1 offsets into caller-owned USM; synchronous.
namespace gslide::sycl_port {
inline void dense_ids(sycl::queue &q, int *ids, int *offsets, int nodes,
                      int batch, int capacity, int local = 64) {
  require(nodes >= 0 && batch >= 0 && std::int64_t(nodes) * batch <= capacity,
          "dense capacity");
  launch_policy(q, local);
  auto n = std::max(std::size_t(nodes) * batch, std::size_t(batch) + 1);
  q.parallel_for(sycl::nd_range<1>(padded(n, local), local),
                 [=](sycl::nd_item<1> it) {
                   auto i = it.get_global_linear_id();
                   if (i < std::size_t(nodes) * batch)
                     ids[i] = i % nodes;
                   if (i <= std::size_t(batch))
                     offsets[i] = i * nodes;
                 })
      .wait_and_throw();
}
} // namespace gslide::sycl_port
