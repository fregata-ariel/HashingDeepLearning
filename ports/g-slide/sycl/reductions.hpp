#pragma once
// TRACE_TEST_ID: GSLIDE-SYCL-DEVICE-115
// Support: translated device reductions; not original CUDA-body execution.
#include "device_policy.hpp"

// Actual subgroup width and local-memory reduction, including padded
// identities.
namespace gslide::sycl_port {
inline void reductions(sycl::queue &q, const float *input, int n, float *sums,
                       float *maxima, float *subgroup_sums,
                       float *subgroup_maxima, int local = 64) {
  require(n >= 0, "reduction extent");
  launch_policy(q, local);
  if (!n)
    return;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> s(local, h), m(local, h);
     h.parallel_for(
         sycl::nd_range<1>(padded(n, local), local), [=](sycl::nd_item<1> it) {
           int i = it.get_global_linear_id(), lane = it.get_local_linear_id();
           auto sg = it.get_sub_group();
           float x = i < n ? input[i] : 0.f,
                 y = i < n ? input[i] : -std::numeric_limits<float>::max();
           auto ss = sycl::reduce_over_group(sg, x, sycl::plus<float>());
           auto sm = sycl::reduce_over_group(sg, y, sycl::maximum<float>());
           subgroup_sums[i] = ss;
           subgroup_maxima[i] = sm;
           s[lane] = x;
           m[lane] = y;
           it.barrier(sycl::access::fence_space::local_space);
           // Odd-sized groups are supported as well as powers of two.
           for (int active = local; active > 1;) {
             int next = (active + 1) / 2;
             if (lane < active - next) {
               s[lane] += s[lane + next];
               m[lane] = sycl::fmax(m[lane], m[lane + next]);
             }
             it.barrier(sycl::access::fence_space::local_space);
             active = next;
           }
           if (!lane) {
             sums[it.get_group_linear_id()] = s[0];
             maxima[it.get_group_linear_id()] = m[0];
           }
         });
   }).wait_and_throw();
}
} // namespace gslide::sycl_port
