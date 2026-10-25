#pragma once
#include "device_policy.hpp"

// minstd RNG/packed released variant. Borrowed USM; completes before return.
namespace gslide::sycl_port {
inline std::uint32_t random_at(std::uint32_t seed, std::uint64_t id) {
  std::uint64_t value = seed % 2147483647u;
  if (!value)
    value = 1;
  std::uint64_t power = 48271, exponent = id + 1;
  while (exponent) {
    if (exponent & 1)
      value = value * power % 2147483647u;
    power = power * power % 2147483647u;
    exponent >>= 1;
  }
  return value;
}
inline void random_keys(sycl::queue &q, unsigned *keys, int n, unsigned seed,
                        int previous, bool packed, int local = 64) {
  require(n >= 0 && previous > 0 && n / previous < 65536,
          "RNG shape/packed prefix");
  launch_policy(q, local);
  if (!n)
    return;
  q.parallel_for(sycl::nd_range<1>(padded(n, local), local),
                 [=](sycl::nd_item<1> it) {
                   int i = it.get_global_linear_id();
                   if (i < n) {
                     auto r = random_at(seed, i);
                     keys[i] = packed ? (unsigned(i / previous) << 16) |
                                            (r & (1u << 15))
                                      : r;
                   }
                 })
      .wait_and_throw();
}
} // namespace gslide::sycl_port
