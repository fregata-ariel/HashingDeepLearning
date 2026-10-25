#pragma once
#include "contracts.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

// Borrowed queue; explicit local-size validation. No device selection here.
namespace gslide::sycl_port {
inline std::size_t padded(std::size_t n, std::size_t local) {
  return (n + local - 1) / local * local;
}
inline void launch_policy(sycl::queue &q, int local) {
  require(local > 0 &&
              std::size_t(local) <=
                  q.get_device()
                      .get_info<sycl::info::device::max_work_group_size>(),
          "local size exceeds device");
}
} // namespace gslide::sycl_port
