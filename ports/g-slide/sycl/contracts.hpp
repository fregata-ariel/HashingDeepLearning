#pragma once
#include <stdexcept>

// Pure host precondition helper; no runtime/device dependency.
namespace gslide::sycl_port {
inline void require(bool ok, const char *msg) {
  if (!ok)
    throw std::invalid_argument(msg);
}
} // namespace gslide::sycl_port
