#pragma once
#include "device_policy.hpp"
#include <vector>

// Owns a queue copy/context and USM, not the caller queue object. Library
// operations complete before release; external submissions must also complete.
// Noncopyable.
namespace gslide::sycl_port {
template <class T> class usm_array {
  sycl::queue queue;

public:
  T *p;
  int size;
  static T *allocate(sycl::queue &q, int n) {
    require(n >= 0, "allocation extent");
    return sycl::malloc_shared<T>(std::max(n, 1), q);
  }
  usm_array(sycl::queue &q, int n) : queue(q), p(allocate(q, n)), size(n) {
    if (!p)
      throw std::bad_alloc();
    q.fill(p, T{}, std::max(n, 1)).wait_and_throw();
  }
  ~usm_array() { sycl::free(p, queue); }
  usm_array(const usm_array &) = delete;
  usm_array &operator=(const usm_array &) = delete;
  T &operator[](int i) { return p[i]; }
  const T &operator[](int i) const { return p[i]; }
  void assign(const std::vector<T> &v) {
    require(v.size() <= unsigned(size), "transfer capacity");
    if (!v.empty())
      queue.memcpy(p, v.data(), v.size() * sizeof(T)).wait_and_throw();
  }
};
struct parameter {
  usm_array<float> value, gradient, moment, velocity;
  parameter(sycl::queue &q, const std::vector<float> &initial)
      : value(q, initial.size()), gradient(q, initial.size()),
        moment(q, initial.size()), velocity(q, initial.size()) {
    value.assign(initial);
  }
};
} // namespace gslide::sycl_port
