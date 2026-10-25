#pragma once
#include "device_policy.hpp"

// Synchronous reference device operators: quadratic scan/rank-sort,
// row/column-major dense product. No optimized-library performance claim.
namespace gslide::sycl_port {
inline void scan_lengths(sycl::queue &q, const int *lengths, int *offsets,
                         int n) {
  require(n >= 0, "scan extent");
  q.parallel_for(sycl::range<1>(n + 1), [=](sycl::id<1> id) {
     int i = id[0], s = 0;
     for (int j = 0; j < i; ++j)
       s += lengths[j];
     offsets[i] = s;
   }).wait_and_throw();
}
inline void sort_keys(sycl::queue &q, const unsigned *keys, const int *values,
                      unsigned *out_keys, int *out_values, int n) {
  require(n >= 0, "sort extent");
  if (!n)
    return;
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
     int i = id[0], rank = 0;
     for (int j = 0; j < n; ++j)
       rank += keys[j] < keys[i] || (keys[j] == keys[i] && j < i);
     out_keys[rank] = keys[i];
     out_values[rank] = values[i];
   }).wait_and_throw();
}
inline void dense_product(sycl::queue &q, const float *x, const float *w,
                          const float *bias, float *y, int batch, int inputs,
                          int outputs, bool weights_rowmajor, bool relu) {
  require(batch >= 0 && inputs > 0 && outputs > 0, "dense shape");
  if (!batch)
    return;
  q.parallel_for(sycl::range<1>(batch * outputs), [=](sycl::id<1> id) {
     int b = id[0] / outputs, o = id[0] % outputs;
     float v = bias[o];
     for (int i = 0; i < inputs; ++i)
       v += x[b * inputs + i] *
            w[weights_rowmajor ? o * inputs + i : i * outputs + o];
     y[id[0]] = relu ? sycl::fmax(0.f, v) : v;
   }).wait_and_throw();
}
} // namespace gslide::sycl_port
