#pragma once
#include "device_policy.hpp"

// Optimizer math is independent of model, layer count and parameter ownership.
namespace gslide::sycl_port {
// Positive signed Adam update; caller supplies bias-corrected rate.
// Accumulators are consumed and cleared. All four n-element arrays are borrowed
// USM.
inline void adam_update(sycl::queue &q, float *w, float *g, float *m, float *v,
                        int n, float rate) {
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
     int i = id[0];
     float grad = g[i];
     m[i] = .9f * m[i] + .1f * grad;
     v[i] = .999f * v[i] + .001f * grad * grad;
     w[i] += rate * m[i] / (sycl::sqrt(v[i]) + 1e-8f);
     g[i] = 0;
   }).wait_and_throw();
}
} // namespace gslide::sycl_port
