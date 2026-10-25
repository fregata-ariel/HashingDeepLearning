#pragma once
#include "device_policy.hpp"

// Selected-output forward/backward primitives; no Network object or buffer
// allocation. Calls complete and propagate asynchronous errors.
namespace gslide::sycl_port {
/** Borrowed selected-output CSC and dense hidden/weight arrays. Weights are
 * [output,hidden], activations [batch,hidden]; probabilities/deltas use CSC
 * slots. Nonempty training labels and at least one active output per nonempty
 * sample are caller preconditions. Released membership/label-count semantics
 * are preserved: duplicate labels do not form a normalized target distribution.
 * All pointers belong to q's USM context. */
inline void selected_softmax(sycl::queue &q, int nb, int nh, const int *off,
                             const int *ids, const float *hv, const float *w,
                             const float *bias, float *prob, float *delta,
                             const int *labs, const int *loff) {
  if (nb)
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
       int s = id[0];
       float maximum = -std::numeric_limits<float>::max();
       for (int p = off[s]; p < off[s + 1]; ++p) {
         int n = ids[p];
         float v = bias[n];
         for (int h = 0; h < nh; ++h)
           v += hv[s * nh + h] * w[n * nh + h];
         prob[p] = v;
         maximum = sycl::fmax(maximum, v);
       }
       float sum = 0;
       for (int p = off[s]; p < off[s + 1]; ++p) {
         prob[p] = sycl::exp(prob[p] - maximum);
         sum += prob[p];
       }
       for (int p = off[s]; p < off[s + 1]; ++p) {
         prob[p] /= sum;
         bool label = false;
         for (int j = loff[s]; j < loff[s + 1]; ++j)
           label |= labs[j] == ids[p];
         delta[p] =
             ((label ? 1.f / (loff[s + 1] - loff[s]) : 0.f) - prob[p]) / nb;
       }
     }).wait_and_throw();
}
// Accumulates hidden deltas and parameter gradients; caller owns reset cadence.
inline void selected_output_backward(sycl::queue &q, int nb, int nh, int no,
                                     const int *off, const int *ids,
                                     const float *hv, const float *d,
                                     const float *w, float *hd, float *gw,
                                     float *gb) {
  if (nb)
    q.parallel_for(sycl::range<1>(nb * no), [=](sycl::id<1> id) {
       int b = id[0] / no, p = off[b] + id[0] % no;
       if (p >= off[b + 1])
         return;
       for (int h = 0; h < nh; ++h)
         if (hv[b * nh + h] > 0) {
           sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
               a(hd[b * nh + h]);
           a.fetch_add(d[p] * w[ids[p] * nh + h]);
         }
     }).wait_and_throw();
  q.parallel_for(sycl::range<1>(no * nh), [=](sycl::id<1> id) {
     int o = id[0] / nh, h = id[0] % nh;
     float g = 0;
     for (int b = 0; b < nb; ++b)
       for (int p = off[b]; p < off[b + 1]; ++p)
         if (ids[p] == o)
           g += d[p] * hv[b * nh + h];
     gw[id[0]] += g;
   }).wait_and_throw();
  q.parallel_for(sycl::range<1>(no), [=](sycl::id<1> id) {
     float g = 0;
     for (int b = 0; b < nb; ++b)
       for (int p = off[b]; p < off[b + 1]; ++p)
         if (ids[p] == int(id[0]))
           g += d[p];
     gb[id[0]] += g;
   }).wait_and_throw();
}
// Dense input [batch,input], hidden deltas [batch,hidden], column-major dW.
inline void hidden_parameter_gradients(sycl::queue &q, int nb, int ni, int nh,
                                       const float *x, const float *d,
                                       float *gw, float *gb) {
  q.parallel_for(sycl::range<1>(ni * nh), [=](sycl::id<1> id) {
     int i = id[0] / nh, h = id[0] % nh;
     float g = 0;
     for (int b = 0; b < nb; ++b)
       g += x[b * ni + i] * d[b * nh + h];
     gw[id[0]] += g;
   }).wait_and_throw();
  q.parallel_for(sycl::range<1>(nh), [=](sycl::id<1> id) {
     float g = 0;
     for (int b = 0; b < nb; ++b)
       g += d[b * nh + id[0]];
     gb[id[0]] += g;
   }).wait_and_throw();
}
} // namespace gslide::sycl_port
