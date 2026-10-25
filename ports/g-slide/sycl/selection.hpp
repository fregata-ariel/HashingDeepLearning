#pragma once
#include "candidates.hpp"
#include "hash_index.hpp"
#include "operators.hpp"

// LSH query/gather/count/label/padding/filter composition. Synchronous;
// direct-address workspace capacity is explicit.
namespace gslide::sycl_port {
// Borrowed query inputs and caller-owned work buffers. No allocation or index
// lifetime is hidden in this operation; buffers are provisioned by the caller.
struct candidate_buffers {
  const int *bins, *random_nodes, *bucket_sizes, *bucket_nodes;
  const float *hidden_values;
  const int *label_nodes, *label_offsets;
  int *query_addresses, *lengths, *gather_offsets, *gathered, *counts, *sizes;
  int *active_offsets, *active_nodes;
};
inline void select_candidates(sycl::queue &q, hash_shape shape,
                              int current_batch, int hidden, int outputs,
                              int count_capacity, int threshold, int minimum,
                              unsigned seed, candidate_buffers buffers) {

  int nb = current_batch, no = outputs, nh = hidden;
  auto z = shape;
  auto bp = buffers.bins;
  auto hp = buffers.hidden_values;
  auto sp = buffers.bucket_sizes;
  auto query = buffers.query_addresses;
  auto len = buffers.lengths;
  if (nb)
    q.parallel_for(sycl::range<1>(nb * z.tables), [=](sycl::id<1> id) {
       int s = id[0] / z.tables, t = id[0] % z.tables;
       int a = address(bp + t * z.k * z.bin, hp + s * nh, z.k, z.bin,
                       int(sycl::log(float(z.bin))));
       query[id[0]] = a;
       len[id[0]] = sycl::min(sp[t * z.buckets + a], z.capacity);
     }).wait_and_throw();
  scan_lengths(q, buffers.lengths, buffers.gather_offsets, nb * z.tables);
  auto go = buffers.gather_offsets;
  auto raw = buffers.gathered;
  auto buckets = buffers.bucket_nodes;
  if (nb)
    q.parallel_for(sycl::range<1>(nb * z.tables), [=](sycl::id<1> id) {
       int t = id[0] % z.tables, a = t * z.buckets + query[id[0]];
       for (int p = go[id[0]]; p < go[id + 1]; ++p)
         raw[p] = buckets[a * z.capacity + p - go[id[0]]];
     }).wait_and_throw();
  q.fill(buffers.counts, 0, count_capacity).wait_and_throw();
  count_candidates(q, buffers.gathered, buffers.gather_offsets, z.tables, nb,
                   no, count_capacity, buffers.counts);
  activate_labels(q, buffers.label_nodes, buffers.label_offsets,
                  buffers.random_nodes, nb, no, threshold, minimum, seed,
                  buffers.counts, buffers.sizes);
  scan_lengths(q, buffers.sizes, buffers.active_offsets, nb);
  auto freq = buffers.counts;
  auto offsets = buffers.active_offsets;
  auto ids = buffers.active_nodes;
  int th = threshold;
  if (nb)
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
       int b = id[0], pos = offsets[b];
       for (int n = 0; n < no; ++n)
         if (freq[b * no + n] >= th)
           ids[pos++] = n;
     }).wait_and_throw();
}
} // namespace gslide::sycl_port
