#pragma once
#include "configuration.hpp"
#include "memory.hpp"

namespace gslide::sycl_port {
// Logical persistent model/optimizer state, independent of index/workspace.
struct ModelState {
  parameter hidden_weights, hidden_bias, output_weights, output_bias;
  ModelState(sycl::queue &q, const InitialState &initial)
      : hidden_weights(q, initial.hidden_weights),
        hidden_bias(q, initial.hidden_bias),
        output_weights(q, initial.output_weights),
        output_bias(q, initial.output_bias) {}
};
struct IndexState {
  usm_array<int> bins, permutation, bucket_nodes, bucket_sizes;
  IndexState(sycl::queue &q, const NetworkConfig &c, const WorkspacePlan &p,
             const std::vector<int> &initial_bins)
      : bins(q, p.bins), permutation(q, c.model.outputs),
        bucket_nodes(q, p.bucket_values), bucket_sizes(q, p.bucket_count) {
    bins.assign(initial_bins);
    for (int i = 0; i < c.model.outputs; ++i)
      permutation[i] = i;
  }
};
struct TrainingWorkspace {
  usm_array<int> input_nodes, input_offsets, label_nodes, label_offsets;
  usm_array<float> input_values, dense_inputs, hidden_values, hidden_deltas;
  usm_array<float> probabilities, output_deltas;
  usm_array<int> query_addresses, lengths, gather_offsets, gathered, counts,
      sizes;
  usm_array<int> active_nodes, active_offsets, sort_values, sorted_values;
  usm_array<unsigned> random_keys, sorted_keys;
  TrainingWorkspace(sycl::queue &q, const NetworkConfig &c,
                    const WorkspacePlan &p)
      : input_nodes(q, c.limits.input_nnz),
        input_offsets(q, c.limits.max_batch + 1),
        label_nodes(q, c.limits.label_nnz),
        label_offsets(q, c.limits.max_batch + 1),
        input_values(q, c.limits.input_nnz), dense_inputs(q, p.input_values),
        hidden_values(q, p.hidden_values), hidden_deltas(q, p.hidden_values),
        probabilities(q, p.output_values), output_deltas(q, p.output_values),
        query_addresses(q, p.queries), lengths(q, p.queries),
        gather_offsets(q, p.queries + 1), gathered(q, p.gathered),
        counts(q, p.output_values), sizes(q, c.limits.max_batch),
        active_nodes(q, p.output_values),
        active_offsets(q, c.limits.max_batch + 1),
        sort_values(q, p.sort_scratch), sorted_values(q, p.sort_scratch),
        random_keys(q, p.sort_scratch), sorted_keys(q, p.sort_scratch) {}
};
} // namespace gslide::sycl_port
