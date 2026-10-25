#pragma once
#include "../include/TrainingControl.h"
#include "batch.hpp"
#include "kernels.hpp"
#include "memory.hpp"
#include "operators.hpp"
#include "optimizer.hpp"
#include "output_ops.hpp"
#include "selection.hpp"
#include <memory>

namespace gslide::sycl_port {
// TRACE_TEST_ID: GSLIDE-SYCL-PRODUCTION-116
// Support: shared production ordering with a two-layer SYCL numerical backend.
class Network {
  static int validate_configuration(
      int i, int h, int o, int nb, int ic, int lc, int tables, int k, int bin,
      int buckets, int threshold, int minimum, const std::vector<float> &hw,
      const std::vector<float> &hb, const std::vector<float> &ow,
      const std::vector<float> &ob, const std::vector<int> &bins) {
    return validate_network_configuration(i, h, o, nb, ic, lc, tables, k, bin,
                                          buckets, threshold, minimum, hw, hb,
                                          ow, ob, bins);
  }
  sycl::queue &q;
  int inputs, hidden, outputs, max_batch, input_capacity, label_capacity;
  hash_shape shape;
  usm_array<int> input_nodes, input_offsets, label_nodes, label_offsets,
      hidden_ids, hidden_offsets;
  usm_array<float> input_values, dense_inputs;
  usm_array<int> bins, random_nodes, query_addresses, lengths, gather_offsets,
      gathered, counts, sizes;
  usm_array<unsigned> random_keys_buffer, sorted_keys;
  usm_array<int> sort_values, sorted_values;
  int current_batch = 0;
  void validate_batch(const sparse_batch &b, bool training_labels = true) {
    validate_sparse_batch(
        b, {inputs, outputs, max_batch, input_capacity, label_capacity},
        training_labels);
  }
  void transfer(const sparse_batch &b, bool training_labels = true) {
    validate_batch(b, training_labels);
    current_batch = b.size();
    input_nodes.assign(b.nodes);
    input_values.assign(b.values);
    input_offsets.assign(b.offsets);
    label_nodes.assign(b.labels);
    label_offsets.assign(b.label_offsets);
    q.fill(dense_inputs.p, 0.f, max_batch * inputs).wait_and_throw();
    auto ids = input_nodes.p;
    auto val = input_values.p;
    auto off = input_offsets.p;
    auto dst = dense_inputs.p;
    int ni = inputs;
    if (current_batch)
      q.parallel_for(sycl::range<1>(current_batch), [=](sycl::id<1> id) {
         int s = id[0];
         for (int j = off[s]; j < off[s + 1]; ++j)
           dst[s * ni + ids[j]] += val[j];
       }).wait_and_throw();
  }
  void query() {
    select_candidates(q, shape, current_batch, hidden, outputs,
                      max_batch * outputs, threshold, minimum, seed,
                      {bins.p, random_nodes.p, bucket_sizes.p, bucket_nodes.p,
                       hidden_values.p, label_nodes.p, label_offsets.p,
                       query_addresses.p, lengths.p, gather_offsets.p,
                       gathered.p, counts.p, sizes.p, active_offsets.p,
                       active_nodes.p});
  }
  void output_forward() {
    query();
    selected_softmax(q, current_batch, hidden, active_offsets.p, active_nodes.p,
                     hidden_values.p, output_weights.value.p,
                     output_bias.value.p, probabilities.p, output_deltas.p,
                     label_nodes.p, label_offsets.p);
  }
  void output_backward() {
    selected_output_backward(
        q, current_batch, hidden, outputs, active_offsets.p, active_nodes.p,
        hidden_values.p, output_deltas.p, output_weights.value.p,
        hidden_deltas.p, output_weights.gradient.p, output_bias.gradient.p);
  }
  void hidden_backward() {
    hidden_parameter_gradients(q, current_batch, inputs, hidden, dense_inputs.p,
                               hidden_deltas.p, hidden_weights.gradient.p,
                               hidden_bias.gradient.p);
  }
  void update(parameter &p, float rate) {
    adam_update(q, p.value.p, p.gradient.p, p.moment.p, p.velocity.p,
                p.value.size, rate);
  }

public:
  parameter hidden_weights, hidden_bias, output_weights, output_bias;
  usm_array<float> hidden_values, hidden_deltas, probabilities, output_deltas;
  usm_array<int> active_nodes, active_offsets, bucket_nodes, bucket_sizes;
  int rebuild_count = 0, step_count = 0, threshold, minimum;
  unsigned seed;
  Network(sycl::queue &q, int i, int h, int o, int maxbatch, int incap,
          int labcap, const std::vector<float> &hw,
          const std::vector<float> &hb, const std::vector<float> &ow,
          const std::vector<float> &ob,
          const std::vector<int> &permutation_bins, int tables = 3, int k = 2,
          int bin = 4, int buckets = 16, int threshold = 2, int minimum = 1,
          unsigned seed = 40)
      : q(q),
        inputs(validate_configuration(i, h, o, maxbatch, incap, labcap, tables,
                                      k, bin, buckets, threshold, minimum, hw,
                                      hb, ow, ob, permutation_bins)),
        hidden(h), outputs(o), max_batch(maxbatch), input_capacity(incap),
        label_capacity(labcap), shape{h, o, tables, k, bin, 2, 2, buckets, o},
        input_nodes(q, incap), input_offsets(q, maxbatch + 1),
        label_nodes(q, labcap), label_offsets(q, maxbatch + 1),
        hidden_ids(q, maxbatch * h), hidden_offsets(q, maxbatch + 1),
        input_values(q, incap), dense_inputs(q, maxbatch * i),
        bins(q, tables * k * bin), random_nodes(q, o),
        query_addresses(q, maxbatch * tables), lengths(q, maxbatch * tables),
        gather_offsets(q, maxbatch * tables + 1),
        gathered(q, maxbatch * tables * o), counts(q, maxbatch * o),
        sizes(q, maxbatch),
        random_keys_buffer(q, std::max(o, tables * k * bin)),
        sorted_keys(q, std::max(o, tables * k * bin)),
        sort_values(q, std::max(o, tables * k * bin)),
        sorted_values(q, std::max(o, tables * k * bin)), hidden_weights(q, hw),
        hidden_bias(q, hb), output_weights(q, ow), output_bias(q, ob),
        hidden_values(q, maxbatch * h), hidden_deltas(q, maxbatch * h),
        probabilities(q, maxbatch * o), output_deltas(q, maxbatch * o),
        active_nodes(q, maxbatch * o), active_offsets(q, maxbatch + 1),
        bucket_nodes(q, tables * buckets * o),
        bucket_sizes(q, tables * buckets), threshold(threshold),
        minimum(minimum), seed(seed) {
    require(i > 0 && h > 0 && o > 0 && maxbatch > 0 && incap >= 0 &&
                labcap >= 0,
            "network shape");
    require(hw.size() == unsigned(i * h) && hb.size() == unsigned(h) &&
                ow.size() == unsigned(o * h) && ob.size() == unsigned(o),
            "parameter shape");
    require(permutation_bins.size() == unsigned(tables * k * bin),
            "bins extent");
    for (int id : permutation_bins)
      require(id >= 0 && id < h, "bin ID");
    bins.assign(permutation_bins);
    for (int n = 0; n < o; ++n)
      random_nodes[n] = n;
    rebuild(false);
  }
  void rebuild(bool reshuffle) {
    if (reshuffle) {
      int n = bins.size;
      require(n % hidden == 0, "shuffle bins requires complete permutations");
      random_keys(q, random_keys_buffer.p, n, seed, hidden, true);
      for (int i = 0; i < n; ++i)
        sort_values[i] = bins[i];
      sort_keys(q, random_keys_buffer.p, sort_values.p, sorted_keys.p,
                sorted_values.p, n);
      q.memcpy(bins.p, sorted_values.p, n * sizeof(int)).wait_and_throw();
      random_keys(q, random_keys_buffer.p, outputs, seed, outputs, false);
      for (int i = 0; i < outputs; ++i)
        sort_values[i] = random_nodes[i];
      sort_keys(q, random_keys_buffer.p, sort_values.p, sorted_keys.p,
                sorted_values.p, outputs);
      q.memcpy(random_nodes.p, sorted_values.p, outputs * sizeof(int))
          .wait_and_throw();
    }
    q.fill(bucket_sizes.p, 0, bucket_sizes.size).wait_and_throw();
    build_hash(q, bins.p, output_weights.value.p, shape, bucket_nodes.p,
               bucket_sizes.p);
    ++rebuild_count;
  }
  void train(const sparse_batch &batch, float corrected_rate,
             bool rebuild_index, bool reshuffle,
             delta_reset reset = delta_reset::each_step) {
    require(std::isfinite(corrected_rate) && corrected_rate >= 0,
            "finite nonnegative learning rate");
    validate_batch(batch);
    if (batch.size() == 0)
      return;
    TrainingOperations ops;
    ops.hidden_layers = 1;
    ops.transfer = [&] { transfer(batch); };
    ops.hidden_forward = [&](int) {
      dense_ids(q, hidden_ids.p, hidden_offsets.p, hidden, current_batch,
                max_batch * hidden);
      dense_product(q, dense_inputs.p, hidden_weights.value.p,
                    hidden_bias.value.p, hidden_values.p, current_batch, inputs,
                    hidden, false, true);
    };
    ops.output_forward = [&] { output_forward(); };
    ops.reset_hidden_deltas = [&] {
      if (reset == delta_reset::each_step)
        q.fill(hidden_deltas.p, 0.f, hidden_deltas.size).wait_and_throw();
    };
    ops.output_backward = [&] { output_backward(); };
    ops.hidden_backward = [&](int) { hidden_backward(); };
    ops.hidden_update = [&](int) {
      update(hidden_weights, corrected_rate);
      update(hidden_bias, corrected_rate);
    };
    ops.output_update = [&] {
      update(output_weights, corrected_rate);
      update(output_bias, corrected_rate);
    };
    ops.rebuild_index = [&](bool shuffle) { rebuild(shuffle); };
    ops.complete = [&] { q.wait_and_throw(); };
    train_step(ops, rebuild_index, reshuffle);
    ++step_count;
  }
  const int *bin_data() const { return bins.p; }
  // Numerical variant: includes learned output bias. The native cuBLAS
  // forward_dense currently omits it. Evaluation accepts unlabeled samples.
  void evaluate_dense(const sparse_batch &batch, float *logits) {
    transfer(batch, false);
    dense_product(q, dense_inputs.p, hidden_weights.value.p,
                  hidden_bias.value.p, hidden_values.p, current_batch, inputs,
                  hidden, false, true);
    dense_product(q, hidden_values.p, output_weights.value.p,
                  output_bias.value.p, logits, current_batch, hidden, outputs,
                  true, false);
  }
};
} // namespace gslide::sycl_port
