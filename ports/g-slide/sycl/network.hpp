#pragma once
#include "../include/TrainingControl.h"
#include "component_api.hpp"
#include "storage.hpp"

namespace gslide::sycl_port {
// TRACE_TEST_ID: GSLIDE-SYCL-PRODUCTION-116
// Support: shared production ordering with a two-layer SYCL numerical backend.
class Network {
  static NetworkConfig checked_configuration(sycl::queue &queue,
                                             NetworkConfig c,
                                             const InitialState &initial) {
    c = validate_configuration(c, initial);
    launch_policy(queue, c.execution.local_size);
    return c;
  }
  sycl::queue
      q; // Retained context/device; caller queue object may be destroyed.
  sycl::context memory_context;
  NetworkConfig config;
  const WorkspacePlan plan;
  hash_shape shape;
  ModelState model;
  IndexState index;
  TrainingWorkspace work;
  int current_batch = 0, current_labels = 0;
  SegmentedIdsView active{};
  template <class T>
  MatrixView<T> matrix(usm_array<T> &a, int rows, int columns,
                       Layout layout = Layout::row_major) {
    return {a.view(), rows, columns, layout};
  }
  SegmentedIdsView labels() const {
    return {work.label_offsets.view(), work.label_nodes.view(), current_batch,
            current_labels};
  }
  SelectedOutput selected() {
    return {
        read_only(matrix(hidden_values, current_batch, config.model.hidden)),
        read_only(matrix(output_weights.value, config.model.outputs,
                         config.model.hidden)),
        active};
  }
  void validate_batch(const sparse_batch &batch, bool training = true) const {
    const auto m = config.model;
    const auto l = config.limits;
    validate_sparse_batch(
        batch, {m.inputs, m.outputs, l.max_batch, l.input_nnz, l.label_nnz},
        training);
  }
  void transfer(const sparse_batch &batch, bool training = true) {
    validate_batch(batch, training);
    current_batch = batch.size();
    current_labels = int(batch.labels.size());
    work.input_nodes.assign(batch.nodes);
    work.input_values.assign(batch.values);
    work.input_offsets.assign(batch.offsets);
    work.label_nodes.assign(batch.labels);
    work.label_offsets.assign(batch.label_offsets);
    q.fill(work.dense_inputs.p, 0.f, plan.input_values).wait_and_throw();
    auto ids = work.input_nodes.p;
    auto values = work.input_values.p;
    auto offsets = work.input_offsets.p;
    auto dense = work.dense_inputs.p;
    const int inputs = config.model.inputs;
    if (current_batch)
      q.parallel_for(sycl::range<1>(current_batch), [=](sycl::id<1> id) {
         int sample = id[0];
         for (int j = offsets[sample]; j < offsets[sample + 1]; ++j)
           dense[sample * inputs + ids[j]] += values[j];
       }).wait_and_throw();
  }
  void hidden_forward() {
    dense_product(q,
                  read_only(matrix(work.dense_inputs, current_batch,
                                   config.model.inputs)),
                  read_only(matrix(hidden_weights.value, config.model.hidden,
                                   config.model.inputs, Layout::column_major)),
                  read_only(hidden_bias.value.view()),
                  matrix(hidden_values, current_batch, config.model.hidden),
                  Activation::relu);
  }
  void output_forward() {
    active = select_candidates(
        q,
        {shape, read_only(index.bins.view()),
         read_only(index.permutation.view()), read_only(bucket_sizes.view()),
         read_only(bucket_nodes.view())},
        read_only(matrix(hidden_values, current_batch, config.model.hidden)),
        labels(), threshold, minimum, seed,
        {work.query_addresses.view(), work.lengths.view(),
         work.gather_offsets.view(), work.gathered.view(), work.counts.view(),
         work.sizes.view(), active_offsets.view(), active_nodes.view()});
    selected_softmax(q, selected(), read_only(output_bias.value.view()),
                     labels(), probabilities.view(), output_deltas.view());
  }
  void output_backward() {
    accumulate_output_gradients(
        q, selected(), read_only(output_deltas.view()),
        {matrix(hidden_deltas, current_batch, config.model.hidden),
         matrix(output_weights.gradient, config.model.outputs,
                config.model.hidden),
         output_bias.gradient.view()});
  }
  void hidden_backward() {
    accumulate_hidden_gradients(
        q,
        read_only(
            matrix(work.dense_inputs, current_batch, config.model.inputs)),
        read_only(matrix(hidden_deltas, current_batch, config.model.hidden)),
        matrix(hidden_weights.gradient, config.model.hidden,
               config.model.inputs, Layout::column_major),
        hidden_bias.gradient.view());
  }
  void update(parameter &p, float rate) {
    adam_update(
        q,
        {p.value.view(), p.gradient.view(), p.moment.view(), p.velocity.view()},
        p.value.size, rate);
  }

public:
  // Compatibility diagnostics. New consumers use configuration()/typed
  // operations; these mutable aliases are not a stable application API.
  parameter &hidden_weights = model.hidden_weights,
            &hidden_bias = model.hidden_bias;
  parameter &output_weights = model.output_weights,
            &output_bias = model.output_bias;
  usm_array<float> &hidden_values = work.hidden_values,
                   &hidden_deltas = work.hidden_deltas;
  usm_array<float> &probabilities = work.probabilities,
                   &output_deltas = work.output_deltas;
  usm_array<int> &active_nodes = work.active_nodes,
                 &active_offsets = work.active_offsets;
  usm_array<int> &bucket_nodes = index.bucket_nodes,
                 &bucket_sizes = index.bucket_sizes;
  int rebuild_count = 0, step_count = 0;
  int &threshold = config.index.threshold, &minimum = config.index.minimum;
  unsigned &seed = config.index.seed;

  Network(sycl::queue &queue, NetworkConfig c, const InitialState &initial)
      : q(queue), memory_context(q.get_context()),
        config(checked_configuration(queue, c, initial)),
        plan(make_workspace_plan(config)), shape{c.model.hidden,
                                                 c.model.outputs,
                                                 c.index.tables,
                                                 c.index.hashes_per_table,
                                                 c.index.bin_size,
                                                 c.execution.tables_per_tile,
                                                 c.execution.tables_per_thread,
                                                 c.index.buckets_per_table,
                                                 c.index.bucket_capacity},
        model(q, initial), index(q, config, plan, initial.bins),
        work(q, config, plan) {
    launch_policy(q, config.execution.local_size);
    rebuild(false);
  }
  // Historical positional constructor delegates to the checked named API.
  Network(sycl::queue &queue, int i, int h, int o, int maxbatch, int incap,
          int labcap, const std::vector<float> &hw,
          const std::vector<float> &hb, const std::vector<float> &ow,
          const std::vector<float> &ob, const std::vector<int> &bins,
          int tables = 3, int k = 2, int bin = 4, int buckets = 16,
          int threshold = 2, int minimum = 1, unsigned seed = 40)
      : Network(queue,
                {{i, h, o},
                 {tables, k, bin, buckets, o, threshold, minimum, seed},
                 {maxbatch, incap, labcap},
                 {}},
                {hw, hb, ow, ob, bins}) {}
  Network(const Network &) = delete;
  Network &operator=(const Network &) = delete;
  const NetworkConfig &configuration() const { return config; }
  const WorkspacePlan &workspace_plan() const { return plan; }
  void rebuild(bool reshuffle) {
    if (reshuffle) {
      int n = index.bins.size;
      const int hidden = config.model.hidden, outputs = config.model.outputs;
      require(n % hidden == 0, "shuffle bins requires complete permutations");
      random_keys(q, work.random_keys.p, n, seed, hidden, true);
      for (int i = 0; i < n; ++i)
        work.sort_values[i] = index.bins[i];
      sort_keys(q, work.random_keys.p, work.sort_values.p, work.sorted_keys.p,
                work.sorted_values.p, n);
      q.memcpy(index.bins.p, work.sorted_values.p, n * sizeof(int))
          .wait_and_throw();
      random_keys(q, work.random_keys.p, outputs, seed, outputs, false);
      for (int i = 0; i < outputs; ++i)
        work.sort_values[i] = index.permutation[i];
      sort_keys(q, work.random_keys.p, work.sort_values.p, work.sorted_keys.p,
                work.sorted_values.p, outputs);
      q.memcpy(index.permutation.p, work.sorted_values.p, outputs * sizeof(int))
          .wait_and_throw();
    }
    q.fill(bucket_sizes.p, 0, bucket_sizes.size).wait_and_throw();
    build_hash(q, index.bins.p, output_weights.value.p, shape, bucket_nodes.p,
               bucket_sizes.p, config.execution.local_size);
    ++rebuild_count;
  }
  void train(const sparse_batch &batch, StepOptions options) {
    train(batch, options.corrected_rate, options.rebuild, options.reshuffle);
  }
  void train(const sparse_batch &batch, float corrected_rate,
             bool rebuild_index, bool reshuffle,
             delta_reset reset = delta_reset::each_step) {
    require(std::isfinite(corrected_rate) && corrected_rate >= 0,
            "finite nonnegative learning rate");
    validate_batch(batch);
    if (!batch.size())
      return;
    TrainingOperations ops;
    ops.hidden_layers = 1;
    ops.transfer = [&] { transfer(batch); };
    ops.hidden_forward = [&](int) { hidden_forward(); };
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
  const int *bin_data() const { return index.bins.p; }
  // Maintained model includes output bias. The archived SGEMM-only eval omits
  // it.
  void evaluate_dense(const sparse_batch &batch, MatrixView<float> logits) {
    validate_batch(batch, false);
    require_matrix(logits, batch.size(), config.model.outputs,
                   Layout::row_major);
    validate_usm(q, logits); // reject before mutating workspace
    transfer(batch, false);
    hidden_forward();
    dense_product(
        q, read_only(matrix(hidden_values, current_batch, config.model.hidden)),
        read_only(matrix(output_weights.value, config.model.outputs,
                         config.model.hidden)),
        read_only(output_bias.value.view()), logits, Activation::identity);
  }
  // Compatibility caller asserts capacity; prefer the MatrixView overload.
  void evaluate_dense(const sparse_batch &batch, float *logits) {
    evaluate_dense(
        batch, {borrow_usm(memory_context, logits,
                           std::size_t(checked_extent(batch.size(),
                                                      config.model.outputs))),
                batch.size(), config.model.outputs, Layout::row_major});
  }
};
} // namespace gslide::sycl_port
