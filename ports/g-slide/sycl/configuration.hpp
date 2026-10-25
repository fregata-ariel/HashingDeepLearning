#pragma once
#include "batch.hpp"
#include "views.hpp"

namespace gslide::sycl_port {
struct ModelSpec {
  int inputs, hidden, outputs;
};
// Released natural-log packing and unordered bounded ring are fixed semantics,
// not tuning switches. See docs/GSLIDE_SEMANTICS.md.
struct IndexConfig {
  int tables = 3, hashes_per_table = 2, bin_size = 4;
  int buckets_per_table = 16, bucket_capacity = 0;
  int threshold = 2, minimum = 1;
  unsigned seed = 40;
};
struct WorkspaceLimits {
  int max_batch, input_nnz, label_nnz;
};
struct ExecutionOptions {
  int local_size = 64, tables_per_tile = 2, tables_per_thread = 2;
};
struct NetworkConfig {
  ModelSpec model;
  IndexConfig index;
  WorkspaceLimits limits;
  ExecutionOptions execution{};
};
struct InitialState {
  std::vector<float> hidden_weights, hidden_bias, output_weights, output_bias;
  std::vector<int> bins;
};
struct WorkspacePlan {
  int input_values, hidden_values, output_values, queries, gathered;
  int bucket_count, bucket_values, bins, sort_scratch;
};
inline WorkspacePlan make_workspace_plan(const NetworkConfig &c) {
  const auto m = c.model;
  const auto i = c.index;
  const auto l = c.limits;
  const auto e = c.execution;
  require(m.inputs > 0 && m.hidden > 0 && m.outputs > 0 && l.max_batch > 0 &&
              l.input_nnz >= 0 && l.label_nnz >= 0 && i.tables > 0 &&
              i.hashes_per_table > 0 && i.bin_size > 0 &&
              i.buckets_per_table > 0 && i.bucket_capacity > 0 &&
              i.threshold > 0 && i.minimum >= 0 && i.minimum <= m.outputs,
          "network configuration");
  require(e.local_size > 0 && e.tables_per_tile > 0 &&
              e.tables_per_thread > 0 &&
              e.tables_per_tile % e.tables_per_thread == 0 &&
              e.local_size % (e.tables_per_tile / e.tables_per_thread) == 0,
          "execution partition");
  checked_extent(l.max_batch);
  checked_extent(l.input_nnz);
  checked_extent(l.label_nnz);
  checked_extent(m.inputs, m.hidden);
  checked_extent(m.outputs, m.hidden);
  checked_extent(l.max_batch, 64); // candidate primitive work-group extent
  checked_extent(m.outputs, e.tables_per_tile / e.tables_per_thread);
  const int bins =
      checked_extent(checked_extent(i.tables, i.hashes_per_table), i.bin_size);
  checked_extent(checked_extent(e.tables_per_tile, i.hashes_per_table),
                 i.bin_size);
  checked_extent(e.local_size / (e.tables_per_tile / e.tables_per_thread),
                 m.hidden);
  const int shift = int(std::log(float(i.bin_size)));
  require(std::int64_t(i.hashes_per_table - 1) * shift < 30, "hash shift");
  std::uint64_t maximum = 0;
  if (!shift)
    maximum = std::uint64_t(i.bin_size - 1) * i.hashes_per_table;
  else
    for (int k = 0; k < i.hashes_per_table; ++k)
      maximum += std::uint64_t(i.bin_size - 1) << (k * shift);
  require(maximum < std::uint64_t(i.buckets_per_table),
          "bucket address capacity");
  checked_extent(std::size_t(i.tables) +
                 std::size_t(i.threshold) * l.label_nnz + i.threshold);
  const int queries = checked_extent(l.max_batch, i.tables);
  const int bucket_count = checked_extent(i.tables, i.buckets_per_table);
  return {checked_extent(l.max_batch, m.inputs),
          checked_extent(l.max_batch, m.hidden),
          checked_extent(l.max_batch, m.outputs),
          queries,
          checked_extent(queries, i.bucket_capacity),
          bucket_count,
          checked_extent(bucket_count, i.bucket_capacity),
          bins,
          std::max(m.outputs, bins)};
}
inline NetworkConfig validate_configuration(NetworkConfig c,
                                            const InitialState &initial) {
  const auto plan = make_workspace_plan(c);
  const auto m = c.model;
  require(initial.hidden_weights.size() ==
                  std::size_t(checked_extent(m.inputs, m.hidden)) &&
              initial.hidden_bias.size() == std::size_t(m.hidden) &&
              initial.output_weights.size() ==
                  std::size_t(checked_extent(m.outputs, m.hidden)) &&
              initial.output_bias.size() == std::size_t(m.outputs),
          "parameter shape");
  for (const auto *values : {&initial.hidden_weights, &initial.hidden_bias,
                             &initial.output_weights, &initial.output_bias})
    for (float value : *values)
      require(std::isfinite(value), "finite parameter");
  require(initial.bins.size() == std::size_t(plan.bins), "bins extent");
  for (int id : initial.bins)
    require(id >= 0 && id < m.hidden, "bin ID");
  return c;
}
struct StepOptions {
  float corrected_rate;
  bool rebuild = false, reshuffle = false;
};
} // namespace gslide::sycl_port
