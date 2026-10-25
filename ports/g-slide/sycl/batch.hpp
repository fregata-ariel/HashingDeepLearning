#pragma once
#include "contracts.hpp"
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

// Host-only sparse input/configuration contracts. No queue, allocation or GPU
// is needed to validate a batch.
namespace gslide::sycl_port {
struct sparse_batch {
  std::vector<int> nodes, offsets, labels, label_offsets;
  std::vector<float> values;
  int size() const { return offsets.size() - 1; }
};
enum class delta_reset { each_step, diagnostic_keep_stale };
struct batch_limits {
  int inputs, outputs, max_batch, input_capacity, label_capacity;
};
inline void validate_sparse_batch(const sparse_batch &b, batch_limits limits,
                                  bool training_labels = true) {
  const auto [inputs, outputs, max_batch, input_capacity, label_capacity] =
      limits;

  require(b.offsets.size() >= 1 && b.label_offsets.size() == b.offsets.size() &&
              b.size() <= max_batch,
          "batch extent");
  require(b.nodes.size() == b.values.size() &&
              b.nodes.size() <= unsigned(input_capacity) &&
              b.labels.size() <= unsigned(label_capacity),
          "batch capacity");
  require(b.offsets.front() == 0 && b.label_offsets.front() == 0 &&
              b.offsets.back() == int(b.nodes.size()) &&
              b.label_offsets.back() == int(b.labels.size()),
          "compressed endpoints");
  for (int i = 0; i < b.size(); ++i) {
    require(b.offsets[i] <= b.offsets[i + 1] &&
                b.label_offsets[i] <= b.label_offsets[i + 1],
            "monotone offsets");
    if (training_labels)
      require(b.label_offsets[i] < b.label_offsets[i + 1],
              "training requires labels");
  }
  for (int n : b.nodes)
    require(n >= 0 && n < inputs, "input ID");
  for (int n : b.labels)
    require(n >= 0 && n < outputs, "label ID");
  for (float x : b.values)
    require(std::isfinite(x), "finite input");
}
inline int validate_network_configuration(
    int i, int h, int o, int nb, int ic, int lc, int tables, int k, int bin,
    int buckets, int threshold, int minimum, const std::vector<float> &hw,
    const std::vector<float> &hb, const std::vector<float> &ow,
    const std::vector<float> &ob, const std::vector<int> &bins) {
  require(i > 0 && h > 0 && o > 0 && nb > 0 && ic >= 0 && lc >= 0 &&
              tables > 0 && k > 0 && bin > 0 && buckets > 0 && threshold > 0 &&
              minimum >= 0 && minimum <= o,
          "network configuration");
  const auto limit = std::int64_t(std::numeric_limits<int>::max());
  auto check_extent = [&](std::initializer_list<int> dimensions) {
    std::int64_t extent = 1;
    for (int dimension : dimensions) {
      require(dimension > 0 && extent <= (limit - 1) / dimension,
              "network extent overflow");
      extent *= dimension;
    }
  };
  check_extent({nb, i});
  check_extent({nb, h});
  check_extent({nb, o});
  check_extent({i, h});
  check_extent({o, h});
  check_extent({tables, k, bin});
  check_extent({nb, tables, o});
  check_extent({tables, buckets, o});
  require(hw.size() == unsigned(i * h) && hb.size() == unsigned(h) &&
              ow.size() == unsigned(o * h) && ob.size() == unsigned(o),
          "parameter shape");
  for (const auto *vector : {&hw, &hb, &ow, &ob})
    for (float value : *vector)
      require(std::isfinite(value), "finite parameter");
  require(bins.size() == unsigned(tables * k * bin), "bins extent");
  for (int id : bins)
    require(id >= 0 && id < h, "bin ID");
  return i;
}
} // namespace gslide::sycl_port
