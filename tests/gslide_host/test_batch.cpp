#include "batch.hpp"
#include <iostream>
using namespace gslide::sycl_port;
int main() {
  const batch_limits limits{3, 4, 2, 3, 2};
  sparse_batch valid{{0, 0, 2}, {0, 2, 3}, {2, 1}, {0, 1, 2}, {.5f, .3f, 2.f}};
  validate_sparse_batch(valid, limits);
  validate_sparse_batch({{}, {0}, {}, {0}, {}}, limits);
  auto unlabeled = valid;
  unlabeled.labels.clear();
  unlabeled.label_offsets = {0, 0, 0};
  validate_sparse_batch(unlabeled, limits, false);
  int rejected = 0;
  auto reject = [&](const sparse_batch &batch) {
    try {
      validate_sparse_batch(batch, limits);
    } catch (const std::invalid_argument &) {
      ++rejected;
      return;
    }
    throw std::runtime_error("invalid sparse batch accepted");
  };
  reject(unlabeled);
  auto bad = valid;
  bad.offsets.clear();
  reject(bad);
  bad = valid;
  bad.values.pop_back();
  reject(bad);
  bad = valid;
  bad.offsets.back() = 2;
  reject(bad);
  bad = valid;
  bad.offsets = {0, 3, 2};
  reject(bad);
  bad = valid;
  bad.nodes[0] = 3;
  reject(bad);
  bad = valid;
  bad.labels[0] = 4;
  reject(bad);
  bad = valid;
  bad.values[0] = std::numeric_limits<float>::infinity();
  reject(bad);
  bad = valid;
  bad.label_offsets.pop_back();
  reject(bad);
  std::cout << "GSLIDE_HOST_UNIT suite=batch rejected=" << rejected
            << " sycl_required=false\n";
}
