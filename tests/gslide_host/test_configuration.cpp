#include "configuration.hpp"
#include <iostream>
using namespace gslide::sycl_port;
static int checks = 0;
static void check(bool ok) {
  ++checks;
  if (!ok)
    throw std::runtime_error("plan oracle");
}
template <class F> static void rejects(F operation) {
  try {
    operation();
  } catch (const std::invalid_argument &) {
    ++checks;
    return;
  }
  throw std::runtime_error("invalid configuration accepted");
}
int main() {
  NetworkConfig c{{3, 4, 5}, {3, 2, 4, 16, 2, 2, 1, 40}, {2, 6, 4}, {64, 2, 2}};
  auto plan = make_workspace_plan(c);
  check(plan.input_values == 6 && plan.hidden_values == 8 &&
        plan.output_values == 10);
  check(plan.queries == 6 &&
        plan.gathered == 12); // bucket capacity, not output count
  check(plan.bucket_count == 48 && plan.bucket_values == 96);
  check(plan.bins == 24 && plan.sort_scratch == 24);
  InitialState initial{std::vector<float>(12), std::vector<float>(4),
                       std::vector<float>(20), std::vector<float>(5),
                       std::vector<int>(24)};
  validate_configuration(c, initial);
  rejects([&] {
    auto bad = c;
    bad.index.buckets_per_table = 6;
    make_workspace_plan(bad);
  });
  rejects([&] {
    auto bad = c;
    bad.limits.max_batch = 1 << 30;
    make_workspace_plan(bad);
  });
  rejects([&] {
    auto bad = c;
    bad.index.bucket_capacity = 1 << 30;
    make_workspace_plan(bad);
  });
  rejects([&] {
    auto bad = c;
    bad.execution.tables_per_thread = 3;
    make_workspace_plan(bad);
  });
  rejects([&] {
    auto bad = c;
    bad.index.hashes_per_table = 1 << 30;
    make_workspace_plan(bad);
  });
  rejects([&] {
    auto bad = initial;
    bad.hidden_weights.pop_back();
    validate_configuration(c, bad);
  });
  rejects([&] {
    auto bad = initial;
    bad.bins[0] = 4;
    validate_configuration(c, bad);
  });
  rejects([&] {
    auto bad = initial;
    bad.output_bias[0] = std::numeric_limits<float>::quiet_NaN();
    validate_configuration(c, bad);
  });
  c.index.bin_size = 1;
  c.index.hashes_per_table = 2;
  c.index.buckets_per_table = 1;
  check(make_workspace_plan(c).bins == 6);
  // The plan must depend on actual bucket capacity, not on a hypothetical
  // full-output allocation in every bucket. No device allocation is made.
  NetworkConfig sparse_index{
      {1, 1, 100000}, {2, 1, 1, 20000, 1, 2, 1, 40}, {1, 0, 0}, {}};
  InitialState sparse_initial{{0.f},
                              {0.f},
                              std::vector<float>(100000),
                              std::vector<float>(100000),
                              {0, 0}};
  validate_configuration(sparse_index, sparse_initial);
  check(make_workspace_plan(sparse_index).bucket_values == 40000);
  std::cout << "GSLIDE_HOST_UNIT suite=configuration checks=" << checks
            << " sycl_required=false\n";
}
