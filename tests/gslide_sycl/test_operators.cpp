#include "operators.hpp"
#include "test_support.hpp"
static void test_library_boundary(sycl::queue &q) {
  usm_array<int> lengths(q, 4), offsets(q, 5);
  for (int i = 0; i < 4; ++i)
    lengths[i] = i % 3;
  scan_lengths(q, lengths.p, offsets.p, 4);
  int sum = 0;
  for (int i = 0; i <= 4; ++i) {
    check(offsets[i] == sum, "scan independent offsets");
    if (i < 4)
      sum += lengths[i];
  }
  scan_lengths(q, lengths.p, offsets.p, 0);
  check(offsets[0] == 0, "empty scan");
  usm_array<unsigned> keys(q, 5), sorted(q, 5);
  usm_array<int> values(q, 5), result(q, 5);
  unsigned k[] = {3, 1, 1, 4, 0};
  for (int i = 0; i < 5; ++i) {
    keys[i] = k[i];
    values[i] = 10 + i;
  }
  sort_keys(q, keys.p, values.p, sorted.p, result.p, 5);
  std::multiset<std::pair<unsigned, int>> expected, actual;
  for (int i = 0; i < 5; ++i) {
    expected.emplace(keys[i], values[i]);
    actual.emplace(sorted[i], result[i]);
    if (i)
      check(sorted[i - 1] <= sorted[i], "sort order");
  }
  check(actual == expected, "sort correspondence/ties");
  sort_keys(q, keys.p, values.p, sorted.p, result.p, 0);
  usm_array<float> x(q, 6), row(q, 6), col(q, 6), bias(q, 2), yr(q, 4),
      yc(q, 4);
  for (int i = 0; i < 6; ++i)
    x[i] = (i - 2) * .3f;
  for (int o = 0; o < 2; ++o)
    for (int i = 0; i < 3; ++i)
      row[o * 3 + i] = col[i * 2 + o] = float(i - o) * .4f;
  bias[0] = -.7f;
  bias[1] = .2f;
  dense_product(q, x.p, row.p, bias.p, yr.p, 2, 3, 2, true, false);
  dense_product(q, x.p, col.p, bias.p, yc.p, 2, 3, 2, false, false);
  for (int b = 0; b < 2; ++b)
    for (int o = 0; o < 2; ++o) {
      double expected = bias[o];
      for (int i = 0; i < 3; ++i)
        expected += double(x[b * 3 + i]) * row[o * 3 + i];
      near(yr[b * 2 + o], expected);
      near(yc[b * 2 + o], expected);
    }
  dense_product(q, x.p, row.p, bias.p, yr.p, 0, 3, 2, true, false);
}
int main() { return run_device_suite("operators", test_library_boundary); }
