// TRACE_TEST_ID: GSLIDE-SYCL-DEVICE-115
#include "reductions.hpp"
#include "test_support.hpp"
static void test_reductions(sycl::queue &q) {
  for (int local : {32, 64, 128})
    for (int n : {1, 31, 33, 65, 129}) {
      int extent = padded(n, local), groups = extent / local;
      memory<float> x(q, extent), s(q, groups), m(q, groups), ss(q, extent),
          sm(q, extent);
      memory<int> sgid(q, extent), sgbegin(q, extent), sgsize(q, extent);
      for (int i = 0; i < n; ++i)
        x[i] = float((i * 17) % 31 - 29) / 8;
      reductions(q, x.p, n, s.p, m.p, ss.p, sm.p, local);
      q.parallel_for(
           sycl::nd_range<1>(extent, local),
           [=, a = sgid.p, b = sgbegin.p, c = sgsize.p](sycl::nd_item<1> it) {
             int i = it.get_global_linear_id();
             auto sg = it.get_sub_group();
             a[i] = sg.get_group_linear_id();
             b[i] = it.get_group_linear_id();
             c[i] = sg.get_local_linear_range();
           })
          .wait_and_throw();
      for (int g = 0; g < groups; ++g) {
        double sum = 0;
        float maximum = -std::numeric_limits<float>::max();
        for (int i = g * local; i < std::min(n, (g + 1) * local); ++i) {
          sum += x[i];
          maximum = std::max(maximum, x[i]);
        }
        check(std::abs(s[g] - sum) < 1e-5, "block sum");
        check(m[g] == maximum, "block max");
      }
      for (int i = 0; i < extent; ++i) {
        double sum = 0;
        float maximum = -std::numeric_limits<float>::max();
        for (int j = 0; j < n; ++j)
          if (sgbegin[j] == sgbegin[i] && sgid[j] == sgid[i]) {
            sum += x[j];
            maximum = std::max(maximum, x[j]);
          }
        check(std::abs(ss[i] - sum) < 1e-5, "subgroup sum");
        check(sm[i] == maximum, "subgroup max");
      }
    }
}
int main() { return run_device_suite("reductions", test_reductions); }
