#include "component_api.hpp"
#include "test_support.hpp"
static void test(sycl::queue &queue) {
  memory<float> w(queue, 3), g(queue, 3), m(queue, 3), v(queue, 3);
  double expected_w[] = {1, -2, .5}, expected_m[] = {0, 0, 0},
         expected_v[] = {0, 0, 0};
  for (int i = 0; i < 3; ++i)
    w[i] = expected_w[i];
  for (int step = 0; step < 3; ++step) {
    const float rate = step == 2 ? 0.f : .01f;
    for (int i = 0; i < 3; ++i) {
      const float gradient = float((step + 1) * (i - 1)) * .3f;
      g[i] = gradient;
      expected_m[i] = .9 * expected_m[i] + .1 * gradient;
      expected_v[i] = .999 * expected_v[i] + .001 * double(gradient) * gradient;
      expected_w[i] +=
          double(rate) * expected_m[i] / (std::sqrt(expected_v[i]) + 1e-8);
    }
    adam_update(queue, {w.view(), g.view(), m.view(), v.view()}, 3, rate);
    for (int i = 0; i < 3; ++i) {
      near(w[i], expected_w[i]);
      near(m[i], expected_m[i]);
      near(v[i], expected_v[i]);
      near(g[i], 0, 1e-12);
    }
  }
  bool rejected = false;
  const float before = w[0];
  try {
    adam_update(queue, {w.view(), g.view(), m.view(), {v.p, 2}}, 3, .01f);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected && w[0] == before,
        "short optimizer state rejected before update");
}
int main() { return run_device_suite("optimizer", test); }
