#include "memory.hpp"
#include "test_support.hpp"
static void test(sycl::queue &queue) {
  std::unique_ptr<usm_array<int>> retained;
  {
    auto short_lived_queue = queue;
    retained = std::make_unique<usm_array<int>>(short_lived_queue, 3);
    retained->assign({2, 4, 6});
  }
  check((*retained)[2] == 6, "allocation outlives caller queue object");
  retained.reset(); // frees using the retained queue/context, not a dangling
                    // reference
  usm_array<float> empty(queue, 0);
  empty.assign({});
  bool rejected = false;
  try {
    empty.assign({1});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "transfer capacity");
  parameter state(queue, {1.f, -2.f});
  for (int i = 0; i < 2; ++i) {
    near(state.gradient[i], 0);
    near(state.moment[i], 0);
    near(state.velocity[i], 0);
  }
}
int main() { return run_device_suite("memory", test); }
