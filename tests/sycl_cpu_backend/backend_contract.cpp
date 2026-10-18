#include <hashing/sycl_backend.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::size_t width = 32;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct free_usm {
  sycl::queue* queue;
  void operator()(int* pointer) const { sycl::free(pointer, *queue); }
};
using allocation = std::unique_ptr<int, free_usm>;
allocation allocate(std::size_t n, sycl::queue& queue) {
  auto* pointer = sycl::malloc_shared<int>(n, queue);
  require(pointer != nullptr, "USM allocation failed");
  return allocation(pointer, free_usm{&queue});
}

// The host oracle sums each scalar independently. The device implementation
// uses padded work-groups, local memory, barriers and an atomic visit counter.
void check_case(sycl::queue& queue, std::size_t n) {
  const auto groups = (n + width - 1) / width;
  auto input = allocate(n + 2, queue);
  auto output = allocate(n + 2, queue);
  auto partial = allocate(groups + 2, queue);
  auto state = allocate(2, queue);
  std::vector<int> expected(n);
  int expected_sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    input.get()[i + 1] = static_cast<int>(i % 13) - 6;
    expected[i] = input.get()[i + 1] * input.get()[i + 1] + 3;
    expected_sum += expected[i];
  }
  input.get()[0] = input.get()[n + 1] = -917;
  for (std::size_t i = 0; i < n + 2; ++i) output.get()[i] = -719;
  for (std::size_t i = 0; i < groups + 2; ++i) partial.get()[i] = -519;
  state.get()[0] = state.get()[1] = 0;
  int* in = input.get();
  int* out = output.get();
  int* sums = partial.get();
  int* counter = state.get();
  if (n != 0) {
    queue.submit([&](sycl::handler& handler) {
      sycl::local_accessor<int, 1> scratch(sycl::range<1>{width}, handler);
      handler.parallel_for(sycl::nd_range<1>{groups * width, width}, [=](sycl::nd_item<1> item) {
        const auto i = item.get_global_linear_id();
        const auto lane = item.get_local_linear_id();
        int value = 0;
        if (i < n) {
          value = in[i + 1] * in[i + 1] + 3;
          out[i + 1] = value;
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(*counter).fetch_add(1);
        }
        scratch[lane] = value;
        item.barrier(sycl::access::fence_space::local_space);
        for (std::size_t stride = width / 2; stride > 0; stride /= 2) {
          if (lane < stride) scratch[lane] += scratch[lane + stride];
          item.barrier(sycl::access::fence_space::local_space);
        }
        if (lane == 0) sums[item.get_group_linear_id() + 1] = scratch[0];
      });
    });
    queue.submit([&](sycl::handler& handler) {
      auto reduction = sycl::reduction(counter + 1, sycl::plus<int>{});
      handler.parallel_for(sycl::range<1>{n}, reduction, [=](sycl::id<1> i, auto& sum) {
        sum.combine(out[i[0] + 1]);
      });
    });
    queue.wait_and_throw();
  }
  int actual_sum = 0;
  for (std::size_t group = 0; group < groups; ++group) {
    int expected_group = 0;
    for (std::size_t i = group * width; i < n && i < (group + 1) * width; ++i)
      expected_group += expected[i];
    require(sums[group + 1] == expected_group, "local memory/barrier reduction mismatch");
    actual_sum += sums[group + 1];
  }
  for (std::size_t i = 0; i < n; ++i)
    require(out[i + 1] == expected[i], "global index/value mismatch");
  require(out[0] == -719 && out[n + 1] == -719, "output guard modified");
  require(in[0] == -917 && in[n + 1] == -917, "input guard modified");
  require(sums[0] == -519 && sums[groups + 1] == -519, "group guard modified");
  require(counter[0] == static_cast<int>(n), "atomic counter mismatch");
  require(actual_sum == expected_sum && counter[1] == expected_sum, "independent sum mismatch");
  std::cout << "CASE PASS n=" << n << " groups=" << groups
            << " visits=" << counter[0] << " sum=" << actual_sum << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: sycl_cpu_backend_test boundaries.txt");
    const auto start = std::chrono::steady_clock::now();
    auto queue = hashing::sycl_backend::make_queue(hashing::sycl_backend::device_kind::cpu);
    hashing::sycl_backend::require_cpu(queue);
    hashing::sycl_backend::describe_queue(std::cout, queue);
    require(queue.get_device().has(sycl::aspect::usm_shared_allocations), "required shared USM unavailable");
    require(queue.get_device().get_info<sycl::info::device::max_work_group_size>() >= width,
            "required work-group size unavailable");
    std::cout << "runtime_initialization_ms=" << std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count() << '\n';
    std::ifstream fixtures(argv[1]);
    require(fixtures.good(), "missing fixture");
    std::size_t n = 0, cases = 0;
    while (fixtures >> n) {
      require(n <= 1025, "fixture exceeds bounded contract");
      check_case(queue, n);
      ++cases;
    }
    require(fixtures.eof() && cases == 9, "malformed/incomplete fixture");
    std::cout << "SYCL_CPU_BACKEND_PASS cases=" << cases << '\n';
  } catch (const std::exception& error) {
    std::cerr << "SYCL_CPU_BACKEND_FAIL " << error.what() << '\n';
    return 1;
  }
}
