#include <array>
#include <cmath>
#include <gslide/sycl.hpp>
#include <iostream>
#include <stdexcept>
void other_sycl_translation_unit(sycl::queue &queue);
static sycl::queue make_queue() {
#ifdef GSLIDE_CONSUMER_REQUIRE_NVIDIA
  for (const auto &device :
       sycl::device::get_devices(sycl::info::device_type::gpu))
    if (device.get_info<sycl::info::device::vendor>().find("NVIDIA") !=
        std::string::npos)
      return sycl::queue(device, sycl::property::queue::in_order{});
  throw std::runtime_error("required NVIDIA device unavailable");
#else
  return sycl::queue(sycl::cpu_selector{}, sycl::property::queue::in_order{});
#endif
}
int main() {
  using namespace gslide::sycl_port;
  auto queue = make_queue();
  std::cout << "device="
            << queue.get_device().get_info<sycl::info::device::name>()
            << " is_cpu=" << queue.get_device().is_cpu()
            << " is_gpu=" << queue.get_device().is_gpu() << '\n';
  other_sycl_translation_unit(queue);
  NetworkConfig config{{2, 2, 2}, {1, 1, 2, 2, 2, 1, 2, 40}, {1, 2, 1}, {}};
  InitialState initial{{1.f, .5f, 0.f, 2.f},
                       {.1f, -.2f},
                       {.2f, -.3f, .5f, .1f},
                       {.05f, -.1f},
                       {0, 1}};
  sparse_batch batch{{0, 1}, {0, 2}, {1}, {0, 1}, {.4f, .3f}};
  // Independent FP64 scalar oracle for a fully selected, one-sample update.
  std::array<double, 2> hidden{};
  for (int h = 0; h < 2; ++h) {
    hidden[h] = initial.hidden_bias[h];
    for (int i = 0; i < 2; ++i)
      hidden[h] += double(batch.values[i]) * initial.hidden_weights[i * 2 + h];
    hidden[h] = std::max(0., hidden[h]);
  }
  std::array<double, 2> logits{}, delta{}, hidden_delta{};
  for (int o = 0; o < 2; ++o) {
    logits[o] = initial.output_bias[o];
    for (int h = 0; h < 2; ++h)
      logits[o] += hidden[h] * initial.output_weights[o * 2 + h];
  }
  double maximum = std::max(logits[0], logits[1]);
  double sum = std::exp(logits[0] - maximum) + std::exp(logits[1] - maximum);
  for (int o = 0; o < 2; ++o)
    delta[o] = (o == 1 ? 1. : 0.) - std::exp(logits[o] - maximum) / sum;
  for (int h = 0; h < 2; ++h)
    if (hidden[h] > 0)
      for (int o = 0; o < 2; ++o)
        hidden_delta[h] += delta[o] * initial.output_weights[o * 2 + h];
  constexpr float rate = .01f;
  auto updated = [](float value, double gradient) {
    return double(value) + double(rate) * (.1 * gradient) /
                               (std::sqrt(.001 * gradient * gradient) + 1e-8);
  };
  std::array<double, 2> next_hidden{}, expected{};
  for (int h = 0; h < 2; ++h) {
    next_hidden[h] = updated(initial.hidden_bias[h], hidden_delta[h]);
    for (int i = 0; i < 2; ++i)
      next_hidden[h] +=
          double(batch.values[i]) * updated(initial.hidden_weights[i * 2 + h],
                                            hidden_delta[h] * batch.values[i]);
    next_hidden[h] = std::max(0., next_hidden[h]);
  }
  for (int o = 0; o < 2; ++o) {
    expected[o] = updated(initial.output_bias[o], delta[o]);
    for (int h = 0; h < 2; ++h)
      expected[o] += next_hidden[h] * updated(initial.output_weights[o * 2 + h],
                                              delta[o] * hidden[h]);
  }
  Network network(queue, config, initial);
  network.train(batch, StepOptions{rate, true, false});
  usm_array<float> output(queue, 2);
  network.evaluate_dense(batch, {output.view(), 1, 2, Layout::row_major});
  std::array<float, 2> actual{};
  queue.memcpy(actual.data(), output.p, sizeof(actual)).wait_and_throw();
  double max_error = 0;
  for (int o = 0; o < 2; ++o) {
    max_error = std::max(max_error, std::abs(double(actual[o]) - expected[o]));
    if (!std::isfinite(actual[o]) || max_error > 8e-6)
      throw std::runtime_error("installed training/evaluation oracle");
  }
  if (std::abs(double(actual[0]) - logits[0]) < 1e-5)
    throw std::runtime_error("training did not change logits");
  std::cout << "GSLIDE_CONSUMER_SYCL_PASS max_abs_error=" << max_error << '\n';
}
