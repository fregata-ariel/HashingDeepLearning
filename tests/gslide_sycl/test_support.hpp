#pragma once
#include "hashing/sycl_backend.hpp"
#include "memory.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <vector>
using namespace gslide::sycl_port;
static int checks = 0;
static double max_error = 0;
static void check(bool value, const char *message) {
  ++checks;
  if (!value)
    throw std::runtime_error(message);
}
static void near(double actual, double expected, double tolerance = 8e-6) {
  max_error = std::max(max_error, std::abs(actual - expected));
  check(std::isfinite(actual) && std::abs(actual - expected) < tolerance,
        "FP64 component oracle mismatch");
}
template <class T> using memory = usm_array<T>;
template <class Function>
int run_device_suite(const char *name, Function function) {
  try {
    auto queue = hashing::sycl_backend::make_validation_queue();
    hashing::sycl_backend::describe_queue(std::cout, queue);
    function(queue);
    std::cout << "GSLIDE_SYCL_UNIT suite=" << name << " checks=" << checks
              << " max_abs_error=" << max_error
              << " gpu_validated=" << queue.get_device().is_gpu() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
