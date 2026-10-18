#pragma once

#include <sycl/sycl.hpp>
#include <exception>
#include <ostream>
#include <stdexcept>

// Shared queue policy only: maintained numerical kernels accept a caller queue,
// allowing CPU contract tests and separately validated GPU configurations.
namespace hashing::sycl_backend {
enum class device_kind { automatic, cpu, gpu };

inline sycl::queue make_queue(device_kind kind = device_kind::automatic) {
  const auto errors = [](sycl::exception_list exceptions) {
    for (const auto& error : exceptions) std::rethrow_exception(error);
  };
  const sycl::property_list properties{sycl::property::queue::in_order{},
                                       sycl::property::queue::enable_profiling{}};
  if (kind == device_kind::cpu)
    return sycl::queue{sycl::cpu_selector_v, errors, properties};
  if (kind == device_kind::gpu)
    return sycl::queue{sycl::gpu_selector_v, errors, properties};
  return sycl::queue{sycl::default_selector_v, errors, properties};
}

inline void require_cpu(const sycl::queue& queue) {
  if (!queue.get_device().is_cpu())
    throw std::runtime_error("required CPU device missing; no host fallback permitted");
}

inline void describe_queue(std::ostream& out, const sycl::queue& queue) {
  const auto device = queue.get_device();
  const auto platform = device.get_platform();
  out << "platform=" << platform.get_info<sycl::info::platform::name>()
      << "\nplatform_version=" << platform.get_info<sycl::info::platform::version>()
      << "\nbackend_enum=" << static_cast<int>(queue.get_backend())
      << "\ndevice=" << device.get_info<sycl::info::device::name>()
      << "\ndevice_vendor=" << device.get_info<sycl::info::device::vendor>()
      << "\ndriver=" << device.get_info<sycl::info::device::driver_version>()
      << "\nis_cpu=" << device.is_cpu() << "\nis_gpu=" << device.is_gpu()
      << "\nmax_work_group_size=" << device.get_info<sycl::info::device::max_work_group_size>()
      << "\nlocal_mem_bytes=" << device.get_info<sycl::info::device::local_mem_size>()
      << "\nusm_shared=" << device.has(sycl::aspect::usm_shared_allocations)
      << "\nqueue_in_order=" << queue.has_property<sycl::property::queue::in_order>()
      << "\nqueue_profiling=" << queue.has_property<sycl::property::queue::enable_profiling>()
      << "\nsub_group_sizes=";
  for (auto size : device.get_info<sycl::info::device::sub_group_sizes>()) out << size << ',';
  out << '\n';
}
}  // namespace hashing::sycl_backend
