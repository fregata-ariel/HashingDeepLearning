#include <gslide/host.hpp>
#include <iostream>
#include <stdexcept>
int other_host_translation_unit();
int main() {
  using namespace gslide::sycl_port;
  NetworkConfig config{{3, 4, 5}, {3, 2, 4, 16, 2, 2, 1, 40}, {2, 6, 4}, {}};
  const auto plan = make_workspace_plan(config);
  if (plan.input_values != 6 || plan.gathered != 12 ||
      plan.bucket_values != 96 || other_host_translation_unit() != 3)
    throw std::runtime_error("installed host API oracle");
  std::cout << "GSLIDE_CONSUMER_HOST_PASS sycl_required=false\n";
}
