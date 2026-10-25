#include <cmath>
#include <gslide/sycl.hpp>
#include <stdexcept>
void other_sycl_translation_unit(sycl::queue &queue) {
  using namespace gslide::sycl_port;
  usm_array<float> x(queue, 2), w(queue, 2), b(queue, 1), y(queue, 1);
  x.assign({2.f, -.5f});
  w.assign({.25f, -2.f});
  b.assign({.1f});
  dense_product(queue, {read_only(x.view()), 1, 2, Layout::row_major},
                {read_only(w.view()), 1, 2, Layout::row_major},
                read_only(b.view()), {y.view(), 1, 1, Layout::row_major},
                Activation::identity);
  float result = 0;
  queue.memcpy(&result, y.p, sizeof(float)).wait_and_throw();
  if (std::abs(double(result) - 1.6) > 1e-6)
    throw std::runtime_error("installed component oracle");
}
