#include <gslide/host.hpp>
int other_host_translation_unit() {
  float values[6]{};
  gslide::sycl_port::MatrixView<float> view{
      {values, 6}, 2, 3, gslide::sycl_port::Layout::row_major};
  view.validate();
  return view.row_stride();
}
