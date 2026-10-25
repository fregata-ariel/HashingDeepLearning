#include "views.hpp"
#include <iostream>
using namespace gslide::sycl_port;
static int checks = 0;
static void check(bool ok) {
  ++checks;
  if (!ok)
    throw std::runtime_error("view oracle");
}
template <class F> static void rejects(F operation) {
  try {
    operation();
  } catch (const std::invalid_argument &) {
    ++checks;
    return;
  }
  throw std::runtime_error("invalid view accepted");
}
int main() {
  float values[12]{};
  MatrixView<float> row{{values, 12}, 2, 3, Layout::row_major};
  row.validate();
  check(row.row_stride() == 3 && row.column_stride() == 1);
  MatrixView<float> column{{values, 12}, 2, 3, Layout::column_major};
  column.validate();
  check(column.row_stride() == 1 && column.column_stride() == 2);
  auto borrowed = read_only(row);
  static_assert(std::is_same_v<decltype(borrowed.storage.data), const float *>);
  check(borrowed.storage.data == values && borrowed.storage.capacity == 12);
  rejects([&] {
    MatrixView<float>{{values, 5}, 2, 3, Layout::row_major}.validate();
  });
  rejects([&] {
    MatrixView<float>{{nullptr, 6}, 2, 3, Layout::row_major}.validate();
  });
  rejects([&] {
    MatrixView<float>{{values, 12}, -1, 3, Layout::row_major}.validate();
  });
  rejects(
      [&] { MatrixView<float>{{values, 12}, 2, 3, Layout(17)}.validate(); });
  rejects([&] { checked_extent(std::numeric_limits<std::size_t>::max(), 2); });
  rejects([&] { checked_extent(65536, 65536); });
  int offsets[]{0, 2, 3}, ids[]{1, 0, 2, 99, 99};
  SegmentedIdsView segmented{{offsets, 3}, {ids, 5}, 2, 3};
  segmented.validate(); // 3 used entries in a 5-entry allocation
  check(segmented.used == 3 && segmented.ids.capacity == 5);
  rejects([&] {
    auto v = segmented;
    v.used = 6;
    v.validate();
  });
  rejects([&] {
    auto v = segmented;
    v.offsets.capacity = 2;
    v.validate();
  });
  MatrixView<float>{{nullptr, 0}, 0, 3, Layout::row_major}.validate();
  std::cout << "GSLIDE_HOST_UNIT suite=views checks=" << checks
            << " sycl_required=false\n";
}
