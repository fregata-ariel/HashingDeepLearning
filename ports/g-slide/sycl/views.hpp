#pragma once
#include "contracts.hpp"
#include <cstddef>
#include <limits>
#include <type_traits>

// SDK-free borrowed descriptors. Capacity is asserted by the allocation owner;
// it cannot be inferred from an arbitrary external pointer. No ownership.
namespace gslide::sycl_port {
inline int checked_extent(std::size_t a, std::size_t b = 1) {
  const auto limit = std::size_t(std::numeric_limits<int>::max() - 1);
  require(a <= limit && b <= limit && (!b || a <= limit / b),
          "extent exceeds supported integer range");
  return int(a * b);
}
template <class T> struct Span {
  T *data;
  std::size_t capacity;
  // Borrowed backend provenance. Host-only descriptors need none. The context
  // token and allocation must outlive every derived device view.
  const void *memory_context = nullptr;
  void require_size(std::size_t used) const {
    require(used <= capacity && (!capacity || data), "buffer capacity/null");
    checked_extent(capacity);
  }
};
template <class T> Span<const T> read_only(Span<T> span) {
  return {span.data, span.capacity, span.memory_context};
}
enum class Layout { row_major, column_major };
enum class Activation { identity, relu };
template <class T> struct MatrixView {
  Span<T> storage;
  int rows, columns;
  Layout layout;
  void validate() const {
    require(rows >= 0 && columns >= 0, "negative matrix extent");
    require(layout == Layout::row_major || layout == Layout::column_major,
            "matrix layout");
    storage.require_size(checked_extent(rows, columns));
  }
  int row_stride() const { return layout == Layout::row_major ? columns : 1; }
  int column_stride() const { return layout == Layout::row_major ? 1 : rows; }
};
template <class T> MatrixView<const T> read_only(MatrixView<T> view) {
  return {read_only(view.storage), view.rows, view.columns, view.layout};
}
// Segments group IDs by sample (the released CSC representation). IDs have a
// used count distinct from capacity. Offsets and ID payload validity are a
// separate contract; host payloads are checked by validate_sparse_batch.
struct SegmentedIdsView {
  Span<const int> offsets, ids;
  int segments, used;
  void validate() const {
    require(segments >= 0 && used >= 0, "negative segmented extent");
    offsets.require_size(std::size_t(segments) + 1);
    ids.require_size(used);
  }
};
} // namespace gslide::sycl_port
