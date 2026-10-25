#pragma once
#include "operators.hpp"
#include "optimizer.hpp"
#include "output_ops.hpp"
#include "selection.hpp"
#include "views.hpp"

namespace gslide::sycl_port {
// External memory adapter: caller asserts the allocation belongs to context,
// capacity is truthful, and context/allocation outlive every derived view.
// Nonconst lvalue reference prevents borrowing a temporary context object.
template <class T>
Span<T> borrow_usm(sycl::context &context, T *data, std::size_t capacity) {
  return {data, capacity, &context};
}
// Checked descriptors do not prove the true allocation extent of external
// pointers. The caller guarantees that capacity is truthful, operands do not
// alias destructively, and segmented payloads contain valid offsets/IDs.
template <class T> inline void validate_usm(sycl::queue &q, Span<T> span) {
  span.require_size(0);
  if (span.capacity) {
    require(span.memory_context != nullptr,
            "missing memory context provenance");
    require(*static_cast<const sycl::context *>(span.memory_context) ==
                q.get_context(),
            "USM context mismatch");
    const auto kind = sycl::get_pointer_type(span.data, q.get_context());
    require(kind != sycl::usm::alloc::unknown, "USM context mismatch");
    if (kind == sycl::usm::alloc::device)
      require(sycl::get_pointer_device(span.data, q.get_context()) ==
                  q.get_device(),
              "USM device mismatch");
  }
}
template <class T>
inline void validate_usm(sycl::queue &q, MatrixView<T> view) {
  view.validate();
  validate_usm(q, view.storage);
}
inline void validate_usm(sycl::queue &q, SegmentedIdsView view) {
  view.validate();
  validate_usm(q, view.offsets);
  validate_usm(q, view.ids);
}
template <class T>
inline void require_matrix(MatrixView<T> v, int rows, int columns,
                           Layout layout) {
  v.validate();
  require(v.rows == rows && v.columns == columns && v.layout == layout,
          "matrix shape/layout mismatch");
}
inline void dense_product(sycl::queue &q, MatrixView<const float> x,
                          MatrixView<const float> weights,
                          Span<const float> bias, MatrixView<float> y,
                          Activation activation) {
  validate_usm(q, x);
  validate_usm(q, weights);
  validate_usm(q, bias);
  validate_usm(q, y);
  require(x.columns > 0 && weights.rows > 0 && weights.columns == x.columns,
          "dense product shape");
  require(activation == Activation::identity || activation == Activation::relu,
          "activation");
  require_matrix(x, x.rows, x.columns, Layout::row_major);
  require_matrix(y, x.rows, weights.rows, Layout::row_major);
  bias.require_size(weights.rows);
  dense_product(q, x.storage.data, weights.storage.data, bias.data,
                y.storage.data, x.rows, x.columns, weights.rows,
                weights.layout == Layout::row_major,
                activation == Activation::relu);
}
struct AdamBuffers {
  Span<float> values, gradients, moments, velocities;
};
// Consumes gradients and clears them, including at corrected_rate == 0.
inline void adam_update(sycl::queue &q, AdamBuffers state, int used,
                        float corrected_rate) {
  require(used >= 0 && std::isfinite(corrected_rate) && corrected_rate >= 0,
          "Adam extent/rate");
  for (auto span :
       {state.values, state.gradients, state.moments, state.velocities}) {
    span.require_size(used);
    validate_usm(q, span);
  }
  if (used)
    adam_update(q, state.values.data, state.gradients.data, state.moments.data,
                state.velocities.data, used, corrected_rate);
}
struct SelectedOutput {
  MatrixView<const float> hidden,
      weights; // weights [output, hidden], row-major
  SegmentedIdsView active;
  void validate(sycl::queue &q) const {
    validate_usm(q, hidden);
    validate_usm(q, weights);
    validate_usm(q, active);
    require(hidden.columns > 0 && weights.rows > 0 &&
                active.segments == hidden.rows,
            "selected output shape");
    require(active.used <= checked_extent(hidden.rows, weights.rows),
            "selected output extent");
    require_matrix(hidden, hidden.rows, hidden.columns, Layout::row_major);
    require_matrix(weights, weights.rows, hidden.columns, Layout::row_major);
  }
};
inline void selected_softmax(sycl::queue &q, SelectedOutput input,
                             Span<const float> bias, SegmentedIdsView labels,
                             Span<float> probabilities, Span<float> deltas) {
  input.validate(q);
  validate_usm(q, labels);
  require(labels.segments == input.active.segments, "label batch mismatch");
  bias.require_size(input.weights.rows);
  validate_usm(q, bias);
  probabilities.require_size(input.active.used);
  deltas.require_size(input.active.used);
  validate_usm(q, probabilities);
  validate_usm(q, deltas);
  selected_softmax(q, input.hidden.rows, input.hidden.columns,
                   input.active.offsets.data, input.active.ids.data,
                   input.hidden.storage.data, input.weights.storage.data,
                   bias.data, probabilities.data, deltas.data, labels.ids.data,
                   labels.offsets.data);
}
struct OutputGradients {
  MatrixView<float> hidden, weights;
  Span<float> bias;
};
inline void accumulate_output_gradients(sycl::queue &q, SelectedOutput input,
                                        Span<const float> deltas,
                                        OutputGradients output) {
  input.validate(q);
  deltas.require_size(input.active.used);
  validate_usm(q, deltas);
  require_matrix(output.hidden, input.hidden.rows, input.hidden.columns,
                 Layout::row_major);
  require_matrix(output.weights, input.weights.rows, input.weights.columns,
                 Layout::row_major);
  output.bias.require_size(input.weights.rows);
  validate_usm(q, output.hidden);
  validate_usm(q, output.weights);
  validate_usm(q, output.bias);
  selected_output_backward(q, input.hidden.rows, input.hidden.columns,
                           input.weights.rows, input.active.offsets.data,
                           input.active.ids.data, input.hidden.storage.data,
                           deltas.data, input.weights.storage.data,
                           output.hidden.storage.data,
                           output.weights.storage.data, output.bias.data);
}
inline void accumulate_hidden_gradients(sycl::queue &q,
                                        MatrixView<const float> x,
                                        MatrixView<const float> deltas,
                                        MatrixView<float> weights,
                                        Span<float> bias) {
  require(x.columns > 0 && deltas.columns > 0 && x.rows == deltas.rows,
          "hidden gradient shape");
  require_matrix(x, x.rows, x.columns, Layout::row_major);
  require_matrix(deltas, x.rows, deltas.columns, Layout::row_major);
  require_matrix(weights, deltas.columns, x.columns, Layout::column_major);
  bias.require_size(deltas.columns);
  validate_usm(q, x);
  validate_usm(q, deltas);
  validate_usm(q, weights);
  validate_usm(q, bias);
  hidden_parameter_gradients(q, x.rows, x.columns, deltas.columns,
                             x.storage.data, deltas.storage.data,
                             weights.storage.data, bias.data);
}
struct IndexView {
  hash_shape shape;
  Span<const int> bins, permutation, bucket_sizes, bucket_nodes;
};
struct CandidateWorkspace {
  Span<int> addresses, lengths, gathered_offsets, gathered, counts, sizes;
  Span<int> active_offsets, active_nodes;
};
inline SegmentedIdsView select_candidates(sycl::queue &q, IndexView index,
                                          MatrixView<const float> hidden,
                                          SegmentedIdsView labels,
                                          int threshold, int minimum,
                                          unsigned seed,
                                          CandidateWorkspace work) {
  const auto z = index.shape;
  const int nb = hidden.rows, no = z.nodes;
  require(z.previous > 0 && no > 0 && z.tables > 0 && z.k > 0 && z.bin > 0 &&
              z.buckets > 0 && z.capacity > 0 && labels.segments == nb &&
              threshold > 0 && minimum >= 0 && minimum <= no,
          "candidate index shape");
  require_matrix(hidden, nb, z.previous, Layout::row_major);
  validate_usm(q, hidden);
  validate_usm(q, labels);
  const int queries = checked_extent(nb, z.tables);
  checked_extent(nb, 64);
  index.bins.require_size(checked_extent(checked_extent(z.tables, z.k), z.bin));
  index.permutation.require_size(no);
  index.bucket_sizes.require_size(checked_extent(z.tables, z.buckets));
  index.bucket_nodes.require_size(
      checked_extent(checked_extent(z.tables, z.buckets), z.capacity));
  for (auto span :
       {index.bins, index.permutation, index.bucket_sizes, index.bucket_nodes})
    validate_usm(q, span);
  work.addresses.require_size(queries);
  work.lengths.require_size(queries);
  work.gathered_offsets.require_size(std::size_t(queries) + 1);
  work.gathered.require_size(checked_extent(queries, z.capacity));
  work.counts.require_size(checked_extent(nb, no));
  work.sizes.require_size(nb);
  work.active_offsets.require_size(std::size_t(nb) + 1);
  work.active_nodes.require_size(checked_extent(nb, no));
  for (auto span :
       {work.addresses, work.lengths, work.gathered_offsets, work.gathered,
        work.counts, work.sizes, work.active_offsets, work.active_nodes})
    validate_usm(q, span);
  select_candidates(
      q, z, nb, z.previous, no, int(work.counts.capacity), threshold, minimum,
      seed,
      {index.bins.data, index.permutation.data, index.bucket_sizes.data,
       index.bucket_nodes.data, hidden.storage.data, labels.ids.data,
       labels.offsets.data, work.addresses.data, work.lengths.data,
       work.gathered_offsets.data, work.gathered.data, work.counts.data,
       work.sizes.data, work.active_offsets.data, work.active_nodes.data});
  int used = 0;
  q.memcpy(&used, work.active_offsets.data + nb, sizeof(int)).wait_and_throw();
  SegmentedIdsView result{read_only(work.active_offsets),
                          read_only(work.active_nodes), nb, used};
  result.validate();
  return result;
}
} // namespace gslide::sycl_port
