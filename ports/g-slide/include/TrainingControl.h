#pragma once
#include <functional>

// TRACE_TEST_ID: GSLIDE-SYCL-PRODUCTION-116
// Support: shared state ordering, not evidence of native runtime execution.
// The maintained production Network::train and SYCL backend use this exact
// operation ordering. Runtime-specific allocation/kernels/libraries live behind
// callbacks; this controller owns the state transition, not a test-only loop.
namespace gslide {
struct TrainingOperations {
  int hidden_layers;
  std::function<void()> transfer, output_forward, reset_hidden_deltas,
      output_backward, output_update, complete;
  std::function<void(int)> hidden_forward, hidden_backward, hidden_update;
  std::function<void(bool)> rebuild_index;
};
inline void train_step(const TrainingOperations &ops, bool rebuild,
                       bool reshuffle) {
  ops.transfer();
  for (int l = 0; l < ops.hidden_layers; ++l)
    ops.hidden_forward(l);
  ops.output_forward();
  // Each new forward may change compressed IDs/extent. Clear full owned
  // capacity immediately before any backward accumulation for this step.
  ops.reset_hidden_deltas();
  ops.output_backward();
  for (int l = ops.hidden_layers - 1; l >= 0; --l)
    ops.hidden_backward(l);
  for (int l = 0; l < ops.hidden_layers; ++l)
    ops.hidden_update(l);
  ops.output_update();
  if (rebuild || reshuffle)
    ops.rebuild_index(reshuffle);
  ops.complete();
}
} // namespace gslide
