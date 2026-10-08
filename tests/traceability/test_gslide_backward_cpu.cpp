// Included after the maintained CUDA definitions by check_gslide_cpu.py.
// These are independent dense arithmetic oracles, not copies of kernel bodies.
#include <array>

namespace {
constexpr int outputs = 4;
constexpr int inputs = 3;
constexpr int samples = 2;
using Row = std::array<double, inputs>;
using Matrix = std::array<Row, outputs>;
using InputBatch = std::array<Row, samples>;
using OutputBatch = std::array<std::array<double, outputs>, samples>;

// Logical weights are indexed [output][input], independent of physical layout.
const Matrix weights = {{{1, -2, 3}, {4, 5, -6}, {-7, 8, 9}, {10, -11, 12}}};
const InputBatch activations = {{{0, -3, 2}, {0, 1.5, -.5}}};
const std::array<std::array<bool, inputs>, samples> selected_inputs =
    {{{true, true, true}, {false, true, true}}};
const std::array<OutputBatch, 2> updates = {{
    {{{-.25, 0, 0, .5}, {0, -.75, 0, .125}}},
    {{{.75, 0, 0, -.125}, {0, .25, 0, -.5}}}
}};

enum class Body { column_major, row_major, no_shared_memory, slide };

struct GuardedBuffer {
  static constexpr float canary = 12345.f;
  std::vector<float> storage;
  explicit GuardedBuffer(const std::vector<float>& values)
      : storage(values.size() + 2, canary) {
    std::copy(values.begin(), values.end(), storage.begin() + 1);
  }
  float* data() { return storage.data() + 1; }
  void check_bounds() const {
    assert(storage.front() == canary && storage.back() == canary);
  }
};
constexpr float GuardedBuffer::canary;

template<class T>
static std::vector<T> pack_matrix(const Matrix& matrix, bool column_major) {
  std::vector<T> packed;
  if (column_major) {
    for (int input = 0; input < inputs; ++input)
      for (const auto& row : matrix) packed.push_back(static_cast<T>(row[input]));
  } else {
    for (const auto& row : matrix)
      for (double value : row) packed.push_back(static_cast<T>(value));
  }
  return packed;
}

struct DenseOracle {
  Matrix gradient{};
  std::array<double, outputs> bias{{.5, -1, 1.5, -2}};
  InputBatch previous_delta{{{-8, 9, 7}, {0, -10, 11}}};

  DenseOracle() {
    int seed = 1;
    for (auto& row : gradient)
      for (double& value : row) value = seed++ / 16.;
  }

  void accumulate(const OutputBatch& delta) {
    // Dense masked outer products and matrix-vector products define the oracle.
    // No compressed offsets, IDs or physical weight-address formula are used.
    for (int sample = 0; sample < samples; ++sample) {
      for (int output = 0; output < outputs; ++output) {
        bias[output] += delta[sample][output];
        for (int input = 0; input < inputs; ++input) {
          if (selected_inputs[sample][input]) {
            gradient[output][input] +=
                delta[sample][output] * activations[sample][input];
          }
        }
      }
      for (int input = 0; input < inputs; ++input) {
        if (!selected_inputs[sample][input]) continue;
        if (activations[sample][input] <= 0) {
          previous_delta[sample][input] = 0;
          continue;
        }
        double product = 0;
        for (int output = 0; output < outputs; ++output)
          product += weights[output][input] * delta[sample][output];
        previous_delta[sample][input] += product;
      }
    }
  }
};

static void check_previous(const GuardedBuffer& actual, const DenseOracle& expected) {
  // Explicit CSC packing is a fixture contract: sample 0 IDs {2,0,1};
  // sample 1 IDs {1,2}. Checking it separately catches sample/ID confusion.
  const double packed[] = {
      expected.previous_delta[0][2], expected.previous_delta[0][0],
      expected.previous_delta[0][1], expected.previous_delta[1][1],
      expected.previous_delta[1][2]};
  for (int i = 0; i < 5; ++i) near(actual.storage[i + 1], packed[i]);
}

static void run_dense_oracle(Body body) {
  const bool column_major = body == Body::column_major;
  // The non-square 4-output, 3-input weights have independent literal physical
  // encodings. This prevents a shared transpose mistake in oracle and fixture.
  const std::vector<float> row_weights = {1, -2, 3, 4, 5, -6, -7, 8, 9, 10, -11, 12};
  const std::vector<float> column_weights = {1, 4, -7, 10, -2, 5, 8, -11, 3, -6, 9, 12};
  const auto& physical_weights = column_major ? column_weights : row_weights;
  OwnedCsc current({3, 0, 1, 3}, {}, {0, 2, 4});
  OwnedCsc previous({2, 0, 1, 1, 2}, {2, 0, -3, 1.5, -.5}, {0, 3, 5});
  DenseOracle expected;
  GuardedBuffer gradient(pack_matrix<float>(expected.gradient, column_major));
  GuardedBuffer bias({.5, -1, 1.5, -2});
  GuardedBuffer previous_delta({7, -8, 9, -10, 11});
  const std::array<std::array<float, 4>, 2> compressed_updates =
      {{{.5, -.25, -.75, .125}, {-.125, .75, .25, -.5}}};
  const int previous_nodes[] = {2, 0, 1, 1, 2};
  const float previous_values[] = {2, 0, -3, 1.5, -.5};
  const int current_nodes[] = {3, 0, 1, 3};

  for (int update = 0; update < 2; ++update) {
    const auto& delta = compressed_updates[update];
    for (int sample = 0; sample < samples; ++sample) {
      cpu_block(sample, samples);
      switch (body) {
        case Body::column_major:
          bp_knl(current, previous, physical_weights.data(), delta.data(),
                 outputs, 2, previous_delta.data(), gradient.data(), bias.data());
          break;
        case Body::row_major:
          bp_rowmajor_knl(current, previous, physical_weights.data(), delta.data(),
                         inputs, 2, previous_delta.data(), gradient.data(), bias.data());
          break;
        case Body::no_shared_memory:
          bp_rowmajor_no_sm_knl(current, previous, physical_weights.data(), delta.data(),
                               inputs, previous_delta.data(), gradient.data(), bias.data());
          break;
        case Body::slide:
          bp_rowmajor_slide_knl(current, previous, physical_weights.data(), delta.data(),
                               inputs, 3, previous_delta.data(), gradient.data(), bias.data());
          break;
      }
    }
    expected.accumulate(updates[update]);
    const auto packed_gradient = pack_matrix<double>(expected.gradient, column_major);
    for (int i = 0; i < outputs * inputs; ++i)
      near(gradient.data()[i], packed_gradient[i]);
    for (int output = 0; output < outputs; ++output)
      near(bias.data()[output], expected.bias[output]);
    check_previous(previous_delta, expected);
    gradient.check_bounds(); bias.check_bounds(); previous_delta.check_bounds();
    for (int i = 0; i < 5; ++i) {
      assert(previous.d_nodes[i] == previous_nodes[i]);
      near(previous.d_vals[i], previous_values[i]);
    }
    for (int i = 0; i < 4; ++i) assert(current.d_nodes[i] == current_nodes[i]);
    assert(previous.d_offsets[0] == 0 && previous.d_offsets[1] == 3 && previous.d_offsets[2] == 5);
    assert(current.d_offsets[0] == 0 && current.d_offsets[1] == 2 && current.d_offsets[2] == 4);
  }
}

// TRACE_TEST_ID: GSLIDE-BP-COLMAJOR-CPU
static void test_column_major() { run_dense_oracle(Body::column_major); }
// TRACE_TEST_ID: GSLIDE-BP-ROWMAJOR-CPU
static void test_row_major() { run_dense_oracle(Body::row_major); }
// TRACE_TEST_ID: GSLIDE-BP-NO-SM-CPU
static void test_no_shared_memory() { run_dense_oracle(Body::no_shared_memory); }
// TRACE_TEST_ID: GSLIDE-BP-SLIDE-CPU
static void test_slide() { run_dense_oracle(Body::slide); }
}  // namespace

int main() {
  test_column_major(); test_row_major(); test_no_shared_memory(); test_slide();
  std::cout << "GSLIDE_BACKWARD_CPU_ORACLES_PASS cases=4 updates=2\n";
}
