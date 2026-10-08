// Included after unchanged maintained CUDA bodies by check_gslide_cpu.py.
// This fixture checks arithmetic with one CPU thread per block. It does not
// exercise CUDA warp reductions, scheduling or parallel shared-memory access.
#include <string>

namespace {
struct SoftmaxFixture {
  int input_width = 4;
  int output_width = 6; // Deliberately non-square row-major matrix.
  std::vector<float> weights = {
      1.f, -.5f, .25f, 2.f,
      -.75f, 1.5f, 2.f, -.25f,
      .5f, -1.f, .75f, 1.25f,
      2.f, .25f, -.5f, 1.f,
      -1.5f, .75f, 1.f, -.5f,
      .25f, -2.f, 1.5f, .5f};
  std::vector<float> biases = {.5f, -.75f, 1.25f, -.25f, 2.f, -.5f};
  // Unsorted CSC IDs make a compressed-index/actual-node mix-up observable.
  std::vector<int> input_nodes = {3, 0, 2, 1, 3, 0};
  std::vector<float> input_values = {1.5f, -.5f, -2.f, .5f, -1.f, 2.f};
  std::vector<int> input_offsets = {0, 2, 2, 3, 6};
  std::vector<int> output_nodes = {5, 1, 3, 4, 0, 2, 3, 5, 0, 2};
  std::vector<int> output_offsets = {0, 3, 5, 6, 10};
  std::vector<int> label_nodes = {3, 5, 0, 2, 5, 2};
  std::vector<int> label_offsets = {0, 2, 3, 4, 6};
  int batches() const { return static_cast<int>(input_offsets.size()) - 1; }
};

struct SoftmaxExpected {
  std::vector<double> probabilities, deltas;
};

// Independently construct dense input vectors and full W*x+b logits in double
// precision. Normalize only the specified active nodes, using a max-subtracted
// exponential and the retained denominator epsilon (stated literally here).
// This oracle calls neither a tested kernel nor the adapter's reductions.
SoftmaxExpected dense_oracle(const SoftmaxFixture& fixture, float shift) {
  SoftmaxExpected result;
  for (int sample = 0; sample < fixture.batches(); ++sample) {
    std::vector<double> dense_input(fixture.input_width, 0.);
    for (int i = fixture.input_offsets[sample]; i < fixture.input_offsets[sample + 1]; ++i)
      dense_input.at(fixture.input_nodes[i]) += fixture.input_values[i];
    std::vector<double> logits(fixture.output_width);
    for (int output = 0; output < fixture.output_width; ++output) {
      logits[output] = static_cast<double>(fixture.biases[output]) + shift;
      for (int input = 0; input < fixture.input_width; ++input)
        logits[output] += fixture.weights[output * fixture.input_width + input] * dense_input[input];
    }
    const int begin = fixture.output_offsets[sample];
    const int end = fixture.output_offsets[sample + 1];
    const int label_begin = fixture.label_offsets[sample];
    const int label_end = fixture.label_offsets[sample + 1];
    assert(begin < end && label_begin < label_end);
    std::vector<bool> target(fixture.output_width, false);
    for (int label = label_begin; label < label_end; ++label) {
      const int node = fixture.label_nodes[label];
      assert(!target.at(node)); // Labels are unique, nonempty, and active.
      target[node] = true;
      assert(std::find(fixture.output_nodes.begin() + begin,
                       fixture.output_nodes.begin() + end, node) != fixture.output_nodes.begin() + end);
    }
    double largest = logits.at(fixture.output_nodes[begin]);
    for (int output = begin + 1; output < end; ++output)
      largest = std::max(largest, logits.at(fixture.output_nodes[output]));
    std::vector<double> exponentials;
    double denominator = 1e-8;
    for (int output = begin; output < end; ++output) {
      exponentials.push_back(std::exp(logits.at(fixture.output_nodes[output]) - largest));
      denominator += exponentials.back();
    }
    for (int output = begin; output < end; ++output) {
      const double probability = exponentials[output - begin] / denominator;
      const double mass = target[fixture.output_nodes[output]] ? 1. / (label_end - label_begin) : 0.;
      result.probabilities.push_back(probability);
      result.deltas.push_back((mass - probability) / fixture.batches());
    }
  }
  return result;
}

enum class Variant { slide_in, slide_out, all_sm };
const char* variant_name(Variant variant) {
  switch (variant) {
    case Variant::slide_in: return "slide_in";
    case Variant::slide_out: return "slide_out";
    case Variant::all_sm: return "all_sm";
  }
  std::abort();
}

void run_fixture(const SoftmaxFixture& fixture, Variant variant, float shift) {
  const SoftmaxExpected expected = dense_oracle(fixture, shift);
  OwnedCsc inputs(fixture.input_nodes, fixture.input_values, fixture.input_offsets);
  OwnedCsc labels(fixture.label_nodes, {}, fixture.label_offsets);
  OwnedCsc outputs(fixture.output_nodes, std::vector<float>(fixture.output_nodes.size(), -99.f),
                   fixture.output_offsets);
  std::vector<float> deltas(fixture.output_nodes.size(), -99.f);
  std::vector<float> shifted_biases = fixture.biases;
  for (float& bias : shifted_biases) bias += shift;
  int max_inputs = 0, max_outputs = 0, max_labels = 0;
  for (int sample = 0; sample < fixture.batches(); ++sample) {
    max_inputs = std::max(max_inputs, fixture.input_offsets[sample + 1] - fixture.input_offsets[sample]);
    max_outputs = std::max(max_outputs, fixture.output_offsets[sample + 1] - fixture.output_offsets[sample]);
    max_labels = std::max(max_labels, fixture.label_offsets[sample + 1] - fixture.label_offsets[sample]);
  }
  for (int sample = 0; sample < fixture.batches(); ++sample) {
    cpu_block(sample, fixture.batches());
    switch (variant) {
      case Variant::slide_in:
        softmax_fwd_bp_rowmajor_slide_in_knl(inputs, fixture.weights.data(), shifted_biases.data(), labels,
                                           fixture.input_width, max_outputs, max_labels, outputs, deltas.data());
        break;
      case Variant::slide_out:
        softmax_fwd_bp_rowmajor_slide_out_knl(inputs, fixture.weights.data(), shifted_biases.data(), labels,
                                            fixture.input_width, max_inputs, max_labels, outputs, deltas.data());
        break;
      case Variant::all_sm:
        softmax_fwd_bp_rowmajor_all_sm_knl(inputs, fixture.weights.data(), shifted_biases.data(), labels,
                                         fixture.input_width, max_inputs, max_outputs, max_labels, outputs, deltas.data());
        break;
    }
  }
  for (size_t i = 0; i < fixture.output_nodes.size(); ++i) {
    if (std::abs(outputs.d_vals[i] - expected.probabilities[i]) > 2e-6 ||
        std::abs(deltas[i] - expected.deltas[i]) > 2e-6)
      std::cerr << "softmax variant=" << variant_name(variant) << " shift=" << shift
                << " batches=" << fixture.batches() << " compressed_output=" << i << '\n';
    near(outputs.d_vals[i], expected.probabilities[i]);
    near(deltas[i], expected.deltas[i]);
    assert(outputs.d_nodes[i] == fixture.output_nodes[i]);
  }
  for (size_t i = 0; i < fixture.input_nodes.size(); ++i) {
    assert(inputs.d_nodes[i] == fixture.input_nodes[i]);
    near(inputs.d_vals[i], fixture.input_values[i], 0.);
  }
  for (size_t i = 0; i < fixture.label_nodes.size(); ++i)
    assert(labels.d_nodes[i] == fixture.label_nodes[i]);
  for (int sample = 0; sample <= fixture.batches(); ++sample) {
    assert(inputs.d_offsets[sample] == fixture.input_offsets[sample]);
    assert(outputs.d_offsets[sample] == fixture.output_offsets[sample]);
    assert(labels.d_offsets[sample] == fixture.label_offsets[sample]);
  }
}

SoftmaxFixture single_sample(const SoftmaxFixture& batch, int sample) {
  SoftmaxFixture single = batch;
  const int ib = batch.input_offsets[sample], ie = batch.input_offsets[sample + 1];
  const int ob = batch.output_offsets[sample], oe = batch.output_offsets[sample + 1];
  const int lb = batch.label_offsets[sample], le = batch.label_offsets[sample + 1];
  single.input_nodes.assign(batch.input_nodes.begin() + ib, batch.input_nodes.begin() + ie);
  single.input_values.assign(batch.input_values.begin() + ib, batch.input_values.begin() + ie);
  single.output_nodes.assign(batch.output_nodes.begin() + ob, batch.output_nodes.begin() + oe);
  single.label_nodes.assign(batch.label_nodes.begin() + lb, batch.label_nodes.begin() + le);
  single.input_offsets = {0, ie - ib};
  single.output_offsets = {0, oe - ob};
  single.label_offsets = {0, le - lb};
  return single;
}

void exercise_variant(Variant variant) {
  const SoftmaxFixture batch;
  // Exact binary fractions keep the large shift independent of rounding in
  // float bias addition. With the historical MAX_INIT=0, -16384 underflows.
  for (float shift : {0.f, -1000.f, -16384.f}) {
    run_fixture(batch, variant, shift);
    // Batch=1 verifies the scale separately from the batch=4 mixed-size case,
    // including an entirely empty input and a singleton candidate/label.
    for (int sample = 0; sample < batch.batches(); ++sample)
      run_fixture(single_sample(batch, sample), variant, shift);
  }
}

// TRACE_TEST_ID: GSLIDE-SOFTMAX-SLIDE-IN-CPU
void test_slide_in() { exercise_variant(Variant::slide_in); }
// TRACE_TEST_ID: GSLIDE-SOFTMAX-SLIDE-OUT-CPU
void test_slide_out() { exercise_variant(Variant::slide_out); }
} // namespace

int main() {
  test_slide_in();
  test_slide_out();
  exercise_variant(Variant::all_sm);
  std::cout << "GSLIDE_SOFTMAX_VARIANTS_CPU_PASS variants=3 fixtures=45 batch_samples=72\n";
}
