// Included after the actual selected transform_kernel and softmax_kernel.
// The driver supplies either a software 16-lane FP32 intrinsic adapter with host
// std::exp(float), or actual AVX512F/FMA intrinsics and avx512_exp_ps polynomial.
// Host QK and WV loops below are explicit substitutes: native QK/WV, BF16,
// PyTorch bindings, OpenMP, LSH sampling and the server lifecycle are not run.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace magicpig_cpu_fixture {
#if defined(MAGIC_NATIVE_AVX512) && MAGIC_NATIVE_AVX512
constexpr bool native_mode = true;
constexpr double attention_tolerance = 5e-3; // Native probabilities also allow 2% relative error.
constexpr double probability_relative_tolerance = .02;
constexpr double output_tolerance = .03;
constexpr double normalizer_tolerance = .03;
#else
constexpr bool native_mode = false;
constexpr double attention_tolerance = 1e-5; // Portable lane exp uses host std::exp.
constexpr double probability_relative_tolerance = 0.;
constexpr double output_tolerance = 1e-5;
constexpr double normalizer_tolerance = 1e-5;
#endif
constexpr double correction_tolerance = 1e-5;
constexpr float guard_value = 123456.75f;
constexpr int head_dim = 16;
constexpr int row_count = 47;

struct Metrics {
  int cases = 0;
  int candidate_checks = 0;
  double correction_error = 0.;
  double probability_error = 0.;
  double output_error = 0.;
  double normalizer_error = 0.;
  double maximum_error = 0.;
};

void require(bool condition, const std::string& description) {
  if (!condition) {
    std::cerr << "MAGICPIG_CPU_BASELINE_FAIL " << description << '\n';
    std::exit(1);
  }
}
void check(double actual, double expected, double tolerance, double& worst,
           const std::string& description) {
  require(std::isfinite(actual) && std::isfinite(expected), description + " nonfinite");
  const double error = std::abs(actual - expected);
  worst = std::max(worst, error);
  if (error > tolerance) {
    std::cerr << std::setprecision(12) << "actual=" << actual << " expected=" << expected
              << " error=" << error << " tolerance=" << tolerance << '\n';
    require(false, description);
  }
}

struct Fixture {
  std::vector<float> query, keys, values;
  Fixture() : query{.25f, -.5f, .75f, 1.f, -.25f, .5f, -.75f, 1.25f,
                    .5f, .25f, -1.f, .75f, 1.25f, -.5f, .25f, -.75f},
              keys(row_count * head_dim), values(row_count * head_dim) {
    // Pairwise rotations give an orthogonal direction without sharing any
    // normalization/correction code with the implementation under test.
    std::vector<float> orthogonal(head_dim);
    for (int d = 0; d < head_dim; d += 2) {
      orthogonal[d] = -query[d + 1];
      orthogonal[d + 1] = query[d];
    }
    for (int row = 0; row < row_count; ++row) {
      const float parallel = row % 9 == 0 ? -.25f : .25f + .125f * (row % 7);
      const float perpendicular = .5f + .125f * (row % 5);
      for (int d = 0; d < head_dim; ++d) {
        keys[row * head_dim + d] = parallel * query[d] + perpendicular * orthogonal[d];
        // Values stay within [-.5625, .5625] for the native output error bound.
        values[row * head_dim + d] = .0625f * (((row * 7 + d * 3) % 19) - 9);
      }
    }
  }
};

// Direct enumeration of the binomial events with 2..L table hits. This does
// not copy the tested cancellation formula 1-(1-p)^(L-1)*(Lp+1-p).
double probability_at_least_two(double table_hit_probability, int tables) {
  double result = 0.;
  double choose = 1.;
  for (int hits = 0; hits <= tables; ++hits) {
    if (hits >= 2)
      result += choose * std::pow(table_hit_probability, hits)
                       * std::pow(1. - table_hit_probability, tables - hits);
    if (hits < tables) choose *= static_cast<double>(tables - hits) / (hits + 1);
  }
  return result;
}

struct Reference {
  std::vector<double> corrected, probabilities, output;
  double max_log2 = 0.;
  double log2_normalizer = 0.;
};

// Dense FP64 reference: all Q/K dots and norms, correction, max-subtracted
// normalization, then dense attention-weighted V. Gather candidate IDs only
// after independently computing complete row logits; no tested body is called.
Reference dense_reference(const Fixture& fixture, const std::vector<int>& candidates,
                          int bits, int tables, double common_shift) {
  const double pi = std::acos(-1.);
  const double log2e = 1. / std::log(2.);
  double query_squared_norm = 0.;
  for (float value : fixture.query)
    query_squared_norm += static_cast<double>(value) * value;
  std::vector<double> logits(row_count);
  for (int row = 0; row < row_count; ++row) {
    double dot = 0., key_squared_norm = 0.;
    for (int d = 0; d < head_dim; ++d) {
      const double key = fixture.keys[row * head_dim + d];
      dot += static_cast<double>(fixture.query[d]) * key;
      key_squared_norm += key * key;
    }
    const double cosine = dot / std::sqrt(query_squared_norm * key_squared_norm);
    require(cosine > -1. && cosine < 1., "reference cosine outside valid interior");
    const double single_bit = 1. - std::acos(cosine) / pi;
    const double table_hit = std::pow(single_bit, bits);
    const double inclusion = probability_at_least_two(table_hit, tables);
    require(inclusion > 0. && inclusion <= 1., "invalid inclusion probability");
    // The archived implementation adds 1e-4; this explicitly characterizes
    // that variant instead of asserting the unsmoothed paper estimator.
    logits[row] = dot / std::sqrt(static_cast<double>(head_dim)) - std::log(inclusion + 1e-4);
  }
  Reference expected;
  for (int row : candidates) expected.corrected.push_back(logits[row]);
  const double maximum = *std::max_element(expected.corrected.begin(), expected.corrected.end());
  double denominator = 0.;
  for (double logit : expected.corrected) denominator += std::exp(logit - maximum);
  for (double logit : expected.corrected)
    expected.probabilities.push_back(std::exp(logit - maximum) / denominator);
  expected.max_log2 = (maximum + common_shift) * log2e;
  expected.log2_normalizer = std::log2(denominator) + expected.max_log2;
  expected.output.assign(head_dim, 0.);
  for (int d = 0; d < head_dim; ++d)
    for (size_t i = 0; i < candidates.size(); ++i)
      expected.output[d] += expected.probabilities[i] * fixture.values[candidates[i] * head_dim + d];
  return expected;
}

// Host float substitutions prepare raw dot products and norms for the actual
// correction body. They do not execute or validate MagicPIG's QK kernels.
void host_qk(const Fixture& fixture, const std::vector<int>& candidates,
             float* scores, float& query_norm, std::vector<float>& key_norms) {
  float sum = 0.f;
  for (float q : fixture.query) sum += q * q;
  query_norm = std::sqrt(sum);
  key_norms.assign(row_count, 0.f);
  for (int row = 0; row < row_count; ++row) {
    sum = 0.f;
    for (int d = 0; d < head_dim; ++d) {
      const float k = fixture.keys[row * head_dim + d];
      sum += k * k;
    }
    key_norms[row] = std::sqrt(sum);
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    float dot = 0.f;
    for (int d = 0; d < head_dim; ++d)
      dot += fixture.query[d] * fixture.keys[candidates[i] * head_dim + d];
    require(std::abs(dot / (query_norm * key_norms[candidates[i]])) < 1.f,
            "host cosine outside valid interior");
    scores[i] = dot;
  }
}

// Host float weighted-value substitution, intentionally separate from the
// dense FP64 reference. Native WV kernels and BF16 output conversion are absent.
std::vector<float> host_wv(const Fixture& fixture, const std::vector<int>& candidates,
                           const float* probabilities) {
  std::vector<float> output(head_dim, 0.f);
  for (size_t i = 0; i < candidates.size(); ++i)
    for (int d = 0; d < head_dim; ++d)
      output[d] += probabilities[i] * fixture.values[candidates[i] * head_dim + d];
  return output;
}

void run_case(const Fixture& fixture, int nnz, int bits, int tables, float shift, Metrics& metrics) {
  const std::string label = "nnz=" + std::to_string(nnz) + " K=" + std::to_string(bits)
                            + " L=" + std::to_string(tables) + " shift=" + std::to_string(shift);
  std::vector<int> candidates(nnz);
  for (int i = 0; i < nnz; ++i) candidates[i] = (i * 13 + 5) % row_count;
  const Reference expected = dense_reference(fixture, candidates, bits, tables, shift);
  const int capacity = ((nnz + 15) / 16) * 16;
  // Allocate 16-rounded score capacity for vector blocks, with guard regions
  // before/after. softmax_kernel must leave every element beyond nnz intact.
  std::vector<float> storage(capacity + 32, guard_value);
  float* score = storage.data() + 16;
  float query_norm;
  std::vector<float> key_norms;
  host_qk(fixture, candidates, score, query_norm, key_norms);
  const std::vector<float> original_norms = key_norms;
  const std::vector<int> original_candidates = candidates;
  transform_kernel(score, nnz, query_norm, key_norms.data(), bits, tables,
                   std::sqrt(static_cast<float>(head_dim)), candidates.data());
  for (int i = 0; i < nnz; ++i) {
    check(score[i], expected.corrected[i], correction_tolerance, metrics.correction_error,
          label + " correction candidate=" + std::to_string(i));
    // Shift after correction to preserve the original valid Q/K cosine norms.
    score[i] += shift;
  }
  float maximum_storage[] = {guard_value, 0.f, guard_value};
  float normalizer_storage[] = {guard_value, 0.f, guard_value};
  softmax_kernel(score, nnz, maximum_storage + 1, normalizer_storage + 1);
  require(maximum_storage[0] == guard_value && maximum_storage[2] == guard_value,
          label + " max canary");
  require(normalizer_storage[0] == guard_value && normalizer_storage[2] == guard_value,
          label + " normalizer canary");
  for (int i = 0; i < 16; ++i) require(storage[i] == guard_value, label + " score prefix canary");
  for (size_t i = nnz + 16; i < storage.size(); ++i)
    require(storage[i] == guard_value, label + " score tail/capacity canary");
  require(key_norms == original_norms && candidates == original_candidates, label + " borrowed inputs modified");
  double mass = 0.;
  for (int i = 0; i < nnz; ++i) {
    require(score[i] >= 0.f && score[i] <= 1.f, label + " probability range");
    check(score[i], expected.probabilities[i],
          attention_tolerance + probability_relative_tolerance * expected.probabilities[i],
          metrics.probability_error,
          label + " probability candidate=" + std::to_string(i));
    mass += score[i];
    ++metrics.candidate_checks;
  }
  require(std::abs(mass - 1.) < 2e-6, label + " probability mass");
  const auto output = host_wv(fixture, candidates, score);
  for (int d = 0; d < head_dim; ++d)
    check(output[d], expected.output[d], output_tolerance, metrics.output_error,
          label + " host weighted output dimension=" + std::to_string(d));
  // Base-2 metadata stores a common-shift-sized float. Allow its unavoidable
  // rounding separately from the mode-specific probability tolerances.
  const double roundoff = 4. * std::numeric_limits<float>::epsilon();
  check(maximum_storage[1], expected.max_log2,
        correction_tolerance + roundoff * (1. + std::abs(expected.max_log2)),
        metrics.maximum_error, label + " maximum log2 metadata");
  check(normalizer_storage[1], expected.log2_normalizer,
        normalizer_tolerance + roundoff * (1. + std::abs(expected.log2_normalizer)),
        metrics.normalizer_error, label + " log2 normalizer metadata");
  ++metrics.cases;
}

// TRACE_TEST_ID: MAGICPIG-BASELINE-CPU
int run() {
  // Analytic event checks also guard the independent binomial enumeration.
  require(std::abs(probability_at_least_two(.5, 2) - .25) < 1e-15, "two-table binomial oracle");
  require(std::abs(probability_at_least_two(.5, 3) - .5) < 1e-15, "three-table binomial oracle");
  require(probability_at_least_two(0., 8) == 0. && probability_at_least_two(1., 8) == 1.,
          "binomial endpoint oracle");
  const Fixture fixture;
  Metrics metrics;
  for (int nnz : {1, 2, 15, 16, 17, 31, 32})
    for (int configuration = 0; configuration < 2; ++configuration)
      for (float shift : {0.f, -32.f, -128.f})
        run_case(fixture, nnz, configuration == 0 ? 2 : 3,
                 configuration == 0 ? 5 : 8, shift, metrics);
  const double maximum_error = std::max({metrics.correction_error, metrics.probability_error,
      metrics.output_error, metrics.maximum_error, metrics.normalizer_error});
  std::cout << std::setprecision(10) << "MAGICPIG_CPU_BASELINE_PASS mode="
            << (native_mode ? "native_avx512_polynomial" : "portable_host_exp")
            << " cases=" << metrics.cases << " candidate_checks=" << metrics.candidate_checks
            << " max_error=" << maximum_error << " correction_max_error=" << metrics.correction_error
            << " probability_max_error=" << metrics.probability_error
            << " weighted_output_max_error=" << metrics.output_error
            << " maximum_metadata_error=" << metrics.maximum_error
            << " normalizer_metadata_error=" << metrics.normalizer_error
            << " probability_absolute_tolerance=" << attention_tolerance
            << " probability_relative_tolerance=" << probability_relative_tolerance
            << " output_absolute_tolerance=" << output_tolerance
            << " normalizer_absolute_tolerance=" << normalizer_tolerance
            << " qk=host_substitute wv=host_substitute\n";
  return 0;
}
} // namespace magicpig_cpu_fixture
int main() { return magicpig_cpu_fixture::run(); }
