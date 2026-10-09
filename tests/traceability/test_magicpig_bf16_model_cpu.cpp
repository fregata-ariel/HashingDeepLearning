// Selected actual wv_kernel with explicitly modeled FP32 FMA/BF16 conversion.
// Independent finite rounding oracle uses arithmetic exponent/step selection;
// fixed hexadecimal exceptional/tie cases check the archived conversion rule.
// TRACE_TEST_ID: MAGICPIG-BF16-MODEL-CPU
#include <array>
#include <limits>
#include <stdexcept>
namespace bf16_model_test {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
float from_bits(std::uint32_t bits) { float x; std::memcpy(&x, &bits, sizeof(x)); return x; }
std::uint32_t bits_of(float x) { std::uint32_t bits; std::memcpy(&bits, &x, sizeof(bits)); return bits; }
double decode_oracle(bfloat16 bits) {
  const bool negative = (bits & 0x8000u) != 0;
  const int exponent = (bits >> 7) & 0xffu;
  const int mantissa = bits & 0x7fu;
  double value = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -133)
                              : std::ldexp(1.0 + mantissa / 128.0, exponent - 127);
  return negative ? -value : value;
}
bfloat16 finite_rounding_oracle(float x) {
  require(std::isfinite(x), "finite rounding oracle received exceptional input");
  if (x == 0) return std::signbit(x) ? 0x8000u : 0u;
  int exponent; std::frexp(std::abs(static_cast<double>(x)), &exponent);
  const double step = std::ldexp(1.0, std::max(exponent - 8, -133));
  const double rounded = std::floor(std::abs(static_cast<double>(x)) / step + .5) * step;
  const float result = static_cast<float>(std::signbit(x) ? -rounded : rounded);
  return static_cast<bfloat16>(bits_of(result) >> 16);
}
void bit_patterns() {
  struct Case { std::uint32_t input; bfloat16 expected; };
  const Case cases[] = {
    {0x00000000u,0x0000u},{0x80000000u,0x8000u},
    {0x3f807fffu,0x3f80u},{0x3f808000u,0x3f81u},{0x3f808001u,0x3f81u},
    {0xbf807fffu,0xbf80u},{0xbf808000u,0xbf81u},{0xbf808001u,0xbf81u},
    {0x3f818000u,0x3f82u},{0xbf818000u,0xbf82u},
    {0x00007fffu,0x0000u},{0x00008000u,0x0001u},
    {0x80008000u,0x8001u},{0x007fffffu,0x0080u},
    {0x7f7fffffu,0x7f80u},{0xff7fffffu,0xff80u},
    {0x7f800000u,0x7f80u},{0xff800000u,0xff80u},
    {0x7fc00000u,0x7fc0u},{0xffc00000u,0xffc0u},
    // Tiny NaN payload rounds to infinity; maximal negative NaN wraps to +0.
    {0x7f800001u,0x7f80u},{0xffffffffu,0x0000u}
  };
  for (const Case& c : cases) require(magicpig_encode_bf16(from_bits(c.input)) == c.expected, "historical conversion bit mismatch");
  const bfloat16 values[] = {0x0000u,0x8000u,0x0001u,0x8001u,0x0080u,0x3f80u,0xbf81u,0x7f7fu};
  for (bfloat16 x : values) {
    require(static_cast<double>(magicpig_decode_bf16(x)) == decode_oracle(x), "BF16 storage decoding mismatch");
    require(bits_of(magicpig_decode_bf16(x)) == static_cast<std::uint32_t>(x) << 16, "BF16 bit placement mismatch");
  }
}
void selected_wv(double& max_quantization, double& max_accumulator, std::size_t& coordinates) {
  constexpr int rows = 37;
  for (int dim : {16,32,128}) for (int nnz : {0,1,2,15,16,17,31,32}) {
    std::vector<bfloat16> values(rows * dim);
    std::vector<float> originals(rows * dim);
    for (int r = 0; r < rows; ++r) for (int d = 0; d < dim; ++d) {
      originals[r * dim + d] = static_cast<float>(((r * 11 + d * 7) % 53) - 26) / 113.0f;
      values[r * dim + d] = magicpig_encode_bf16(originals[r * dim + d]);
    }
    std::vector<int> ids(nnz); std::vector<float> scores(nnz);
    for (int i = 0; i < nnz; ++i) { ids[i] = (i * 7 + 3) % rows; scores[i] = static_cast<float>((i % 11) + 1) / 127.0f; }
    std::vector<bfloat16> guarded(dim + 2, 0x4badu);
    magicpig_captured_accumulators.clear();
    wv_kernel(values.data(), ids.data(), scores.data(), guarded.data() + 1, dim, nnz);
    require(guarded.front() == 0x4badu && guarded.back() == 0x4badu, "WV output guard changed");
    require(magicpig_captured_accumulators.size() == static_cast<std::size_t>(dim), "WV accumulator observation count");
    for (int d = 0; d < dim; ++d) {
      double dense_original = 0, dense_quantized = 0, sum_abs = 0;
      float fp32_expected = 0;
      for (int i = 0; i < nnz; ++i) {
        const double decoded = decode_oracle(values[ids[i] * dim + d]);
        const double term = decoded * static_cast<double>(scores[i]);
        dense_original += static_cast<double>(originals[ids[i] * dim + d]) * scores[i];
        dense_quantized += term; sum_abs += std::abs(term);
        fp32_expected = std::fma(static_cast<float>(decoded), scores[i], fp32_expected);
      }
      const float actual_accumulator = magicpig_captured_accumulators[d];
      require(bits_of(actual_accumulator) == bits_of(fp32_expected), "selected WV FP32 accumulation order mismatch");
      const double error = std::abs(actual_accumulator - dense_quantized);
      require(error <= (nnz + 1) * std::numeric_limits<float>::epsilon() * sum_abs + 1e-12, "FP32 accumulation exceeds conservative bound");
      require(guarded[d + 1] == finite_rounding_oracle(fp32_expected), "selected WV historical output rounding mismatch");
      max_quantization = std::max(max_quantization, std::abs(dense_original - dense_quantized));
      max_accumulator = std::max(max_accumulator, error); ++coordinates;
    }
  }
}
void accumulation_and_tie_witnesses() {
  constexpr int dim = 16;
  std::array<int,3> ids{0,1,2}; std::array<float,3> weights{1,1,1};
  std::array<bfloat16,3 * dim> values{}; std::array<bfloat16,dim> output{};
  for (int d = 0; d < dim; ++d) {
    values[d] = 0x4b80u; // 2^24
    values[dim + d] = 0x3f80u; // +1
    values[2 * dim + d] = 0xcb80u; // -2^24
  }
  magicpig_captured_accumulators.clear();
  wv_kernel(values.data(),ids.data(),weights.data(),output.data(),dim,3);
  for (int d = 0; d < dim; ++d) {
    require(output[d] == 0 && magicpig_captured_accumulators[d] == 0, "FP32 cancellation witness changed");
    const double fp64 = decode_oracle(values[d]) + decode_oracle(values[dim + d]) + decode_oracle(values[2 * dim + d]);
    require(fp64 == 1, "FP64 cancellation witness construction");
  }
  for (int d = 0; d < dim; ++d) { values[d] = 0x3f80u; values[dim + d] = 0x3b80u; }
  magicpig_captured_accumulators.clear();
  wv_kernel(values.data(),ids.data(),weights.data(),output.data(),dim,2);
  for (int d = 0; d < dim; ++d) require(output[d] == 0x3f81u, "historical tie-away witness changed");
}
} // namespace bf16_model_test
int main() {
  try {
    double quantization = 0, accumulation = 0; std::size_t coordinates = 0;
    bf16_model_test::bit_patterns();
    bf16_model_test::selected_wv(quantization, accumulation, coordinates);
    bf16_model_test::accumulation_and_tie_witnesses();
    std::cout << "MAGICPIG_BF16_MODEL_PASS cases=24 coordinates=" << coordinates
              << " bit_cases=22 decode_cases=8 witnesses=2 max_input_quantization_error=" << quantization
              << " max_fp32_accumulator_error=" << accumulation
              << " converter=historical_add_0x8000_shift16 native_conversion=false\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
