// Actual qk_kernel_bf16_impl is extracted before this fixture. BF16 uint16_t
// storage matches archived FBGEMM Types.h. No conversion/dispatch library runs.
// Full nnz tiles need ceil(nnz/16) accessible IDs/scores; HEAD_DIM multiples of
// 32 are supported. Padded tails/dimension truncation are safe characterizations.
// TRACE_TEST_ID: MAGICPIG-BF16-SELECTED-CPU
// TRACE_TEST_ID: MAGICPIG-BF16-BOUNDARY-CPU
#include <array>
#include <limits>
#include <stdexcept>
namespace native_bf16_test {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
float decode(bfloat16 x) {
  std::uint32_t bits = static_cast<std::uint32_t>(x) << 16;
  float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
bfloat16 encode(float x) {
  std::uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
  // Explicit input quantizer model, not actual native FBGEMM execution.
  return static_cast<bfloat16>((bits + UINT32_C(0x8000)) >> 16);
}
double decode_oracle(bfloat16 bits) {
  const int exponent = (bits >> 7) & 0xffu;
  const int mantissa = bits & 0x7fu;
  const double value = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -133)
                                   : std::ldexp(1.0 + mantissa / 128.0, exponent - 127);
  return bits & 0x8000u ? -value : value;
}
void check_case(int dim, int nnz, bool nontrivial, bool padded_tail,
                double& maximum_error, double& maximum_quantization, std::size_t& checks) {
  constexpr int rows = 37;
  const int padded = ((nnz + 15) / 16) * 16;
  std::vector<bfloat16> keys(rows * dim), query(dim);
  std::vector<float> originals(rows * dim), original_query(dim);
  for (int d = 0; d < dim; ++d) {
    original_query[d] = nontrivial ? static_cast<float>((d * 7 % 23) - 11) / 91.0f : .125f * ((d % 9) - 4);
    query[d] = encode(original_query[d]);
  }
  for (int r = 0; r < rows; ++r) for (int d = 0; d < dim; ++d) {
    originals[r * dim + d] = nontrivial ? static_cast<float>(((r * 3 + d * 7) % 47) - 23) / 113.0f
                                       : .0625f * (((r * 3 + d * 7) % 17) - 8);
    keys[r * dim + d] = encode(originals[r * dim + d]);
  }
  std::vector<int> ids(std::max(1,padded));
  for (int i = 0; i < padded; ++i) ids[i] = (i * 7 + 3) % rows;
  // Original body zeros only [0,nnz), but accumulates the full padded tile.
  constexpr float seed = .75f;
  std::vector<float> guarded(padded + 2, seed);
  guarded.front() = 321.25f; guarded.back() = -123.5f;
  qk_kernel_bf16_impl(keys.data(),ids.data(),query.data(),guarded.data()+1,dim,nnz);
  require(guarded.front() == 321.25f && guarded.back() == -123.5f, "BF16 QK guard changed");
  for (int i = 0; i < padded; ++i) {
    double expected = 0, unquantized = 0, sum_abs = 0;
    for (int d = 0; d < dim; ++d) {
      const double term = decode_oracle(query[d]) * decode_oracle(keys[ids[i]*dim+d]);
      expected += term; sum_abs += std::abs(term);
      unquantized += static_cast<double>(original_query[d]) * originals[ids[i]*dim+d];
    }
    if (i >= nnz) expected += seed;
    const double error = std::abs(guarded[i+1] - expected);
    const double bound = (dim + 2) * std::numeric_limits<float>::epsilon() * (sum_abs + seed) + 1e-6;
    require(std::isfinite(guarded[i+1]) && error <= bound, "BF16 QK quantized dense-dot mismatch");
    if (dim == 32 && nnz == 16 && !nontrivial)
      require(error <= 1e-6, "preserved bootstrap tile absolute bound");
    maximum_error = std::max(maximum_error,error);
    if (i < nnz) maximum_quantization = std::max(maximum_quantization,std::abs(unquantized-expected));
    ++checks;
  }
  require(padded_tail == (nnz != padded), "test tail classification mismatch");
}
void dimension_truncation() {
  constexpr int nnz = 16;
  for (int dim : {16,48,80}) {
    std::vector<bfloat16> keys(nnz * dim,0x3f80u), query(dim,0x3f80u);
    std::array<int,nnz> ids{}; for (int i=0;i<nnz;++i)ids[i]=i;
    std::array<float,nnz+2> guarded{}; guarded.front()=17; guarded.back()=-23;
    qk_kernel_bf16_impl(keys.data(),ids.data(),query.data(),guarded.data()+1,dim,nnz);
    const int processed=(dim/32)*32;
    for(int i=0;i<nnz;++i) {
      require(guarded[i+1]==processed,"known BF16 dimension truncation changed");
      require(guarded[i+1]!=dim,"dimension witness must differ from complete dense dot");
    }
    require(guarded.front()==17&&guarded.back()==-23,"dimension witness guard changed");
  }
}
} // namespace native_bf16_test
int main() {
  try {
    double maximum = 0, quantization = 0; std::size_t checks = 0;
    for(int dim : {32,64,128}) for(int nnz : {0,16,32}) for(bool nontrivial : {false,true})
      native_bf16_test::check_case(dim,nnz,nontrivial,false,maximum,quantization,checks);
    for(int dim : {32,64,128}) for(int nnz : {1,15,17,31}) for(bool nontrivial : {false,true})
      native_bf16_test::check_case(dim,nnz,nontrivial,true,maximum,quantization,checks);
    native_bf16_test::dimension_truncation();
    std::cout << "MAGICPIG_BF16_BODY_PASS supported_cases=18 padded_tail_cases=24 score_checks=" << checks
              << " dimension_truncation_cases=3 max_fp32_accumulator_error=" << maximum
              << " max_input_quantization_error=" << quantization
              << " native_conversion=false padded_tail_requires_accessible_ids_and_scores=true\n";
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
