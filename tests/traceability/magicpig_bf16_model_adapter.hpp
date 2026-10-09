// Explicit software model; no FBGEMM conversion function or SIMD ISA executes.
// Historical AVX512 conversion: (raw FP32 bits + 0x8000u) >> 16. This differs
// from IEEE ties-to-even and preserves the archived exceptional-bit behavior.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
using bfloat16 = std::uint16_t;
inline std::vector<float> magicpig_captured_accumulators;
inline float magicpig_decode_bf16(bfloat16 x) {
  std::uint32_t bits = static_cast<std::uint32_t>(x) << 16;
  float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
inline bfloat16 magicpig_encode_bf16(float x) {
  std::uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
  return static_cast<bfloat16>((bits + UINT32_C(0x8000)) >> 16);
}
inline void Bfloat16ToFloat_avx512(const bfloat16* src, float* dst, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) dst[i] = magicpig_decode_bf16(src[i]);
}
inline void FloatToBfloat16_avx512(const float* src, bfloat16* dst, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    magicpig_captured_accumulators.push_back(src[i]);
    dst[i] = magicpig_encode_bf16(src[i]);
  }
}
inline __m512 _mm512_fmadd_ps(__m512 a, __m512 b, __m512 c) {
  for (int i = 0; i < 16; ++i) c.lane[i] = std::fma(a.lane[i], b.lane[i], c.lane[i]);
  return c;
}
