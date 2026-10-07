#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>

// TRACE_TEST_ID: OPT2021-BF16-NATIVE-DOT

static uint16_t scalarBf16Rne(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t high = bits >> 16;
    bits += 0x7FFFu + (high & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

static float bf16BitsToFloat(uint16_t bits) {
    const uint32_t raw = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

int main() {
    alignas(64) float a[32];
    alignas(64) float b[32];
    for (int i = 0; i < 32; ++i) {
        a[i] = (static_cast<float>((i * 7) % 19) - 9.0f) / 7.0f;
        b[i] = (static_cast<float>((i * 5) % 17) - 8.0f) / 9.0f;
    }

    const __m512 a0 = _mm512_load_ps(a);
    const __m512 a1 = _mm512_load_ps(a + 16);
    const __m512 b0 = _mm512_load_ps(b);
    const __m512 b1 = _mm512_load_ps(b + 16);

    const __m512bh abf = _mm512_cvtne2ps_pbh(a1, a0);
    const __m512bh bbf = _mm512_cvtne2ps_pbh(b1, b0);
    const __m512 partial = _mm512_dpbf16_ps(_mm512_setzero_ps(), abf, bbf);

    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, partial);
    float actual = 0.0f;
    for (float lane : lanes) actual += lane;

    float expected = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float ar = bf16BitsToFloat(scalarBf16Rne(a[i]));
        const float br = bf16BitsToFloat(scalarBf16Rne(b[i]));
        expected += ar * br;
    }

    assert(std::fabs(actual - expected) < 1e-4f);
    std::cout << "native_bf16_dot actual=" << actual
              << " expected=" << expected << "\n";
    std::cout << "TRACE_TEST_PASS OPT2021-BF16-NATIVE-DOT\n";
    return 0;
}
