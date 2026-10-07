#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <vector>

// TRACE_TEST_ID: OPT2021-AVX-FORWARD-ORACLE

static float scalar_dot_bias(const float* x, const float* w, int n, float bias) {
    float out = bias;
    for (int i = 0; i < n; ++i) out += x[i] * w[i];
    return out;
}

static float avx_dot_bias(const float* x, const float* w, int n, float bias) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(w + i), sum);
    if (i < n) {
        const int lanes = n - i;
        const __mmask16 mask = static_cast<__mmask16>((1u << lanes) - 1u);
        sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, x + i),
                              _mm512_maskz_loadu_ps(mask, w + i), sum);
    }
    return bias + _mm512_reduce_add_ps(sum);
}

int main() {
    for (int n : {15, 16, 17, 127, 128, 129}) {
        std::vector<float> x(n), w(n);
        for (int i = 0; i < n; ++i) {
            x[i] = (i % 7 - 3) * 0.125f;
            w[i] = (i % 5 - 2) * 0.25f;
        }
        const float bias = 1.75f;
        const float expected = scalar_dot_bias(x.data(), w.data(), n, bias);
        const float actual = avx_dot_bias(x.data(), w.data(), n, bias);
        assert(std::fabs(expected - actual) < 1e-5f);
    }
    std::cout << "TRACE_TEST_PASS OPT2021-AVX-FORWARD-ORACLE\n";
}
