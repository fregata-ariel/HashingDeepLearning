#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <vector>

// TRACE_TEST_ID: OPT2021-AVX-BACKWARD-ORACLE

static bool use_avx_oi_backward(int activeInputs, int inputDim) {
    return activeInputs == inputDim && activeInputs % 128 == 0;
}

int main() {
    assert(!use_avx_oi_backward(127, 127));
    assert(use_avx_oi_backward(128, 128));
    assert(!use_avx_oi_backward(129, 129));

    constexpr int n = 128;
    std::vector<float> x(n), w(n), gw(n, 0.0f), gx(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        x[i] = (i % 9 - 4) * 0.125f;
        w[i] = (i % 7 - 3) * 0.0625f;
    }
    const float gy = 0.75f;
    const __m512 vgy = _mm512_set1_ps(gy);
    const __m512 zero = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        const __m512 vx = _mm512_loadu_ps(x.data() + i);
        const __m512 vw = _mm512_loadu_ps(w.data() + i);
        __m512 vgw = _mm512_loadu_ps(gw.data() + i);
        vgw = _mm512_fmadd_ps(vgy, vx, vgw);
        _mm512_storeu_ps(gw.data() + i, vgw);
        const __mmask16 positive = _mm512_cmp_ps_mask(zero, vx, _CMP_LT_OQ);
        const __m512 vgx = _mm512_maskz_mul_ps(positive, vgy, vw);
        _mm512_storeu_ps(gx.data() + i, vgx);
    }

    for (int i = 0; i < n; ++i) {
        assert(std::fabs(gw[i] - gy * x[i]) < 1e-6f);
        const float expected_gx = x[i] > 0.0f ? gy * w[i] : 0.0f;
        assert(std::fabs(gx[i] - expected_gx) < 1e-6f);
    }
    std::cout << "TRACE_TEST_PASS OPT2021-AVX-BACKWARD-ORACLE\n";
}
