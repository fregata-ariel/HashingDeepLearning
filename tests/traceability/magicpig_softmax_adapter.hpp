#pragma once
// Test-only additions to the software lanes; native builds use actual intrinsics.
#include <cmath>
#ifndef MAGIC_NATIVE_AVX512
inline __m512 _mm512_mul_ps(__m512 a, __m512 b) {
    for (int i=0;i<16;++i) a.lane[i]*=b.lane[i]; return a;
}
inline __m512 _mm512_max_ps(__m512 a, __m512 b) {
    for (int i=0;i<16;++i) a.lane[i]=std::max(a.lane[i],b.lane[i]); return a;
}
inline __m512 _mm512_maskz_loadu_ps(unsigned mask, const float* p) {
    __m512 a=_mm512_setzero_ps();
    for (int i=0;i<16;++i) if (mask&(1u<<i)) a.lane[i]=p[i]; return a;
}
inline void _mm512_mask_storeu_ps(float* p, unsigned mask, __m512 a) {
    for (int i=0;i<16;++i) if (mask&(1u<<i)) p[i]=a.lane[i];
}
inline float _mm512_reduce_add_ps(__m512 a) {
    float s=0; for (float x:a.lane) s+=x; return s;
}
inline float _mm512_reduce_max_ps(__m512 a) {
    float s=-INFINITY; for (float x:a.lane) s=std::max(s,x); return s;
}
#endif
// These aliases stand in for the upstream scalar headers, independently of
// whether the selected vector exponential is software or the native polynomial.
inline float expf32(float x) { return std::exp(x); }
inline float log2f32(float x) { return std::log2(x); }
