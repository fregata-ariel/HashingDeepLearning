// Explicit portable substitute for the eight used AVX512 FP32 operations.
// __m512 here is sixteen float32 lanes, not an IEEE float16 data type.
// Host exp replaces the upstream polynomial only in the portable backend.
#pragma once
#include <array>
#include <cmath>
struct __m512 { std::array<float,16> lane; };
inline __m512 _mm512_set1_ps(float x) { __m512 a; a.lane.fill(x); return a; }
inline __m512 _mm512_setzero_ps() { return _mm512_set1_ps(0); }
inline __m512 _mm512_loadu_ps(const float* p) { __m512 a; for(int i=0;i<16;++i)a.lane[i]=p[i]; return a; }
inline void _mm512_storeu_ps(float* p,__m512 a) { for(int i=0;i<16;++i)p[i]=a.lane[i]; }
inline __m512 _mm512_add_ps(__m512 a,__m512 b) { for(int i=0;i<16;++i)a.lane[i]+=b.lane[i]; return a; }
inline __m512 _mm512_sub_ps(__m512 a,__m512 b) { for(int i=0;i<16;++i)a.lane[i]-=b.lane[i]; return a; }
inline __m512 _mm512_div_ps(__m512 a,__m512 b) { for(int i=0;i<16;++i)a.lane[i]/=b.lane[i]; return a; }
inline __m512 avx512_exp_ps(__m512 a) { for(float& x:a.lane)x=std::exp(x); return a; }
