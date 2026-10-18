#pragma once
/**
 * CPU-only compatibility lowering; TRACE_TEST_ID: MAGICPIG-TORCH-CPU.
 * Include real Torch/FBGEMM before selective aliases to preserve their ABI and
 * avoid rewriting Torch/GCC SSE declarations. The maintained .cc/.h bodies
 * remain unchanged. Every AVX512 operation uses pinned upstream SIMDe, with
 * native instruction selection disabled even on a capable build host.
 * FBGEMM conversions remain actual wheel code, explicitly routed through its
 * safe runtime dispatcher instead of its unconditional AVX512 entrypoints.
 * Normal finite fixture coverage does not establish arbitrary BF16 ISA
 * DAZ/FTZ/special-value equivalence. See EXTENSION_CPU_CONTRACT.md.
 */
#if defined(__AVX__) || defined(__FMA__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#error SIMDe extension requires the baseline target without AVX/FMA flags
#endif
#include <torch/extension.h>
#include <immintrin.h>
#include "fbgemm/FbgemmConvert.h"
#ifndef SIMDE_NO_NATIVE
#error SIMDE_NO_NATIVE must be supplied by the emulated build
#endif
#define SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES
#define SIMDE_X86_AVX512BW_ENABLE_NATIVE_ALIASES
#define SIMDE_X86_AVX512DQ_ENABLE_NATIVE_ALIASES
#define SIMDE_X86_AVX512VL_ENABLE_NATIVE_ALIASES
#define SIMDE_X86_AVX512BF16_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#if defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_X86_AVX512BF16_NATIVE)
#error SIMDe native AVX512/BF16 dispatch must remain disabled
#endif
// Select the existing maintained BF16 branch only after all dependency headers.
#ifdef MAGICPIG_EMULATE_BF16
#define __AVX512BF16__ 1
#endif
#define FloatToBfloat16_avx512 FloatToBfloat16_simd
#define Bfloat16ToFloat_avx512 Bfloat16ToFloat_simd
