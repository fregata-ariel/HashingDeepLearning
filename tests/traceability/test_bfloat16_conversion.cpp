#include <cassert>
#include <cstdint>
#include <iostream>
#ifndef OPT_AVX512_BF16
#define OPT_AVX512_BF16 0
#endif
#include "../../ports/slide-optimized-avx512/SLIDE/Bfloat16.h"

// TRACE_TEST_ID: OPT2021-BF16-CONVERSION
int main() {
    constexpr float value = 1.005859375f;
    const bfloat16 rounded(value);
    bfloat16 assigned;
    assigned = value;
    const std::uint16_t rne =
        bfloat16::cvt_float_to_bfloat16(value, bfloat16::Rounding::RNE);
    const std::uint16_t trunc =
        bfloat16::cvt_float_to_bfloat16(value, bfloat16::Rounding::TRUNC);
    assert(rounded.bits_ == rne);
    assert(assigned.bits_ == trunc);
    assert(rounded.bits_ != assigned.bits_);
    assert(static_cast<float>(rounded) == 1.0078125f);
    assert(static_cast<float>(assigned) == 1.0f);
    std::cout << "TRACE_TEST_PASS OPT2021-BF16-CONVERSION\n";
}
