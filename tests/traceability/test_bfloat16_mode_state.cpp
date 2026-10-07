#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "../../ports/slide-optimized-avx512/SLIDE/Bfloat16.h"
#include "../../ports/slide-optimized-avx512/SLIDE/Config.h"
#include "../../ports/slide-optimized-avx512/SLIDE/Layer.h"

// TRACE_TEST_ID: OPT2021-BF16-MODE-STATE

static bfloat16 truncStore(float value) {
    bfloat16 out;
    out = value;
    return out;
}

static int oi(int oc, int ic, int inputCount) {
    return oc * inputCount + ic;
}

int main() {
    // Mode-2 master storage must round-trip all 32 bits exactly.
    for (float value : {0.25f, 1.005859375f, -2.75f, 123.456f}) {
        bfloat16 high;
        uint16_t low = 0;
        store_split_fp32(value, high, low);
        const float reconstructed = load_split_fp32(high, low);
        float_raw a, b;
        a.fraw = value;
        b.fraw = reconstructed;
        assert(a.iraw == b.iraw);
        assert(high.bits_ == static_cast<uint16_t>(a.iraw >> 16));
        assert(low == static_cast<uint16_t>(a.iraw & 0xFFFFu));
    }

    constexpr int OC = 2;
    constexpr int IC = 4;
    constexpr int batch = 1;

    Layer<bfloat16, float> mode1(
        OC, IC, 0, NodeType::ReLU, batch,
        /*K=*/1, /*L=*/1, /*RangePow=*/4, /*Sparsity=*/0.5f);
    Layer<bfloat16, bfloat16> mode2(
        OC, IC, 0, NodeType::ReLU, batch,
        /*K=*/1, /*L=*/1, /*RangePow=*/4, /*Sparsity=*/0.5f);
    assert(mode1._weightsOrder == WeightsOrder::OI);
    assert(mode2._weightsOrder == WeightsOrder::OI);

    const float logical[OC * IC] = {
        0.3333f, -0.7777f, 1.2345f, 0.1251f,
       -0.4444f,  0.6253f, 0.8752f, 1.5001f,
    };
    const float bias[OC] = {0.1011f, -0.2022f};

    for (int oc = 0; oc < OC; ++oc) {
        mode1._bias[oc] = bias[oc];
        store_split_fp32(bias[oc], mode2._bias[oc], mode2._biasLo[oc]);
        for (int ic = 0; ic < IC; ++ic) {
            const int idx = oi(oc, ic, IC);
            mode1._weights[idx] = logical[idx];
            store_split_fp32(
                logical[idx], mode2._weights[idx], mode2._weightsLo[idx]);
        }
    }

    int active[3] = {0, 2, 3};
    bfloat16 x[3] = {
        bfloat16(1.25f), bfloat16(-0.375f), bfloat16(0.625f)
    };

    mode1.queryActiveNodeandComputeActivationsOpt(
        active, x, 3, 0, 0, nullptr, 0, 1.0f, 0);
    mode2.queryActiveNodeandComputeActivationsOpt(
        active, x, 3, 0, 0, nullptr, 0, 1.0f, 0);

    for (int oc = 0; oc < OC; ++oc) {
        float expected1 = bias[oc];
        float expected2 = static_cast<float>(mode2._bias[oc]);
        for (int i = 0; i < 3; ++i) {
            const int idx = oi(oc, active[i], IC);
            expected1 += logical[idx] * static_cast<float>(x[i]);
            expected2 += static_cast<float>(mode2._weights[idx]) *
                         static_cast<float>(x[i]);
        }
        expected1 = std::max(0.0f, expected1);
        expected2 = std::max(0.0f, expected2);

        // The maintained bfloat16 assignment operator stores by truncation.
        const bfloat16 stored1 = truncStore(expected1);
        const bfloat16 stored2 = truncStore(expected2);
        assert(mode1._nodeDataOpt[0].values[oc].bits_ == stored1.bits_);
        assert(mode2._nodeDataOpt[0].values[oc].bits_ == stored2.bits_);
    }

    // Scalar Adam uses FP32 arithmetic in both modes. Mode 2 reconstructs the
    // full master value from high+low words, not from the BF16 compute value.
    const float gradient = 0.375f;
    const float step = 0.01f;

    float fp32Value = 1.234567f;
    float fp32Moment = 0.2f;
    float fp32Velocity = 0.3f;

    bfloat16 splitHigh;
    uint16_t splitLow = 0;
    store_split_fp32(fp32Value, splitHigh, splitLow);
    float splitMoment = fp32Moment;
    float splitVelocity = fp32Velocity;

    float expectedMoment =
        BETA1 * fp32Moment + (1.0f - BETA1) * gradient;
    float expectedVelocity =
        BETA2 * fp32Velocity + (1.0f - BETA2) * gradient * gradient;
    float expectedValue =
        fp32Value + step * expectedMoment /
        (std::sqrt(expectedVelocity) + EPS);

    apply_storage_adam(
        gradient, step, BETA1, BETA2, EPS,
        fp32Value, nullptr, fp32Moment, fp32Velocity);
    apply_storage_adam(
        gradient, step, BETA1, BETA2, EPS,
        splitHigh, &splitLow, splitMoment, splitVelocity);

    assert(std::fabs(fp32Moment - expectedMoment) < 1e-7f);
    assert(std::fabs(splitMoment - expectedMoment) < 1e-7f);
    assert(std::fabs(fp32Velocity - expectedVelocity) < 1e-7f);
    assert(std::fabs(splitVelocity - expectedVelocity) < 1e-7f);
    assert(std::fabs(fp32Value - expectedValue) < 1e-7f);
    assert(std::fabs(load_split_fp32(splitHigh, splitLow) - expectedValue) < 1e-7f);

    std::cout << "TRACE_TEST_PASS OPT2021-BF16-MODE-STATE\n";
    return 0;
}
