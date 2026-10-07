#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "../../ports/slide-optimized-avx512/SLIDE/Layer.h"

// TRACE_TEST_ID: OPT2021-LAYER-LAYOUT-INTEGRATION

static int weightIndex(
    WeightsOrder order, int oc, int ic, int outputCount, int inputCount) {
    return order == WeightsOrder::OI
        ? oc * inputCount + ic
        : ic * outputCount + oc;
}

static void setLogicalWeights(
    Layer<float, float>& layer,
    const float* logical, int outputCount, int inputCount,
    const float* bias) {
    for (int oc = 0; oc < outputCount; ++oc) {
        layer._bias[oc] = bias[oc];
        for (int ic = 0; ic < inputCount; ++ic) {
            layer._weights[
                weightIndex(layer._weightsOrder, oc, ic, outputCount, inputCount)
            ] = logical[oc * inputCount + ic];
        }
    }
    std::fill_n(layer._weightGrads, outputCount * inputCount, 0.0f);
    std::fill_n(layer._biasGrads, outputCount, 0.0f);
}

static void setSparsePrevious(
    Layer<float, float>& layer,
    const int* indices, const float* values, int count) {
    layer._nodeDataOpt[0].size = count;
    for (int i = 0; i < count; ++i) {
        layer._nodeDataOpt[0].indices[i] = indices[i];
        layer._nodeDataOpt[0].values[i] = values[i];
        layer._nodeDataOpt[0].grads[i] = 0.0f;
    }
}

int main() {
    constexpr int OC = 3;
    constexpr int IC = 4;
    constexpr int batch = 1;

    // Constructor sparsity selects the physical layout in the maintained port.
    Layer<float, float> oi(
        OC, IC, 0, NodeType::ReLU, batch,
        /*K=*/1, /*L=*/1, /*RangePow=*/4, /*Sparsity=*/0.5f);
    Layer<float, float> io(
        OC, IC, 0, NodeType::ReLU, batch,
        /*K=*/1, /*L=*/1, /*RangePow=*/4, /*Sparsity=*/1.0f);

    assert(oi._weightsOrder == WeightsOrder::OI);
    assert(io._weightsOrder == WeightsOrder::IO);

    const float logical[OC * IC] = {
         0.5f, -1.0f,  0.25f,  2.0f,
        -0.5f,  0.75f, 1.5f,  -0.25f,
         1.0f,  0.0f, -2.0f,   0.5f,
    };
    const float bias[OC] = {0.1f, -0.2f, 0.3f};
    setLogicalWeights(oi, logical, OC, IC, bias);
    setLogicalWeights(io, logical, OC, IC, bias);

    int activeInputs[3] = {0, 2, 3};
    float x[3] = {2.0f, -0.5f, 0.25f};

    oi.queryActiveNodeandComputeActivationsOpt(
        activeInputs, x, 3, 0, 0, nullptr, 0, 1.0f, 0);
    io.queryActiveNodeandComputeActivationsOpt(
        activeInputs, x, 3, 0, 0, nullptr, 0, 1.0f, 0);

    for (int oc = 0; oc < OC; ++oc) {
        float expected = bias[oc];
        for (int i = 0; i < 3; ++i)
            expected += logical[oc * IC + activeInputs[i]] * x[i];
        expected = std::max(0.0f, expected);
        assert(std::fabs(oi._nodeDataOpt[0].values[oc] - expected) < 1e-6f);
        assert(std::fabs(io._nodeDataOpt[0].values[oc] - expected) < 1e-6f);
    }

    // Backward uses a separate previous-layer sparse state. The current layer
    // receives the same selected outputs and upstream deltas in both layouts.
    Layer<float, float> prevOi(
        IC, IC, 0, NodeType::ReLU, batch,
        /*K=*/1, /*L=*/1, /*RangePow=*/4, /*Sparsity=*/1.0f);
    Layer<float, float> prevIo(
        IC, IC, 0, NodeType::ReLU, batch,
        /*K=*/1, /*L=*/1, /*RangePow=*/4, /*Sparsity=*/1.0f);
    setSparsePrevious(prevOi, activeInputs, x, 3);
    setSparsePrevious(prevIo, activeInputs, x, 3);

    const float gy[OC] = {0.4f, -0.5f, 0.25f};
    oi._nodeDataOpt[0].size = OC;
    io._nodeDataOpt[0].size = OC;
    for (int oc = 0; oc < OC; ++oc) {
        oi._nodeDataOpt[0].indices[oc] = oc;
        io._nodeDataOpt[0].indices[oc] = oc;
        oi._nodeDataOpt[0].grads[oc] = gy[oc];
        io._nodeDataOpt[0].grads[oc] = gy[oc];
    }

    oi.backPropagateOpt(&prevOi, 0, 0.01f);
    io.backPropagateOpt(&prevIo, 0, 0.01f);

    for (int oc = 0; oc < OC; ++oc) {
        assert(std::fabs(oi._biasGrads[oc] - gy[oc]) < 1e-6f);
        assert(std::fabs(io._biasGrads[oc] - gy[oc]) < 1e-6f);
        for (int i = 0; i < 3; ++i) {
            const int ic = activeInputs[i];
            const float expectedGw = gy[oc] * x[i];
            const int oiIndex = weightIndex(WeightsOrder::OI, oc, ic, OC, IC);
            const int ioIndex = weightIndex(WeightsOrder::IO, oc, ic, OC, IC);
            assert(std::fabs(oi._weightGrads[oiIndex] - expectedGw) < 1e-6f);
            assert(std::fabs(io._weightGrads[ioIndex] - expectedGw) < 1e-6f);
        }
    }

    for (int i = 0; i < 3; ++i) {
        float expectedGx = 0.0f;
        if (x[i] > 0.0f) {
            const int ic = activeInputs[i];
            for (int oc = 0; oc < OC; ++oc)
                expectedGx += gy[oc] * logical[oc * IC + ic];
        }
        assert(std::fabs(prevOi._nodeDataOpt[0].grads[i] - expectedGx) < 1e-6f);
        assert(std::fabs(prevIo._nodeDataOpt[0].grads[i] - expectedGx) < 1e-6f);
    }

    std::cout << "TRACE_TEST_PASS OPT2021-LAYER-LAYOUT-INTEGRATION\n";
    return 0;
}
