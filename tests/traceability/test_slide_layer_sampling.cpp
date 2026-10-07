#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "../../ports/slide-original/SLIDE/Layer.h"

// TRACE_TEST_ID: SLIDE2020-LAYER-SAMPLING-INTEGRATION

int main() {
    constexpr int inputDim = 32;
    constexpr int nodes = 4;
    constexpr int batch = 1;

    Layer layer(
        nodes, inputDim, 0, NodeType::ReLU, batch,
        /*K=*/2, /*L=*/2, /*RangePow=*/6, /*Sparsity=*/0.5f);

    // Replace the random initialization after hash-table construction with a
    // deterministic arithmetic fixture. Node weight pointers already refer to
    // slices of Layer::_weights; biases are copied into each Node and therefore
    // are set explicitly as well.
    for (int oc = 0; oc < nodes; ++oc) {
        for (int ic = 0; ic < inputDim; ++ic) {
            layer._weights[oc * inputDim + ic] =
                0.01f * static_cast<float>((oc + 1) * (ic + 1));
        }
        const float bias = -0.15f + 0.1f * static_cast<float>(oc);
        layer._bias[oc] = bias;
        layer.getNodebyID(oc)->_bias = bias;
    }

    int inputIds[] = {1, 5, 9};
    float inputValues[] = {2.0f, -0.5f, 1.25f};

    int* activeNodes[2] = {inputIds, nullptr};
    float* activeValues[2] = {inputValues, nullptr};
    int lengths[2] = {3, 0};

    const int retrievedBeforeFill =
        layer.queryActiveNodeandComputeActivations(
            activeNodes, activeValues, lengths,
            /*layerIndex=*/0, /*inputID=*/0,
            /*label=*/nullptr, /*labelsize=*/0,
            /*Sparsity=*/0.5f, /*iter=*/0);

    // Mode 4 fills candidate sets smaller than 1000 from the layer's random
    // node permutation. Since this fixture has only four nodes, every node must
    // be present after the helper is wired through Layer.
    assert(retrievedBeforeFill >= 0 && retrievedBeforeFill <= nodes);
    assert(lengths[1] == nodes);
    for (int oc = 0; oc < nodes; ++oc)
        assert(activeNodes[1][oc] == oc);

    for (int oc = 0; oc < nodes; ++oc) {
        float expected = layer._bias[oc];
        for (int i = 0; i < lengths[0]; ++i)
            expected += layer._weights[oc * inputDim + inputIds[i]] *
                        inputValues[i];
        expected = std::max(0.0f, expected);
        assert(std::fabs(activeValues[1][oc] - expected) < 1e-6f);
    }

    delete[] activeNodes[1];
    delete[] activeValues[1];

    std::cout << "TRACE_TEST_PASS SLIDE2020-LAYER-SAMPLING-INTEGRATION\n";
    return 0;
}
