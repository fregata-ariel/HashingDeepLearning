#include <cassert>
#include <cmath>
#include <iostream>
#include "../../ports/slide-original/SLIDE/Node.h"

// TRACE_TEST_ID: SLIDE2020-SOFTMAX-GRADIENT

int main() {
    constexpr int classes = 3;
    float weights[classes][2] = {{1.0f, 0.0f}, {0.5f, 1.0f}, {-0.5f, 0.25f}};
    float bias[classes] = {0.0f, 0.25f, -0.5f};
    float mom[classes][2] = {};
    float vel[classes][2] = {};
    train states[classes] = {};
    Node nodes[classes];

    int ids[2] = {0, 1};
    float x[2] = {1.0f, 2.0f};
    float logits[classes];
    float maxLogit = -1e30f;

    for (int c = 0; c < classes; ++c) {
        nodes[c].Update(2, c, 0, NodeType::Softmax, 1,
                        weights[c], bias[c], mom[c], vel[c], states);
        nodes[c]._indicesInTables = nullptr;
        nodes[c]._indicesInBuckets = nullptr;
        logits[c] = nodes[c].getActivation(ids, x, 2, 0);
        if (logits[c] > maxLogit) maxLogit = logits[c];
    }

    float exps[classes];
    float z = 0.0f;
    for (int c = 0; c < classes; ++c) {
        exps[c] = std::exp(logits[c] - maxLogit);
        z += exps[c];
        nodes[c].SetlastActivation(0, exps[c]);
    }

    int label[1] = {1};
    float probSum = 0.0f;
    for (int c = 0; c < classes; ++c) {
        nodes[c].ComputeExtaStatsForSoftMax(z, 0, label, 1);
        const float p = exps[c] / z;
        probSum += p;
        const float expectedDelta = (c == 1 ? 1.0f - p : -p);
        assert(std::fabs(states[c]._lastActivations - p) < 1e-6f);
        assert(std::fabs(states[c]._lastDeltaforBPs - expectedDelta) < 1e-6f);
    }
    assert(std::fabs(probSum - 1.0f) < 1e-6f);

    std::cout << "TRACE_TEST_PASS SLIDE2020-SOFTMAX-GRADIENT\n";
    return 0;
}
