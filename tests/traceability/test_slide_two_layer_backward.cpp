#include <cassert>
#include <cmath>
#include <iostream>
#include "../../ports/slide-original/SLIDE/Node.h"

// TRACE_TEST_ID: SLIDE2020-TWO-LAYER-BACKWARD

int main() {
    float prevWeights[2][1] = {{1.0f}, {-1.0f}};
    float prevMom[2][1] = {};
    float prevVel[2][1] = {};
    train prevState[2] = {};
    Node prev[2];

    int inputId[1] = {0};
    float inputVal[1] = {2.0f};

    for (int i = 0; i < 2; ++i) {
        prev[i].Update(1, i, 0, NodeType::ReLU, 1,
                       prevWeights[i], 0.0f, prevMom[i], prevVel[i], prevState);
        prev[i]._indicesInTables = nullptr;
        prev[i]._indicesInBuckets = nullptr;
        prev[i].getActivation(inputId, inputVal, 1, 0);
    }
    assert(prevState[0]._lastActivations > 0.0f);
    assert(prevState[1]._lastActivations == 0.0f);

    float outWeights[2] = {0.5f, -1.25f};
    float outMom[2] = {};
    float outVel[2] = {};
    train outState[1] = {};
    Node out;
    out.Update(2, 0, 1, NodeType::Softmax, 1,
               outWeights, 0.0f, outMom, outVel, &outState);
    out._indicesInTables = nullptr;
    out._indicesInBuckets = nullptr;

    int activePrev[2] = {0, 1};
    float activePrevValues[2] = {
        prevState[0]._lastActivations,
        prevState[1]._lastActivations
    };
    out.getActivation(activePrev, activePrevValues, 2, 0);

    const float delta = 0.4f;
    outState[0]._lastDeltaforBPs = delta;
    out.backPropagate(prev, activePrev, 2, 0.01f, 0);

    assert(std::fabs(prevState[0]._lastDeltaforBPs - delta * outWeights[0]) < 1e-6f);
    assert(std::fabs(prevState[1]._lastDeltaforBPs) < 1e-6f);
    assert(std::fabs(out._t[0] - delta * activePrevValues[0]) < 1e-6f);
    assert(std::fabs(out._t[1]) < 1e-6f);
    assert(std::fabs(out._tbias - delta) < 1e-6f);

    std::cout << "TRACE_TEST_PASS SLIDE2020-TWO-LAYER-BACKWARD\n";
    return 0;
}
