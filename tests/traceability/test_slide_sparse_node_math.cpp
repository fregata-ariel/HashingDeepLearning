#include <cassert>
#include <cmath>
#include <iostream>
#include "../../ports/slide-original/SLIDE/Node.h"

// TRACE_TEST_ID: SLIDE2020-SPARSE-NODE-MATH

int main() {
    float weights[4] = {0.5f, -1.0f, 2.0f, 0.25f};
    float mom[4] = {};
    float vel[4] = {};
    train state[1] = {};
    Node node;
    node.Update(4, 0, 0, NodeType::ReLU, 1, weights, 0.75f, mom, vel, state);

    int ids[2] = {0, 2};
    float vals[2] = {2.0f, -0.5f};
    const float expected = std::max(0.0f, 0.75f + 0.5f * 2.0f + 2.0f * -0.5f);
    const float actual = node.getActivation(ids, vals, 2, 0);
    assert(std::fabs(actual - expected) < 1e-6f);
    assert(node.getInputActive(0));

    // For the first layer, a supplied upstream delta d produces d*x_j for
    // each active input and d for the bias. ADAM stores those sparse gradients
    // in _t/_tbias before Network::ProcessInput applies the optimizer update.
    const float delta = 0.4f;
    state[0]._lastDeltaforBPs = delta;
    node.backPropagateFirstLayer(ids, vals, 2, 0.01f, 0);
    assert(std::fabs(node._t[0] - delta * vals[0]) < 1e-6f);
    assert(std::fabs(node._t[2] - delta * vals[1]) < 1e-6f);
    assert(std::fabs(node._t[1]) < 1e-6f);
    assert(std::fabs(node._t[3]) < 1e-6f);
    assert(std::fabs(node._tbias - delta) < 1e-6f);
    assert(!node.getInputActive(0));

    // Node does not own these stack-backed table-index arrays.
    node._indicesInTables = nullptr;
    node._indicesInBuckets = nullptr;
    std::cout << "TRACE_TEST_PASS SLIDE2020-SPARSE-NODE-MATH\n";
}
