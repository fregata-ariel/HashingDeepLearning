#include <cassert>
#include <cmath>
#include <iostream>
#include "../../ports/slide-original/SLIDE/Adam.h"

// TRACE_TEST_ID: SLIDE2020-ADAM-STATE-UPDATE

int main() {
    float parameter = 1.0f;
    float moment = 0.5f;
    float velocity = 0.25f;
    const float gradient = 0.0f;
    const float step = 0.01f;

    const float expectedMoment = BETA1 * 0.5f;
    const float expectedVelocity = BETA2 * 0.25f;
    const float expectedParameter =
        1.0f + step * expectedMoment / (std::sqrt(expectedVelocity) + EPS);

    slide::applyAdamUpdate(gradient, step, parameter, moment, velocity);

    assert(std::fabs(moment - expectedMoment) < 1e-7f);
    assert(std::fabs(velocity - expectedVelocity) < 1e-7f);
    assert(std::fabs(parameter - expectedParameter) < 1e-7f);
    assert(parameter != 1.0f);

    std::cout << "TRACE_TEST_PASS SLIDE2020-ADAM-STATE-UPDATE\n";
    return 0;
}
