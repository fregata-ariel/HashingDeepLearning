#pragma once
#include <cmath>
#include "Config.h"

namespace slide {

/**
 * @brief Apply one Adam state/update step using the caller's bias-corrected step size.
 *
 * @param gradient Current accumulated gradient for one parameter.
 * @param stepSize Bias-corrected learning-rate factor computed by Network.
 * @param parameter Parameter value updated in place.
 * @param moment First-moment state updated in place.
 * @param velocity Second-moment state updated in place.
 *
 * A zero current gradient can still move the parameter when historical moment
 * state is non-zero. Sparse gradient accumulation therefore does not imply
 * that the subsequent Adam parameter/state traversal is sparse.
 *
 * TRACE_TEST_ID: SLIDE2020-ADAM-STATE-UPDATE.
 */
inline void applyAdamUpdate(
    float gradient, float stepSize,
    float& parameter, float& moment, float& velocity) {
    moment = BETA1 * moment + (1.0f - BETA1) * gradient;
    velocity = BETA2 * velocity + (1.0f - BETA2) * gradient * gradient;
    parameter += stepSize * moment / (std::sqrt(velocity) + EPS);
}

}  // namespace slide
