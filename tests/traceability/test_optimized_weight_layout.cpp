#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// TRACE_TEST_ID: OPT2021-WEIGHT-LAYOUT-ORACLE

enum class Order { OI, IO };

static int index(Order order, int oc, int ic, int OC, int IC) {
    return order == Order::OI ? oc * IC + ic : ic * OC + oc;
}

int main() {
    constexpr int OC = 3;
    constexpr int IC = 4;
    const float dense[OC][IC] = {
        {0.5f, -1.0f, 0.25f, 2.0f},
        {-0.5f, 0.75f, 1.5f, -0.25f},
        {1.0f, 0.0f, -2.0f, 0.5f},
    };
    std::vector<float> oi(OC * IC), io(OC * IC);
    for (int oc = 0; oc < OC; ++oc)
        for (int ic = 0; ic < IC; ++ic) {
            oi[index(Order::OI, oc, ic, OC, IC)] = dense[oc][ic];
            io[index(Order::IO, oc, ic, OC, IC)] = dense[oc][ic];
        }

    int active_ic[3] = {0, 2, 3};
    float x[3] = {2.0f, -0.5f, 0.25f};
    float bias[OC] = {0.1f, -0.2f, 0.3f};
    float gy[OC] = {0.4f, -0.5f, 0.25f};

    for (int oc = 0; oc < OC; ++oc) {
        float y_oi = bias[oc], y_io = bias[oc];
        for (int i = 0; i < 3; ++i) {
            y_oi += oi[index(Order::OI, oc, active_ic[i], OC, IC)] * x[i];
            y_io += io[index(Order::IO, oc, active_ic[i], OC, IC)] * x[i];
        }
        assert(std::fabs(y_oi - y_io) < 1e-6f);
    }

    for (int oc = 0; oc < OC; ++oc)
        for (int i = 0; i < 3; ++i) {
            const int ic = active_ic[i];
            const float gw_oi = gy[oc] * x[i];
            const float gw_io = gy[oc] * x[i];
            assert(std::fabs(gw_oi - gw_io) < 1e-6f);
            assert(oi[index(Order::OI, oc, ic, OC, IC)] ==
                   io[index(Order::IO, oc, ic, OC, IC)]);
        }

    for (int i = 0; i < 3; ++i) {
        const int ic = active_ic[i];
        float gx_oi = 0.0f, gx_io = 0.0f;
        if (x[i] > 0.0f) {
            for (int oc = 0; oc < OC; ++oc) {
                gx_oi += gy[oc] * oi[index(Order::OI, oc, ic, OC, IC)];
                gx_io += gy[oc] * io[index(Order::IO, oc, ic, OC, IC)];
            }
        }
        assert(std::fabs(gx_oi - gx_io) < 1e-6f);
    }

    std::cout << "TRACE_TEST_PASS OPT2021-WEIGHT-LAYOUT-ORACLE\n";
}
