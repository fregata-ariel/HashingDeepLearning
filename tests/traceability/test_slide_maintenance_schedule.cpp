#include <cassert>
#include <iostream>
#include "../../ports/slide-original/SLIDE/Maintenance.h"

// TRACE_TEST_ID: SLIDE2020-MAINTENANCE-SCHEDULE

int main() {
    constexpr std::size_t batch = 128;
    constexpr std::size_t rehash = 6400;    // 50 batches
    constexpr std::size_t rebuild = 128000; // 1000 batches

    assert(!slide::maintenanceDue(0, batch, rehash));
    assert(!slide::maintenanceDue(48, batch, rehash));
    assert(slide::maintenanceDue(49, batch, rehash));
    assert(!slide::maintenanceDue(49, batch, rebuild));
    assert(slide::maintenanceDue(999, batch, rehash));
    assert(slide::maintenanceDue(999, batch, rebuild));

    // Non-aligned periods: rebuild can be requested without the rehash period
    // ending on this batch, but the effective transition must still rehash.
    const bool requestedRehash = slide::maintenanceDue(74, batch, rehash);
    const bool requestedRebuild = slide::maintenanceDue(74, batch, 9600);
    assert(!requestedRehash);
    assert(requestedRebuild);
    const auto sparse = slide::effectiveMaintenance(
        requestedRehash, requestedRebuild, true);
    assert(sparse.rehash && sparse.rebuild);

    const auto rehashOnly = slide::effectiveMaintenance(true, false, true);
    assert(rehashOnly.rehash && !rehashOnly.rebuild);
    const auto neither = slide::effectiveMaintenance(false, false, true);
    assert(!neither.rehash && !neither.rebuild);
    const auto dense = slide::effectiveMaintenance(true, true, false);
    assert(!dense.rehash && !dense.rebuild);

    std::cout << "TRACE_TEST_PASS SLIDE2020-MAINTENANCE-SCHEDULE\n";
}
