#pragma once
#include <cstddef>

namespace slide {

struct MaintenanceDecision {
    bool rehash;
    bool rebuild;
};

/**
 * @brief Return whether a record-count maintenance interval ends on this batch.
 *
 * batchIndex is zero-based. The check is expressed in processed records so
 * Rehash and Rebuild use their own independent intervals.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 4.2 discusses reducing update overhead and an
 * increasing interval heuristic.
 *
 * @par Traceability relation
 * Variant. The maintained driver intentionally uses fixed record-count
 * intervals; this helper must not be described as the paper's increasing
 * interval heuristic.
 * TRACE_TEST_ID: SLIDE2020-MAINTENANCE-SCHEDULE.
 */
inline bool maintenanceDue(
    std::size_t batchIndex, std::size_t batchSize,
    std::size_t intervalRecords) {
    if (batchSize == 0 || intervalRecords == 0) return false;
    const std::size_t processedRecords = (batchIndex + 1) * batchSize;
    return processedRecords % intervalRecords == 0;
}

/**
 * @brief Convert requested maintenance flags to a consistent sparse-layer transition.
 *
 * Rebuilding the hash function necessarily implies rebuilding the hash-table
 * contents under that new function.
 *
 * TRACE_TEST_ID: SLIDE2020-MAINTENANCE-SCHEDULE.
 */
inline MaintenanceDecision effectiveMaintenance(
    bool requestedRehash, bool requestedRebuild, bool sparseLayer) {
    if (!sparseLayer) return {false, false};
    return {requestedRehash || requestedRebuild, requestedRebuild};
}

}  // namespace slide
