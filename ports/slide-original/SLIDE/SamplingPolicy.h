#pragma once
#include <cstddef>
#include <map>

namespace slide {

/**
 * @brief Aggregate Mode 4 candidates from the L queried LSH buckets.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 3.1 and Algorithm 1 sampling/forward steps.
 *
 * @par Implementation note
 * Bucket entries are stored as one-based neuron ids. Softmax labels are
 * inserted with a count equal to the table count so they remain candidates.
 * This is the maintained snapshot's sampling policy, not a literal statement
 * of the paper's sampling distribution.
 *
 * @see TRACE_TEST_ID SLIDE2020-SAMPLING-POLICY
 */
std::map<int, std::size_t> collectSamplingCandidates(
    int** buckets, int tableCount, const int* labels, int labelCount,
    bool includeLabels);

/**
 * @brief Fill a small candidate set from the layer's randomized node order.
 *
 * @param counts Candidate id to collision-count map, updated in place.
 * @param randomNodes Permutation maintained by Layer::updateRandomNodes().
 * @param nodeCount Number of entries in randomNodes.
 * @param start Deterministic start offset; Layer supplies its historical
 *        time-seeded choice while tests provide a fixed value.
 * @param targetSize Desired candidate count, capped at nodeCount.
 *
 * @see TRACE_TEST_ID SLIDE2020-SAMPLING-POLICY
 */
void fillSamplingCandidates(
    std::map<int, std::size_t>& counts, const int* randomNodes,
    std::size_t nodeCount, std::size_t start, std::size_t targetSize);

}  // namespace slide
