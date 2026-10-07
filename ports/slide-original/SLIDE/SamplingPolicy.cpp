#include "SamplingPolicy.h"
#include "Config.h"
#include <algorithm>

namespace slide {

/**
 * @brief Aggregate one-based bucket ids into zero-based candidate collision counts.
 *
 * @param buckets Borrowed L-element array. Each non-null inner pointer is
 *        borrowed Bucket storage terminated by -1 before BUCKETSIZE when not full.
 * @param labels Borrowed zero-based label ids; may be null when labelCount is 0.
 *
 * @par Traceability relation
 * Variant. This is the maintained Mode 4 candidate policy, including explicit
 * label insertion, rather than a literal restatement of the paper's sampling
 * distribution.
 * TRACE_TEST_ID: SLIDE2020-SAMPLING-POLICY.
 */
std::map<int, std::size_t> collectSamplingCandidates(
    int** buckets, int tableCount, const int* labels, int labelCount,
    bool includeLabels) {
    std::map<int, std::size_t> counts;
    if (includeLabels && labels != nullptr) {
        for (int i = 0; i < labelCount; ++i)
            counts[labels[i]] = static_cast<std::size_t>(tableCount);
    }
    for (int table = 0; table < tableCount; ++table) {
        if (buckets[table] == nullptr) continue;
        for (int j = 0; j < BUCKETSIZE; ++j) {
            const int node = buckets[table][j] - 1;
            if (node < 0) break;
            counts[node] += 1;
        }
    }
    return counts;
}

void fillSamplingCandidates(
    std::map<int, std::size_t>& counts, const int* randomNodes,
    std::size_t nodeCount, std::size_t start, std::size_t targetSize) {
    targetSize = std::min(targetSize, nodeCount);
    if (nodeCount == 0 || counts.size() >= targetSize) return;
    start %= nodeCount;
    for (std::size_t offset = 0; offset < nodeCount && counts.size() < targetSize; ++offset) {
        const std::size_t i = (start + offset) % nodeCount;
        const int node = randomNodes[i];
        if (counts.count(node) == 0) counts[node] = 0;
    }
}

}  // namespace slide
