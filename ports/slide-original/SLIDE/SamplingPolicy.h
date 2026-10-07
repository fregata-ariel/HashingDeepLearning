#pragma once
#include <cstddef>
#include <map>

namespace slide {

std::map<int, std::size_t> collectSamplingCandidates(
    int** buckets, int tableCount, const int* labels, int labelCount,
    bool includeLabels);

void fillSamplingCandidates(
    std::map<int, std::size_t>& counts, const int* randomNodes,
    std::size_t nodeCount, std::size_t start, std::size_t targetSize);

}  // namespace slide
