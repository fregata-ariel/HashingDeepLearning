#include <cassert>
#include <iostream>
#include <map>
#include "../../ports/slide-original/SLIDE/SamplingPolicy.h"
#include "../../ports/slide-original/SLIDE/Config.h"

// TRACE_TEST_ID: SLIDE2020-SAMPLING-POLICY
int main() {
    int bucket0[BUCKETSIZE] = {};
    int bucket1[BUCKETSIZE] = {};
    bucket0[0] = 2;  // node 1
    bucket0[1] = 4;  // node 3
    bucket0[2] = -1;
    bucket1[0] = 2;  // node 1 again
    bucket1[1] = 5;  // node 4
    bucket1[2] = -1;
    int* buckets[2] = {bucket0, bucket1};
    int labels[1] = {7};

    auto counts = slide::collectSamplingCandidates(
        buckets, 2, labels, 1, true);
    assert(counts.size() == 4);
    assert(counts[1] == 2);
    assert(counts[3] == 1);
    assert(counts[4] == 1);
    assert(counts[7] == 2);

    int order[10] = {9,8,7,6,5,4,3,2,1,0};
    slide::fillSamplingCandidates(counts, order, 10, 8, 8);
    assert(counts.size() == 8);
    assert(counts.count(1) == 1);  // pre-existing candidate
    assert(counts.count(0) == 1);  // added from deterministic wrap-around
    assert(counts.count(9) == 1);
    assert(counts.count(8) == 1);
    assert(counts.count(6) == 1);
    assert(counts[0] == 0);        // filler carries no collision count

    std::cout << "TRACE_TEST_PASS SLIDE2020-SAMPLING-POLICY\n";
}
