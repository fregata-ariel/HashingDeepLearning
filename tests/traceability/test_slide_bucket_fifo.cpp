#include <cassert>
#include <iostream>
#include "../../ports/slide-original/SLIDE/Bucket.h"

// TRACE_TEST_ID: SLIDE2020-BUCKET-FIFO
int main() {
    Bucket bucket;
    for (int id = 1; id <= BUCKETSIZE; ++id)
        assert(bucket.add(id) == id - 1);
    assert(bucket.getTotalCounts() == BUCKETSIZE);
    assert(bucket.add(BUCKETSIZE + 1) == 0);
    assert(bucket.getTotalCounts() == BUCKETSIZE + 1);
    assert(bucket.retrieve(0) == BUCKETSIZE + 1);
    assert(bucket.retrieve(1) == 2);
    std::cout << "TRACE_TEST_PASS SLIDE2020-BUCKET-FIFO\n";
}
