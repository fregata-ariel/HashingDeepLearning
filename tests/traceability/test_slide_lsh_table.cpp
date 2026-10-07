#include <cassert>
#include <iostream>
#include "../../ports/slide-original/SLIDE/LSH.h"

// TRACE_TEST_ID: SLIDE2020-LSH-TABLE
int main() {
    LSH lsh(/*K=*/1, /*L=*/2, /*RangePow=*/3);
    int indices[2] = {2, 5};
    int* positions = lsh.add(indices, 11);
    assert(positions[0] == 0 && positions[1] == 0);
    delete[] positions;

    int** raw = lsh.retrieveRaw(indices);
    assert(raw[0] != nullptr && raw[1] != nullptr);
    assert(raw[0][0] == 11 && raw[1][0] == 11);
    assert(raw[0][1] == -1 && raw[1][1] == -1);
    delete[] raw;

    lsh.clear();
    raw = lsh.retrieveRaw(indices);
    assert(raw[0] == nullptr && raw[1] == nullptr);
    delete[] raw;
    std::cout << "TRACE_TEST_PASS SLIDE2020-LSH-TABLE\n";
}
