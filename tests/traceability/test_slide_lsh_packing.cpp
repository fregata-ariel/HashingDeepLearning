#include <cassert>
#include <iostream>
#include "../../ports/slide-original/SLIDE/LSH.h"

// TRACE_TEST_ID: SLIDE2020-LSH-PACKING
int main() {
    LSH lsh(2, 1, 6);
    int a[2] = {0, 4};
    int b[2] = {1, 0};
    int* pa = lsh.hashesToIndex(a);
    int* pb = lsh.hashesToIndex(b);
    std::cout << "packed([0,4])=" << pa[0] << "\n";
    std::cout << "packed([1,0])=" << pb[0] << "\n";
    // Characterize the maintained snapshot. With binsize=8, hashesToIndex
    // shifts by floor(log(8)) == 2, so these distinct component pairs collide.
    assert(pa[0] == 4);
    assert(pb[0] == 4);
    assert(pa[0] == pb[0]);
    delete[] pa;
    delete[] pb;
    std::cout << "TRACE_TEST_PASS SLIDE2020-LSH-PACKING\n";
}
