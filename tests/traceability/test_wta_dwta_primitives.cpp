#include <cassert>
#include <climits>
#include <iostream>

#include "../../ports/slide-original/SLIDE/WtaHash.h"
#include "../../ports/slide-original/SLIDE/DensifiedWtaHash.h"
#include "../../ports/slide-original/SLIDE/Config.h"

// TRACE_TEST_ID: SLIDE2020-WTA-DWTA-PRIMITIVES

int main() {
    float dense[6] = {1.0f, 5.0f, -1.0f, 5.0f, 4.0f, 0.0f};
    int permutation[4] = {3, 1, 4, 2};
    // Equal maxima at features 3 and 1: strict '>' keeps the first position.
    assert(WtaHash::selectWinnerPosition(dense, permutation, 4) == 0);

    WtaHash wta(/*numHashes=*/4, /*dimension=*/32);
    float wtaData[32];
    for (int i = 0; i < 32; ++i) wtaData[i] = static_cast<float>((i * 7) % 11);
    int* wtaHashes = wta.getHash(wtaData);
    for (int i = 0; i < 4; ++i)
        assert(wtaHashes[i] >= 0 && wtaHashes[i] < binsize);
    delete[] wtaHashes;

    int hashes[3] = {INT_MIN, INT_MIN, INT_MIN};
    float values[3] = {static_cast<float>(INT_MIN),
                       static_cast<float>(INT_MIN),
                       static_cast<float>(INT_MIN)};
    int mappedBins[6] = {0, 1, 0, 2, 1, 2};
    int mappedPositions[6] = {0, 0, 1, 0, 1, 1};
    float mappedData[6] = {1.0f, 2.0f, 4.0f, 3.0f, 5.0f, 3.0f};

    DensifiedWtaHash::updateMappedWinners(
        hashes, values, 3, mappedBins, mappedPositions,
        nullptr, mappedData, 6);
    assert(hashes[0] == 1);
    assert(hashes[1] == 1);
    assert(hashes[2] == 0);  // equal value at feature 5 does not replace first max

    int sparseHashes[3] = {1, INT_MIN, 6};
    int probesGood[2] = {1, 2};
    int probesFail[2] = {1, 1};
    assert(DensifiedWtaHash::resolveEmptyBin(
               sparseHashes, 3, probesGood, 2) == 6);
    assert(DensifiedWtaHash::resolveEmptyBin(
               sparseHashes, 3, probesFail, 2) == INT_MIN);

    // Production probe generator must stay inside the table range even when
    // numHashes is not a power of two.
    DensifiedWtaHash dwta40(/*numHashes=*/40, /*dimension=*/32);
    for (int bin = 0; bin < 40; ++bin)
        for (int count = 1; count <= 64; ++count) {
            const int probe = dwta40.getRandDoubleHash(bin, count);
            assert(probe >= 0 && probe < 40);
        }

    DensifiedWtaHash dwta4(/*numHashes=*/4, /*dimension=*/32);
    int* dwtaHashes = dwta4.getHashEasy(wtaData, 32, 30);
    for (int i = 0; i < 4; ++i)
        assert(dwtaHashes[i] >= 0 && dwtaHashes[i] < binsize);
    delete[] dwtaHashes;

    std::cout << "TRACE_TEST_PASS SLIDE2020-WTA-DWTA-PRIMITIVES\n";
    return 0;
}
