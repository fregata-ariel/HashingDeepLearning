#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <vector>
#include <string.h>
#include "MurmurHash.h"
/*
*  Algorithm from the paper The Power of Comparative Reasoning. Jay Yagnik, Dennis Strelow, David A. Ross, Ruei-sung Lin

*/
using namespace std;
/**
 * @brief Winner-Take-All hash prerequisite used by released SLIDE modes.
 *
 * The hash component is the position of the maximum value inside a permuted
 * bin, hence its domain is [0, binsize). WTA predates SLIDE and should be
 * treated as an LSH-family dependency, not as a SLIDE algorithm contribution.
 *
 * TRACE_TEST_ID: SLIDE2020-WTA-DWTA-PRIMITIVES.
 */
class WtaHash
{
private:
    int *_indices, _numhashes, _rangePow;
public:
    WtaHash(int numHashes, int noOfBitsToHash);
    static int selectWinnerPosition(const float* data, const int* featureIndices, int count);
    int * getHash(float* data);
    ~WtaHash();
};