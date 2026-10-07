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
*  Algorithm from the paper Densified Winner Take All (WTA) Hashing for Sparse Datasets. Beidi Chen, Anshumali Shrivastava
*/
using namespace std;
class DensifiedWtaHash
{
private:
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash, *_indices, *_pos, _permute;
public:
    DensifiedWtaHash(int numHashes, int noOfBitsToHash);
    static void updateMappedWinners(
        int* hashes, float* values, int numHashes,
        const int* mappedBins, const int* mappedPositions,
        const int* featureIndices, const float* data, int dataLen);
    static int resolveEmptyBin(
        const int* hashes, int numHashes, const int* probes, int probeCount);
    int * getHash(int* indices, float* data, int dataLen);
    int getRandDoubleHash(int binid, int count);
    int * getHashEasy(float* data, int dataLen, int topK);
    ~DensifiedWtaHash();
};