#include "DensifiedWtaHash.h"
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <climits>
#include <algorithm>
#include <map>
#include "Config.h"
using namespace std;


DensifiedWtaHash::DensifiedWtaHash(int numHashes, int noOfBitsToHash)
{

    _numhashes = numHashes;
    _rangePow = noOfBitsToHash;

    std::random_device rd;
    std::mt19937 gen(rd());

    _permute = ceil(_numhashes * binsize * 1.0 / noOfBitsToHash);

    int* n_array = new int[_rangePow];
    _indices = new int[_rangePow * _permute];
    _pos = new int[_rangePow * _permute];

    for (int i = 0; i < _rangePow; i++) {
        n_array[i] = i;
    }

    for (int p = 0; p < _permute ;p++) {
        std::shuffle(n_array, n_array + _rangePow, rd);
        for (int j = 0; j < _rangePow; j++) {
            _indices[p * _rangePow + n_array[j]] = (p * _rangePow + j) / binsize;
            _pos[p * _rangePow + n_array[j]] = (p * _rangePow + j)%binsize;
        }
    }
    delete [] n_array;

    _lognumhash = static_cast<int>(ceil(log2(numHashes)));
    std::uniform_int_distribution<> dis(1, INT_MAX);

    _randa = dis(gen);
    if (_randa % 2 == 0)
        _randa++;
    _randHash = new int[2];
    _randHash[0] = dis(gen);
    if (_randHash[0] % 2 == 0)
        _randHash[0]++;
    _randHash[1] = dis(gen);
    if (_randHash[1] % 2 == 0)
        _randHash[1]++;

}


void DensifiedWtaHash::updateMappedWinners(
    int* hashes, float* values, int numHashes,
    const int* mappedBins, const int* mappedPositions,
    const int* featureIndices, const float* data, int dataLen)
{
    for (int i = 0; i < dataLen; ++i) {
        const int feature = featureIndices ? featureIndices[i] : i;
        const int bin = mappedBins[feature];
        if (bin >= 0 && bin < numHashes && values[bin] < data[i]) {
            values[bin] = data[i];
            hashes[bin] = mappedPositions[feature];
        }
    }
}

int DensifiedWtaHash::resolveEmptyBin(
    const int* hashes, int numHashes, const int* probes, int probeCount)
{
    for (int i = 0; i < probeCount; ++i) {
        const int probe = probes[i];
        if (probe >= 0 && probe < numHashes && hashes[probe] != INT_MIN)
            return hashes[probe];
    }
    return INT_MIN;
}


int * DensifiedWtaHash::getHashEasy(float* data, int dataLen, int topk)
{
    // binsize is the number of times the range is larger than the total number of hashes we need.

    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    for (int p = 0; p < _permute; ++p) {
        updateMappedWinners(
            hashes, values, _numhashes,
            &_indices[p * _rangePow], &_pos[p * _rangePow],
            nullptr, data, dataLen);
    }

    for (int i = 0; i < _numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes);

            next = hashes[index]; // Kills GPU.
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] values;
    return hashArray;
}

int* DensifiedWtaHash::getHash(int* indices, float* data, int dataLen)
{
    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    // init hashes and values to INT_MIN to start
    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    //
    for (int p = 0; p < _permute; ++p) {
        updateMappedWinners(
            hashes, values, _numhashes,
            &_indices[p * _rangePow], &_pos[p * _rangePow],
            indices, data, dataLen);
    }

    for (int i = 0; i < _numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes);

            next = hashes[index]; // Kills GPU.
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }

    delete[] hashes;
    delete[] values;

    return hashArray;
}


int DensifiedWtaHash::getRandDoubleHash(int binid, int count) {
    unsigned int tohash = ((binid + 1) << 6) + count;
    const unsigned int raw =
        (_randHash[0] * tohash << 3) >> (32 - _lognumhash);
    return static_cast<int>(raw % static_cast<unsigned int>(_numhashes));
}


DensifiedWtaHash::~DensifiedWtaHash()
{
    delete[] _randHash;
    delete[] _indices;
    delete[] _pos;
}
