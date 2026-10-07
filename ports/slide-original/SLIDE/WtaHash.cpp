#include "WtaHash.h"
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <climits>
#include <algorithm>
#include <map>
#include "Config.h"
using namespace std;


WtaHash::WtaHash(int numHashes, int noOfBitsToHash)
{

    _numhashes = numHashes;
    _rangePow = noOfBitsToHash;

    std::random_device rd;
    std::mt19937 gen(rd());

    int permute = ceil(_numhashes*binsize*1.0/noOfBitsToHash);

    int* n_array = new int[_rangePow];
    _indices = new int[_rangePow*permute];

    for (int i = 0; i < _rangePow; i++) {
        n_array[i] = i;
    }
    for (int p=0; p<permute ;p++) {
        std::shuffle(n_array, n_array+_rangePow, rd);
        std::copy ( n_array, n_array+_rangePow, _indices+(p*_rangePow) );
    }
    delete [] n_array;
}


int WtaHash::selectWinnerPosition(
    const float* data, const int* featureIndices, int count)
{
    int winner = 0;
    float best = data[featureIndices[0]];
    for (int j = 1; j < count; ++j) {
        const float candidate = data[featureIndices[j]];
        if (candidate > best) {
            best = candidate;
            winner = j;
        }
    }
    return winner;
}


int * WtaHash::getHash(float* data)
{
    int *hashes = new int[_numhashes];
    for (int i = 0; i < _numhashes; ++i) {
        hashes[i] = selectWinnerPosition(
            data, &_indices[i * binsize], binsize);
    }
    return hashes;
}


WtaHash::~WtaHash()
{
    delete[] _indices;
}
