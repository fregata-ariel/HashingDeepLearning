#include <iostream>
#include <unordered_map>
#include "LSH.h"
#include <climits>
#include "Config.h"
#include <chrono>

using namespace std;

LSH::LSH(int K, int L, int RangePow)
{
	_K = K;
	_L = L;
	_RangePow = RangePow;
	_bucket = new Bucket*[L];

//#pragma omp parallel for
	for (int i = 0; i < L; i++)
	{
		_bucket[i] = new Bucket[1 << _RangePow];
	}

	rand1 = new int[_K*_L];

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, INT_MAX);

//#pragma omp parallel for
	for (int i = 0; i < _K*_L; i++)
	{
		rand1[i] = dis(gen);
		if (rand1[i] % 2 == 0)
			rand1[i]++;
	}
}


void LSH::clear()
{
    for (int i = 0; i < _L; i++)
    {
    	delete[] _bucket[i];
    	_bucket[i] = new Bucket[1 << _RangePow];

    }
}


void LSH::count()
{
	for (int j=0; j<_L;j++) {
		int total = 0;
		for (int i = 0; i < 1 << _RangePow; i++) {
			if (_bucket[j][i].getSize()!=0) {
				cout <<_bucket[j][i].getSize() << " ";
			}
			total += _bucket[j][i].getSize();
		}
		cout << endl;
		cout <<"TABLE "<< j << "Total "<< total << endl;
	}
}


/**
 * @brief Combines K component hashes into one bucket address for each of L tables.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 2 "Locality Sensitive Hashing" and Section 2.1
 * "LSH for Estimation and Sampling", especially Algorithm 2.
 *
 * @par Implementation note
 * The paper's (K, L) construction forms one meta-hash per table from K base
 * hashes. This function is the bucket-address step of that construction. The
 * exact bit shifting / integer mixing below is specific to this snapshot and
 * depends on the selected HashFunction.
 *
 * @see https://arxiv.org/abs/1903.03129
 */
int* LSH::hashesToIndex(int * hashes)
{

	int * indices = new int[_L];
	for (int i = 0; i < _L; i++)
	{
		unsigned int index = 0;

		for (int j = 0; j < _K; j++)
		{

			if (HashFunction==4){
				unsigned int h = hashes[_K*i + j];
				index += h<<(_K-1-j);
			}else if (HashFunction==1 | HashFunction==2){
                unsigned int h = hashes[_K*i + j];
                index += h<<((_K-1-j)*(int)floor(log(binsize)));

            }else {
                unsigned int h = rand1[_K*i + j];
                h *= rand1[_K * i + j];
                h ^= h >> 13;
                h ^= rand1[_K * i + j];
                index += h * hashes[_K * i + j];
            }
		}
		if (HashFunction==3) {
			index = index&((1<<_RangePow)-1);
		}
		indices[i] = index;
	}

	return indices;
}


int* LSH::add(int *indices, int id)
{
	int * secondIndices = new int[_L];
	for (int i = 0; i < _L; i++)
	{
		secondIndices[i] = _bucket[i][indices[i]].add(id);
	}

	return secondIndices;
}


int LSH::add(int tableId, int indices, int id)
{
	int secondIndices = _bucket[tableId][indices].add(id);
	return secondIndices;
}


/*
* Returns all the buckets
*/
/**
 * @brief Returns the candidate bucket for each LSH table.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 2.1 and Algorithm 2: hash the query, probe one
 * bucket in each of L tables, then combine the retrieved candidates into the
 * active-neuron sample.
 *
 * @par Implementation note
 * This routine exposes raw bucket storage; candidate union/counting and the
 * sampling policy are implemented by Layer::queryActiveNodeandComputeActivations.
 *
 * @see https://arxiv.org/abs/1903.03129
 */
int** LSH::retrieveRaw(int *indices)
{
	int ** rawResults = new int*[_L];

	for (int i = 0; i < _L; i++)
	{
		rawResults[i] = _bucket[i][indices[i]].getAll();
	}
	return rawResults;
}


int LSH::retrieve(int table, int indices, int bucket)
{
	return _bucket[table][indices].retrieve(bucket);
}

LSH::~LSH()
{
	delete [] rand1;
	 for (int i = 0; i < _L; i++)
	 {
	 	delete[] _bucket[i];
	 }
	 delete[] _bucket;
}
