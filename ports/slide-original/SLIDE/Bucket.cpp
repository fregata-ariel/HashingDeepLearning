#include <iostream>
#include "Bucket.h"


Bucket::Bucket()
{
    isInit = -1;
    arr = new int[BUCKETSIZE]();
}


Bucket::~Bucket()
{
    delete[] arr;
}


int Bucket::getTotalCounts()
{
    return _counts;
}


int Bucket::getSize()
{
    return _counts;
}


/**
 * @brief Insert one stored one-based neuron id and return its bucket position.
 *
 * @par Implementation support
 * With FIFO enabled, storage is a BUCKETSIZE-sized circular array and the
 * logical insertion count continues to grow after older entries are replaced.
 * This replacement policy is implementation support for the released SLIDE
 * hash table; it is not a separate algorithmic claim of the SLIDE paper.
 *
 * @par Ownership
 * The bucket copies id into Bucket-owned storage; no pointer ownership crosses
 * this API.
 *
 * @par Traceability relation
 * Support.
 * TRACE_TEST_ID: SLIDE2020-BUCKET-FIFO.
 */
int Bucket::add(int id) {

    //FIFO
    if (FIFO) {
        isInit += 1;
        int index = _counts & (BUCKETSIZE - 1);
        _counts++;
        arr[index] = id;
        return index;
    }
    //Reservoir Sampling
    else {
        _counts++;
        if (index == BUCKETSIZE) {
            int randnum = rand() % (_counts) + 1;
            if (randnum == 2) {
                int randind = rand() % BUCKETSIZE;
                arr[randind] = id;
                return randind;
            } else {
                return -1;
            }
        } else {
            arr[index] = id;
            int returnIndex = index;
            index++;
            return returnIndex;
        }
    }
}


int Bucket::retrieve(int indice)
{
    if (indice >= BUCKETSIZE)
        return -1;
    return arr[indice];
}


/**
 * @brief Borrow the contiguous bucket storage used by LSH::retrieveRaw().
 *
 * @return nullptr for an uninitialized bucket; otherwise a borrowed pointer
 *         to Bucket-owned storage. The pointer must not be freed and becomes
 *         invalid when this Bucket is destroyed/replaced.
 */
int * Bucket::getAll()
{
    if (isInit == -1)
        return NULL;
    if(_counts<BUCKETSIZE){
        arr[_counts]=-1;
    }
    return arr;
}
