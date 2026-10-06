#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <climits>
#include "Config.h"
#include <bitset>
#include <fstream>
#include <omp.h>

using namespace std;


Layer::Layer(size_t noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize,  int K, int L, int RangePow, float Sparsity, float* weights, float* bias, float *adamAvgMom, float *adamAvgVel) {
    _Nodes = nullptr;
    _randNode = nullptr;
    _normalizationConstants = nullptr;
    _train_array = nullptr;
    _weights = nullptr;
    _bias = nullptr;
    _adamAvgMom = nullptr;
    _adamAvgVel = nullptr;
    _hashTables = nullptr;
    _wtaHasher = nullptr;
    _MinHasher = nullptr;
    _srp = nullptr;
    _dwtaHasher = nullptr;
    _binids = nullptr;
    _ownsParameters = !LOADWEIGHT;

    _layerID = layerID;
    _noOfNodes = noOfNodes;
    _Nodes = new Node[noOfNodes];
    _type = type;
    _noOfActive = floor(_noOfNodes * Sparsity);
    _K = K;
    _L = L;
    _batchsize = batchsize;
    _RangeRow = RangePow;
    _previousLayerNumOfNodes = previousLayerNumOfNodes;

// create a list of random nodes just in case not enough nodes from hashtable for active nodes.
    _randNode = new int[_noOfNodes];
    for (size_t n = 0; n < _noOfNodes; n++) {
        _randNode[n] = n;
    }

    std::random_shuffle(_randNode, _randNode + _noOfNodes);

//TODO: Initialize Hash Tables and add the nodes. Done by Beidi
    _hashTables = new LSH(_K, _L, RangePow);

    if (HashFunction == 1) {
        _wtaHasher = new WtaHash(_K * _L, previousLayerNumOfNodes);
    } else if (HashFunction == 2) {
        _binids = new int[previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash(_K * _L, previousLayerNumOfNodes);
    } else if (HashFunction == 3) {
        _binids = new int[previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash(_K * _L, previousLayerNumOfNodes);
        _MinHasher->getMap(previousLayerNumOfNodes, _binids);
    } else if (HashFunction == 4) {
        _srp = new SparseRandomProjection(previousLayerNumOfNodes, _K * _L, Ratio);
    }

    if (LOADWEIGHT) {
        _weights = weights;
        _bias = bias;

        if (ADAM){
            _adamAvgMom = adamAvgMom;
            _adamAvgVel = adamAvgVel;
        }

    }else{
        _weights = new float[_noOfNodes * previousLayerNumOfNodes]();
        _bias = new float[_noOfNodes];
        random_device rd;
        default_random_engine dre(rd());
        normal_distribution<float> distribution(0.0, 0.01);

        generate(_weights, _weights + _noOfNodes * previousLayerNumOfNodes, [&] () { return distribution(dre); });
        generate(_bias, _bias + _noOfNodes, [&] () { return distribution(dre); });


        if (ADAM)
        {
            _adamAvgMom = new float[_noOfNodes * previousLayerNumOfNodes]();
            _adamAvgVel = new float[_noOfNodes * previousLayerNumOfNodes]();

        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    _train_array = new train[noOfNodes*batchsize];

    // create nodes for this layer
#pragma omp parallel for
    for (size_t i = 0; i < noOfNodes; i++)
    {
        _Nodes[i].Update(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i,
                _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i, _train_array);
        addtoHashTable(_Nodes[i]._weights, previousLayerNumOfNodes, _Nodes[i]._bias, i);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout<< noOfNodes<<" "<<1.0 * timeDiffInMiliseconds<<std::endl;

    if (type == NodeType::Softmax)
    {
        _normalizationConstants = new float[batchsize]();
    }
}


void Layer::updateTable()
{

    if (HashFunction == 1) {
         delete _wtaHasher;
        _wtaHasher = new WtaHash(_K * _L, _previousLayerNumOfNodes);
    } else if (HashFunction == 2) {
         delete _dwtaHasher, _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash(_K * _L, _previousLayerNumOfNodes);
    } else if (HashFunction == 3) {

         delete _MinHasher,  _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash(_K * _L, _previousLayerNumOfNodes);
        _MinHasher->getMap(_previousLayerNumOfNodes, _binids);
    } else if (HashFunction == 4) {

        _srp = new SparseRandomProjection(_previousLayerNumOfNodes, _K * _L, Ratio);

    }
}


void Layer::updateRandomNodes()
{
    std::random_shuffle(_randNode, _randNode + _noOfNodes);
}


/**
 * @brief Hashes a neuron's weight vector and inserts the neuron into this layer's LSH tables.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 3.1 "Introduction to the overall system",
 * initialization in Figure 2, and Algorithm 1 initialization steps.
 *
 * @par Implementation note
 * The selected hash family produces K*L component hashes; hashesToIndex maps
 * them to L buckets and the neuron id is inserted into each table. The stored
 * bucket positions are later used when the layer is maintained.
 *
 * @see https://arxiv.org/abs/1903.03129
 */
void Layer::addtoHashTable(float* weights, int length, float bias, int ID)
{
    //LSH logic
    int *hashes;
    if(HashFunction==1) {
        hashes = _wtaHasher->getHash(weights);
    }else if (HashFunction==2) {
        hashes = _dwtaHasher->getHashEasy(weights, length, TOPK);
    }else if (HashFunction==3) {
        hashes = _MinHasher->getHashEasy(_binids, weights, length, TOPK);
    }else if (HashFunction==4) {
        hashes = _srp->getHash(weights, length);
    }

    int * hashIndices = _hashTables->hashesToIndex(hashes);
    int * bucketIndices = _hashTables->add(hashIndices, ID+1);

    _Nodes[ID]._indicesInTables = hashIndices;
    _Nodes[ID]._indicesInBuckets = bucketIndices;

    delete [] hashes;

}


Node* Layer::getNodebyID(size_t nodeID)
{
    assert(("nodeID less than _noOfNodes" , nodeID < _noOfNodes));
    return &_Nodes[nodeID];
}


Node* Layer::getAllNodes()
{
    return _Nodes;
}

int Layer::getNodeCount()
{
    return _noOfNodes;
}

float Layer::getNomalizationConstant(int inputID)
{
    assert(("Error Call to Normalization Constant for non - softmax layer", _type == NodeType::Softmax));
    return _normalizationConstants[inputID];
}


float innerproduct(int* index1, float* value1, int len1, float* value2){
    float total = 0;
    for (int i=0; i<len1; i++){
        total+=value1[i]*value2[index1[i]];
    }
    return total;
}


float collision(int* hashes, int* table_hashes, int k, int l){
    int cp = 0;
    for (int i=0; i<l; i=i+k){
        int tmp = 0;
        for (int j=i; j< i+k;j++){
            if(hashes[j]==table_hashes[j]){
                tmp++;
            }
        }
        if (tmp==k){
            cp++;
        }
    }
    return cp*1.0/(l/k);
}


/**
 * @brief Selects active neurons with LSH and computes only their activations.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Algorithm 1 sampling/forward steps, Section 3.1
 * "Sparse Feed-Forward Pass with Hash Table Sampling", and Figure 3.
 *
 * @par Implementation note
 * For LSH modes, the input activation is hashed, one bucket per table is
 * retrieved, and candidates are aggregated into a sparse active-neuron set.
 * Activations are then evaluated only for that set; inactive neurons are
 * represented implicitly by their absence from the returned sparse arrays.
 * The function also contains non-LSH comparison modes used by experiments.
 *
 * @par Reported effect
 * The paper attributes SLIDE's system-level compute reduction to this adaptive
 * sparsity. No paper speedup number should be attributed to this function in
 * isolation.
 *
 * @see https://arxiv.org/abs/1903.03129
 */
int Layer::queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* lengths, int layerIndex, int inputID, int* label, int labelsize, float Sparsity, int iter)
{
    //LSH QueryLogic

    //Beidi. Query out all the candidate nodes
    int len;
    int in = 0;

    if(Sparsity == 1.0){
        len = _noOfNodes;
        lengths[layerIndex + 1] = len;
        activenodesperlayer[layerIndex + 1] = new int[len]; //assuming not intitialized;
        for (int i = 0; i < len; i++)
        {
            activenodesperlayer[layerIndex + 1][i] = i;
        }
    }
    else
    {
        if (Mode==1) {
            int *hashes;
            if (HashFunction == 1) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (HashFunction == 2) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (HashFunction == 3) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
            } else if (HashFunction == 4) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int **actives = _hashTables->retrieveRaw(hashIndices);

            // Get candidates from hashtable
            auto t00 = std::chrono::high_resolution_clock::now();

            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        counts[label[i]] = _L;
                    }
                }
            }

            for (int i = 0; i < _L; i++) {
                if (actives[i] == NULL) {
                    continue;
                } else {
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            auto t11 = std::chrono::high_resolution_clock::now();

            //thresholding
            auto t3 = std::chrono::high_resolution_clock::now();
            vector<int> vect;
            for (auto &&x : counts){
                if (x.second>THRESH){
                    vect.push_back(x.first);
                }
            }

            len = vect.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = vect[i];
            }
            auto t33 = std::chrono::high_resolution_clock::now();
            in = len;

            delete[] hashes;
            delete[] hashIndices;
            delete[] actives;

        }
        if (Mode==4) {
            int *hashes;
            if (HashFunction == 1) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (HashFunction == 2) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (HashFunction == 3) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
            } else if (HashFunction == 4) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int **actives = _hashTables->retrieveRaw(hashIndices);
            // we now have a sparse array of indices of active nodes

            // Get candidates from hashtable
            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax && labelsize > 0) {
                for (int i = 0; i < labelsize ;i++){
                    counts[label[i]] = _L;
                }
            }

            for (int i = 0; i < _L; i++) {
                if (actives[i] == NULL) {
                    continue;
                } else {
                    // copy sparse array into (dense) map
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            in = counts.size();
            if (counts.size()<1500){
                srand(time(NULL));
                size_t start = rand() % _noOfNodes;
                for (size_t i = start; i < _noOfNodes; i++) {
                    if (counts.size() >= 1000) {
                        break;
                    }
                    if (counts.count(_randNode[i]) == 0) {
                        counts[_randNode[i]] = 0;
                    }
                }

                if (counts.size() < 1000) {
                    for (size_t i = 0; i < _noOfNodes; i++) {
                        if (counts.size() >= 1000) {
                            break;
                        }
                        if (counts.count(_randNode[i]) == 0) {
                            counts[_randNode[i]] = 0;
                        }
                    }
                }
            }

            len = counts.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            // copy map into new array
            int i=0;
            for (auto &&x : counts) {
                activenodesperlayer[layerIndex + 1][i] = x.first;
                i++;
            }

            delete[] hashes;
            delete[] hashIndices;
            delete[] actives;

        }
        else if (Mode == 2 & _type== NodeType::Softmax) {
            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            auto t1 = std::chrono::high_resolution_clock::now();
            bitset <MAPLEN> bs;
            int tmpsize = 0;
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        activenodesperlayer[layerIndex + 1][i] = label[i];
                        bs[label[i]] = 1;
                    }
                    tmpsize = labelsize;
                }
            }

            while(tmpsize<len){
                int v = rand() % _noOfNodes;
                if(bs[v]==0) {
                    activenodesperlayer[layerIndex + 1][tmpsize] = v;
                    bs[v]=1;
                    tmpsize++;
                }
            }



            auto t2 = std::chrono::high_resolution_clock::now();
//            std::cout << "sampling "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;

        }

        else if (Mode==3 & _type== NodeType::Softmax){

            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];
            vector<pair<float, int> > sortW;
            int what = 0;

            for (size_t s = 0; s < _noOfNodes; s++) {
                float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                         lengths[layerIndex], _Nodes[s]._weights);
                tmp += _Nodes[s]._bias;
                if (find(label, label + labelsize, s) != label + labelsize) {
                    sortW.push_back(make_pair(-1000000000, s));
                    what++;
                }
                else{
                    sortW.push_back(make_pair(-tmp, s));
                }
            }

            std::sort(begin(sortW), end(sortW));

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = sortW[i].second;
                if (find (label, label+labelsize, sortW[i].second)!= label+labelsize){
                    in=1;
                }
            }
        }
    }

    //***********************************
    activeValuesperlayer[layerIndex + 1] = new float[len]; //assuming its not initialized else memory leak;
    float maxValue = 0;
    if (_type == NodeType::Softmax)
        _normalizationConstants[inputID] = 0;

    // find activation for all ACTIVE nodes in layer
    for (int i = 0; i < len; i++)
    {
        activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]].getActivation(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
        if(_type == NodeType::Softmax && activeValuesperlayer[layerIndex + 1][i] > maxValue){
            maxValue = activeValuesperlayer[layerIndex + 1][i];
        }
    }

    if(_type == NodeType::Softmax) {
        for (int i = 0; i < len; i++) {
            float realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
            activeValuesperlayer[layerIndex + 1][i] = realActivation;
            _Nodes[activenodesperlayer[layerIndex + 1][i]].SetlastActivation(inputID, realActivation);
            _normalizationConstants[inputID] += realActivation;
        }
    }

    return in;
}

void Layer::saveWeights(string file)
{
    if (_layerID==0) {
        cnpy::npz_save(file, "w_layer_0", _weights, {_noOfNodes, _Nodes[0]._dim}, "w");
        cnpy::npz_save(file, "b_layer_0", _bias, {_noOfNodes}, "a");
        cnpy::npz_save(file, "am_layer_0", _adamAvgMom, {_noOfNodes, _Nodes[0]._dim}, "a");
        cnpy::npz_save(file, "av_layer_0", _adamAvgVel, {_noOfNodes, _Nodes[0]._dim}, "a");
        cout<<"save for layer 0"<<endl;
        cout<<_weights[0]<<" "<<_weights[1]<<endl;
    }else{
        cnpy::npz_save(file, "w_layer_"+ to_string(_layerID), _weights, {_noOfNodes, _Nodes[0]._dim}, "a");
        cnpy::npz_save(file, "b_layer_"+ to_string(_layerID), _bias, {_noOfNodes}, "a");
        cnpy::npz_save(file, "am_layer_"+ to_string(_layerID), _adamAvgMom, {_noOfNodes, _Nodes[0]._dim}, "a");
        cnpy::npz_save(file, "av_layer_"+ to_string(_layerID), _adamAvgVel, {_noOfNodes, _Nodes[0]._dim}, "a");
        cout<<"save for layer "<<to_string(_layerID)<<endl;
        cout<<_weights[0]<<" "<<_weights[1]<<endl;
    }
}


Layer::~Layer()
{
    if (_type == NodeType::Softmax) {
        delete[] _normalizationConstants;
    }

    delete[] _Nodes;
    if (_ownsParameters) {
        delete[] _weights;
        delete[] _bias;
        delete[] _adamAvgMom;
        delete[] _adamAvgVel;
    }

    delete _hashTables;
    delete _wtaHasher;
    delete _dwtaHasher;
    delete _srp;
    delete _MinHasher;
    delete[] _binids;
    delete[] _randNode;
    delete[] _train_array;
}
