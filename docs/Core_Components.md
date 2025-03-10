# Core Neural Network Components

This document describes the core classes that constitute the neural network structure in SLIDE. These components are fundamental to building and operating neural networks within the SLIDE framework.

## 1. Network Class (Network.h)

### Overview
The `Network` class is the central class in SLIDE, representing a complete neural network. It manages layers, orchestrates the training process, and handles prediction. 

### Class Definition
```cpp
class Network
{
private:
    Layer** _hiddenlayers;
    float _learningRate;
    int _numberOfLayers;
    int* _sizesOfLayers;
    NodeType* _layersTypes;
    float * _Sparsity;
    int  _currentBatchSize;

public:
    Network(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr);
    Layer* getLayer(int LayerID);
    int predictClass(int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
    int ProcessInput(int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize, int iter, bool rehash, bool rebuild);
    void saveWeights(string file);
    ~Network();
    // Memory management operators for HugePages support
    void * operator new(size_t size);
    void operator delete(void * pointer);
};
```

### Methods

#### `Network(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr)`
- **Constructor**: Initializes a new `Network` object.
- **Parameters**:
    - `sizesOfLayers`: An array of integers specifying the number of nodes in each layer.
    - `layersTypes`: An array of `NodeType` enums, specifying the activation type for each layer.
    - `noOfLayers`: The total number of layers in the network.
    - `batchsize`: The batch size used for training.
    - `lr`: The learning rate for training.
    - `inputdim`: The dimensionality of the input data.
    - `K`, `L`, `RangePow`, `Sparsity`: Arrays of hyperparameters for LSH configuration in each layer.
    - `arr`: `cnpy::npz_t` object for loading pre-trained weights (optional).

#### `Layer* getLayer(int LayerID)`
- **Purpose**: Retrieves a specific layer from the network.
- **Parameters**:
    - `LayerID`: The index of the layer to retrieve.
- **Returns**: Pointer to the `Layer` object.

#### `int predictClass(int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize)`
- **Purpose**: Performs prediction for a batch of input data.
- **Parameters**:
    - `inputIndices`, `inputValues`, `length`: Sparse input data representation.
    - `labels`, `labelsize`: Ground truth labels (used for evaluation, not prediction itself).
- **Returns**: Number of correctly predicted instances in the batch.

#### `int ProcessInput(int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize, int iter, bool rehash, bool rebuild)`
- **Purpose**: Processes a batch of input data for training, including forward and backward passes.
- **Parameters**:
    - `inputIndices`, `inputValues`, `lengths`: Sparse input data representation.
    - `label`, `labelsize`: Ground truth labels for the batch.
    - `iter`: Current training iteration number.
    - `rehash`, `rebuild`: Flags to trigger LSH table rehash and rebuild operations.
- **Returns**: (Currently unclear from header - needs implementation review).

#### `void saveWeights(string file)`
- **Purpose**: Saves the network's weights to a file.
- **Parameters**:
    - `file`: Path to the file where weights will be saved (in `.npz` format).

#### `~Network()`
- **Destructor**: Frees allocated memory for the network, including layers.

#### `void * operator new(size_t size)` and `void operator delete(void * pointer)`
- **Memory Management Operators**: Overloaded `new` and `delete` operators to utilize HugePages for memory allocation, potentially improving performance by reducing TLB misses.

## 2. Layer Class (Layer.h)

### Overview
The `Layer` class represents a single layer in the neural network. It contains an array of `Node` objects (neurons), manages LSH structures for efficient sparse computations, and handles layer-specific operations.

### Class Definition
```cpp
class Layer
{
private:
    NodeType _type;
    Node* _Nodes;
    int * _randNode;
    float* _normalizationConstants;
    int _K, _L, _RangeRow, _previousLayerNumOfNodes, _batchsize;
    train* _train_array;

public:
    int _layerID, _noOfActive;
    size_t _noOfNodes;
    float* _weights;
    float* _adamAvgMom;
    float* _adamAvgVel;
    float* _bias;
    LSH *_hashTables;
    WtaHash *_wtaHasher;
    DensifiedMinhash *_MinHasher;
    SparseRandomProjection *_srp;
    DensifiedWtaHash *_dwtaHasher;
    int * _binids;
    Layer(size_t _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity, float* weights=NULL, float* bias=NULL, float *adamAvgMom=NULL, float *adamAvgVel=NULL);
    Node* getNodebyID(size_t nodeID);
    Node* getAllNodes();
    int getNodeCount();
    void addtoHashTable(float* weights, int length, float bias, int id);
    float getNomalizationConstant(int inputID);
    int queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int queryActiveNodes(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int computeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int computeSoftmax(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    void saveWeights(string file);
    void updateTable();
    void updateRandomNodes();
    ~Layer();
    // Memory management operators for HugePages support
    void * operator new(size_t size);
    void operator delete(void * pointer);
};
```

### Methods

#### `Layer(size_t _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity, float* weights=NULL, float* bias=NULL, float *adamAvgMom=NULL, float *adamAvgVel=NULL)`
- **Constructor**: Initializes a `Layer` object.
- **Parameters**:
    - `_numNodex`: Number of nodes in the layer.
    - `previousLayerNumOfNodes`: Number of nodes in the previous layer.
    - `layerID`: Index of the current layer.
    - `type`: `NodeType` enum specifying the activation type.
    - `batchsize`: Batch size for training.
    - `K`, `L`, `RangePow`, `Sparsity`: LSH hyperparameters for this layer.
    - `weights`, `bias`, `adamAvgMom`, `adamAvgVel`: Optional pre-trained weights and optimizer state.

#### `Node* getNodebyID(size_t nodeID)`
- **Purpose**: Retrieves a specific node from the layer by its ID.
- **Parameters**:
    - `nodeID`: ID of the node to retrieve.
- **Returns**: Pointer to the `Node` object.

#### `Node* getAllNodes()`
- **Purpose**: Returns a pointer to the array of all nodes in the layer.
- **Returns**: Pointer to the first `Node` object in the array.

#### `int getNodeCount()`
- **Purpose**: Returns the number of nodes in the layer.
- **Returns**: Number of nodes as an integer.

#### `void addtoHashTable(float* weights, int length, float bias, int id)`
- **Purpose**: Adds a neuron's weights and bias to the LSH hash tables for efficient retrieval during forward pass.
- **Parameters**:
    - `weights`: Weight vector of the neuron.
    - `length`: Dimensionality of the weight vector.
    - `bias`: Bias term of the neuron.
    - `id`: Neuron ID.

#### `float getNomalizationConstant(int inputID)`
- **Purpose**: Retrieves the normalization constant for a given input ID (used in Softmax).
- **Parameters**:
    - `inputID`: ID of the input instance.
- **Returns**: Normalization constant as a float.

#### `int queryActiveNodeandComputeActivations(...)`, `int queryActiveNodes(...)`, `int computeActivations(...)`, `int computeSoftmax(...)`
- **Purpose**: These methods are central to the sparse forward pass in SLIDE. They handle querying LSH tables to identify active neurons and compute their activations. 
- **Parameters**: (Parameters are consistent across these methods)
    - `activenodesperlayer`, `activeValuesperlayer`: Data structures to store IDs and activations of active neurons.
    - `inlenght`: Length of the input.
    - `layerID`: ID of the current layer.
    - `inputID`: ID of the input instance.
    - `label`, `labelsize`: Ground truth labels.
    - `Sparsity`: Sparsity parameter for the layer.
    - `iter`: Training iteration number.
- **Returns**: (Return type and specific purpose of each method need further clarification from implementation).

#### `void saveWeights(string file)`
- **Purpose**: Saves the layer's weights and biases to a file.
- **Parameters**:
    - `file`: Path to the file for saving weights.

#### `void updateTable()`, `void updateRandomNodes()`
- **Purpose**: Methods for updating LSH hash tables and random node selections (details need further investigation).

#### `~Layer()`
- **Destructor**: Frees memory allocated for the layer, including nodes and hash tables.

#### `void * operator new(size_t size)` and `void operator delete(void * pointer)`
- **Memory Management Operators**: Overloaded operators for HugePages support, similar to the `Network` class.


## 3. Node Class (Node.h)

### Overview
The `Node` class represents a single neuron within a layer. It stores weights, biases, activations, gradients, and implements the core logic for neuron behavior in forward and backward propagation.

### Class Definition
```cpp
class Node
{
private:
    int _activeInputs;
    NodeType _type;

public:
    train* _train;
    int _currentBatchsize;
    size_t _dim, _layerNum, _IDinLayer;
    int* _indicesInTables;
    int* _indicesInBuckets;
    float* _weights;
    float* _mirrorWeights;
    float* _adamAvgMom;
    float* _adamAvgVel;
    float* _t; //for adam
    int* _update;
    float _bias =0;
    float _tbias = 0;
    float _adamAvgMombias=0;
    float _adamAvgVelbias=0;
    float _mirrorbias =0;

    Node(){};
    Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel);
    void Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, train* train_blob);
    void updateWeights(float* newWeights, float newbias);
    float getLastActivation(int inputID);
    void incrementDelta(int inputID, float incrementValue);
    float getActivation(int* indices, float* values, int length, int inputID);
    bool getInputActive(int inputID);
    bool getActiveInputs(void);
    void SetlastActivation(int inputID, float realActivation);
    void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize);
    void backPropagate(Node* previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
    void backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID);
    ~Node();

    // Memory management operators for HugePages support
    void * operator new(size_t size);
    void operator delete(void * pointer);

    // Debugging utility
    float purturbWeight(int weightid, float delta);
    float getGradient(int weightid, int inputID, float InputVal);
};
```

### Methods

#### `Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel)`
- **Constructor**: Initializes a `Node` object.
- **Parameters**:
    - `dim`: Dimensionality of the input to this node (number of incoming connections).
    - `nodeID`: ID of the node within its layer.
    - `layerID`: ID of the layer containing this node.
    - `type`: `NodeType` enum specifying the activation type.
    - `batchsize`: Batch size for training.
    - `weights`, `bias`, `adamAvgMom`, `adamAvgVel`: Optional pre-trained weights and optimizer state.

#### `void Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, train* train_blob)`
- **Purpose**: Updates the node's parameters (weights, bias, optimizer state). (Potentially redundant with constructor - needs review).

#### `void updateWeights(float* newWeights, float newbias)`
- **Purpose**: Directly updates the node's weights and bias with provided values.
- **Parameters**:
    - `newWeights`: New weight vector.
    - `newbias`: New bias value.

#### `float getLastActivation(int inputID)`
- **Purpose**: Retrieves the last computed activation value for a given input ID.
- **Parameters**:
    - `inputID`: ID of the input instance.
- **Returns**: Last activation value as a float.

#### `void incrementDelta(int inputID, float incrementValue)`
- **Purpose**: Increments the delta (error signal) for backpropagation for a given input ID.
- **Parameters**:
    - `inputID`: ID of the input instance.
    - `incrementValue`: Value to increment the delta by.

#### `float getActivation(int* indices, float* values, int length, int inputID)`
- **Purpose**: Computes the activation of the node for a given input.
- **Parameters**:
    - `indices`, `values`, `length`: Sparse representation of the input vector.
    - `inputID`: ID of the input instance.
- **Returns**: Computed activation value as a float.

#### `bool getInputActive(int inputID)`, `bool getActiveInputs(void)`
- **Purpose**: Methods related to checking if the node is considered "active" for a given input (likely based on LSH sampling). (Need more detail from implementation).

#### `void SetlastActivation(int inputID, float realActivation)`
- **Purpose**: Sets the last computed activation value for a given input ID.
- **Parameters**:
    - `inputID`: ID of the input instance.
    - `realActivation`: Activation value to set.

#### `void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)`
- **Purpose**: Computes extra statistics needed for Softmax activation (likely related to loss calculation and backpropagation).
- **Parameters**:
    - `normalizationConstant`: Normalization constant for Softmax.
    - `inputID`: ID of the input instance.
    - `label`, `labelsize`: Ground truth label.

#### `void backPropagate(...)`, `void backPropagateFirstLayer(...)`
- **Purpose**: Implement backpropagation logic for the node, updating gradients and propagating errors to previous layers.
- **Parameters**: (Parameters vary between the two methods and need detailed examination of implementation).

#### `~Node()`
- **Destructor**: Frees memory allocated for the node, including weights and optimizer state.

#### `void * operator new(size_t size)` and `void operator delete(void * pointer)`
- **Memory Management Operators**: Overloaded operators for HugePages support.

#### Debugging Utilities: `float purturbWeight(int weightid, float delta)`, `float getGradient(int weightid, int inputID, float InputVal)`
- **Purpose**: These methods are likely for debugging and testing, allowing for manual manipulation of weights and gradient inspection.


This documentation provides a detailed overview of the `Network`, `Layer`, and `Node` classes, which are the fundamental building blocks of neural networks in the SLIDE framework. Understanding these components is crucial for comprehending the system's architecture and functionality.
