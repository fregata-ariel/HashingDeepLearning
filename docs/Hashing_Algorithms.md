# Hashing Algorithms in SLIDE

This document details the Locality Sensitive Hashing (LSH) algorithms and related components used in SLIDE for efficient sparse computations. LSH is crucial to SLIDE's performance, enabling adaptive sampling of neurons and reducing computational costs.

## 1. LSH Class (LSH.h)

### Overview
The `LSH` class is the core component for implementing Locality Sensitive Hashing in SLIDE. It manages multiple hash tables (Buckets) and provides methods for adding items (neuron IDs) and retrieving them based on hash keys.

### Class Definition
```cpp
class LSH {
private:
    Bucket ** _bucket;
    int _K;
    int _L;
    int _RangePow;
    int *rand1;

public:
    LSH(int K, int L, int RangePow);
    void clear();
    int* add(int *indices, int id);
    int add(int indices, int tableId, int id);
    int * hashesToIndex(int * hashes);
    int** retrieveRaw(int *indices);
    int retrieve(int table, int indices, int bucket);
    void count();
    ~LSH();
};
```

### Methods

#### `LSH(int K, int L, int RangePow)`
- **Constructor**: Initializes an `LSH` object.
- **Parameters**:
    - `K`: Number of hash functions per hash table.
    - `L`: Number of hash tables.
    - `RangePow`: Range parameter for hash functions (needs clarification).

#### `void clear()`
- **Purpose**: Clears all hash tables, removing all stored items.

#### `int* add(int *indices, int id)`
- **Purpose**: Adds an item (neuron ID) to the hash tables based on provided hash indices.
- **Parameters**:
    - `indices`: Array of hash indices.
    - `id`: ID of the item (neuron) to add.
- **Returns**: (Return type and purpose need clarification from implementation).

#### `int add(int indices, int tableId, int id)`
- **Purpose**: Adds an item to a specific hash table.
- **Parameters**:
    - `indices`: Hash index for the specific table.
    - `tableId`: ID of the hash table to add to.
    - `id`: ID of the item to add.
- **Returns**: (Return type and purpose need clarification from implementation).

#### `int * hashesToIndex(int * hashes)`
- **Purpose**: Converts raw hash values to table indices.
- **Parameters**:
    - `hashes`: Array of raw hash values.
- **Returns**: Array of table indices.

#### `int** retrieveRaw(int *indices)`
- **Purpose**: Retrieves raw bucket contents for given hash indices across all hash tables.
- **Parameters**:
    - `indices`: Array of hash indices.
- **Returns**: 2D array (or similar structure) containing raw bucket contents.

#### `int retrieve(int table, int indices, int bucket)`
- **Purpose**: Retrieves items from a specific bucket in a specific hash table.
- **Parameters**:
    - `table`: ID of the hash table.
    - `indices`: Hash index.
    - `bucket`: Bucket ID (within the hash table - needs clarification if bucket ID is separate from hash index).
- **Returns**: (Return type and purpose need clarification from implementation - likely returns items in the bucket).

#### `void count()`
- **Purpose**: Likely a debugging or utility method to count items in hash tables (needs clarification).

#### `~LSH()`
- **Destructor**: Frees memory allocated for hash tables and related structures.


## 2. Bucket Class (Bucket.h)

### Overview
The `Bucket` class represents a bucket within a hash table in the LSH implementation. It stores a limited number of item IDs that hash to the same bucket.

### Class Definition
```cpp
class Bucket
{
private:
    int *arr;
    int isInit = -1;
    int index = 0;
    int _counts = 0;
    
public:
    Bucket();
    int add(int id);
    int retrieve(int index);
    int * getAll();
    int getTotalCounts();
    int getSize();
    ~Bucket();
};
```

### Methods

#### `Bucket()`
- **Constructor**: Initializes a `Bucket` object.

#### `int add(int id)`
- **Purpose**: Adds an item ID to the bucket.
- **Parameters**:
    - `id`: ID of the item to add.
- **Returns**: (Return type and purpose need clarification - likely success/failure indicator).

#### `int retrieve(int index)`
- **Purpose**: Retrieves an item ID from the bucket at a specific index.
- **Parameters**:
    - `index`: Index within the bucket.
- **Returns**: Item ID at the specified index.

#### `int * getAll()`
- **Purpose**: Retrieves all item IDs stored in the bucket.
- **Returns**: Array of item IDs in the bucket.

#### `int getTotalCounts()`
- **Purpose**: Returns the total number of items currently in the bucket.
- **Returns**: Item count as an integer.

#### `int getSize()`
- **Purpose**: Returns the maximum capacity (size) of the bucket.
- **Returns**: Bucket capacity as an integer.

#### `~Bucket()`
- **Destructor**: Frees memory allocated for the bucket.


## 3. WtaHash Class (WtaHash.h)

### Overview
The `WtaHash` class implements Winner-Takes-All (WTA) hashing, an LSH algorithm used in SLIDE. WTA hashing is based on permutations and selecting the "winner" (minimum or maximum value) within permuted segments of the input vector to generate hash codes.

### Class Definition
```cpp
class WtaHash
{
private:
    int *_indices, _numhashes, _rangePow;
public:
    WtaHash(int numHashes, int noOfBitsToHash);
    int * getHash(float* data);
    ~WtaHash();
};
```

### Methods

#### `WtaHash(int numHashes, int noOfBitsToHash)`
- **Constructor**: Initializes a `WtaHash` object.
- **Parameters**:
    - `numHashes`: Number of hash codes to generate.
    - `noOfBitsToHash`: Number of bits to use for each hash code (related to range).

#### `int * getHash(float* data)`
- **Purpose**: Generates WTA hash codes for a given input data vector.
- **Parameters**:
    - `data`: Input data vector (float array).
- **Returns**: Array of integer hash codes.

#### `~WtaHash()`
- **Destructor**: Frees allocated memory.


## 4. DensifiedWtaHash Class (DensifiedWtaHash.h)

### Overview
The `DensifiedWtaHash` class implements Densified Winner-Takes-All (DWTA) hashing, an enhanced version of WTA hashing optimized for sparse datasets. DWTA hashing addresses the limitations of WTA hashing when applied to sparse data by incorporating a densification scheme.

### Class Definition
```cpp
class DensifiedWtaHash
{
private:
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash, *_indices, *_pos, _permute;
public:
    DensifiedWtaHash(int numHashes, int noOfBitsToHash);
    int * getHash(int* indices, float* data, int dataLen);
    int getRandDoubleHash(int binid, int count);
    int * getHashEasy(float* data, int dataLen, int topK);
    ~DensifiedWtaHash();
};
```

### Methods

#### `DensifiedWtaHash(int numHashes, int noOfBitsToHash)`
- **Constructor**: Initializes a `DensifiedWtaHash` object.
- **Parameters**:
    - `numHashes`: Number of hash codes to generate.
    - `noOfBitsToHash`: Number of bits per hash code.

#### `int * getHash(int* indices, float* data, int dataLen)`
- **Purpose**: Generates DWTA hash codes for sparse input data.
- **Parameters**:
    - `indices`: Array of indices of non-zero elements in sparse data.
    - `data`: Array of non-zero values.
    - `dataLen`: Length of the sparse data.
- **Returns**: Array of integer DWTA hash codes.

#### `int getRandDoubleHash(int binid, int count)`
- **Purpose**: (Purpose needs clarification - likely related to the densification scheme, generating a secondary hash).

#### `int * getHashEasy(float* data, int dataLen, int topK)`
- **Purpose**: Generates DWTA hash codes using a simplified interface (potentially for dense input or a top-k approximation).
- **Parameters**:
    - `data`: Input data (potentially dense or top-k values).
    - `dataLen`: Length of the data.
    - `topK`: Parameter related to top-k approximation (needs clarification).
- **Returns**: Array of integer DWTA hash codes.

#### `~DensifiedWtaHash()`
- **Destructor**: Frees allocated memory.


## 5. DensifiedMinhash Class (DensifiedMinhash.h)

### Overview
The `DensifiedMinhash` class implements Densified Minhash, another LSH algorithm used in SLIDE, particularly suitable for binary or binarized data. Like DWTA, Densified Minhash aims to improve upon traditional Minhash for sparse inputs.

### Class Definition
```cpp
class DensifiedMinhash
{
private:
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash;
public:
    DensifiedMinhash(int numHashes, int noOfBitsToHash);
    int * getHash(int* indices, float* data, int* binids, int dataLen);
    int getRandDoubleHash(int binid, int count);
    int * getHashEasy(int* binids, float* data, int dataLen, int topK);
    void getMap(int n, int* binid);
    ~DensifiedMinhash();
};
```

### Methods

#### `DensifiedMinhash(int numHashes, int noOfBitsToHash)`
- **Constructor**: Initializes a `DensifiedMinhash` object.
- **Parameters**:
    - `numHashes`: Number of hash codes to generate.
    - `noOfBitsToHash`: Number of bits per hash code.

#### `int * getHash(int* indices, float* data, int* binids, int dataLen)`
- **Purpose**: Generates Densified Minhash codes for sparse input.
- **Parameters**:
    - `indices`: Indices of non-zero elements.
    - `data`: Non-zero values.
    - `binids`: Bin IDs (needs clarification - likely related to densification).
    - `dataLen`: Length of the sparse data.
- **Returns**: Array of integer Densified Minhash codes.

#### `int getRandDoubleHash(int binid, int count)`
- **Purpose**: (Purpose needs clarification - likely related to densification, generating a secondary hash).

#### `int * getHashEasy(int* binids, float* data, int dataLen, int topK)`
- **Purpose**: Simplified hash generation interface (potentially for dense input or top-k).
- **Parameters**:
    - `binids`: Bin IDs.
    - `data`: Input data.
    - `dataLen`: Data length.
    - `topK`: Top-K parameter (needs clarification).
- **Returns**: Array of integer Densified Minhash codes.

#### `void getMap(int n, int* binid)`
- **Purpose**: (Purpose needs clarification - likely related to bin mapping in densification).

#### `~DensifiedMinhash()`
- **Destructor**: Frees allocated memory.


## 6. SparseRandomProjection Class (srp.h)

### Overview
The `SparseRandomProjection` class implements Sparse Random Projection (SRP), another LSH technique used in SLIDE. SRP uses sparse random matrices to project high-dimensional input vectors into lower-dimensional hash codes, preserving similarity.

### Class Definition
```cpp
class SparseRandomProjection 
{
private:
    size_t _dim;
    size_t _numhashes, _samSize;
    short ** _randBits;
    int ** _indices;
public:
    SparseRandomProjection(size_t dimention, size_t numOfHashes, int ratio);
    int * getHash(float * vector, int length);
    int * getHashSparse(int* indices, float *values, size_t length);
    ~SparseRandomProjection();
};
```

### Methods

#### `SparseRandomProjection(size_t dimention, size_t numOfHashes, int ratio)`
- **Constructor**: Initializes a `SparseRandomProjection` object.
- **Parameters**:
    - `dimention`: Input vector dimensionality.
    - `numOfHashes`: Number of hash codes to generate.
    - `ratio`: Sparsity ratio for random projection matrix (needs clarification).

#### `int * getHash(float * vector, int length)`
- **Purpose**: Generates SRP hash codes for a dense input vector.
- **Parameters**:
    - `vector`: Dense input vector (float array).
    - `length`: Length of the vector.
- **Returns**: Array of integer SRP hash codes.

#### `int * getHashSparse(int* indices, float *values, size_t length)`
- **Purpose**: Generates SRP hash codes for sparse input data.
- **Parameters**:
    - `indices`: Indices of non-zero elements.
    - `values`: Non-zero values.
    - `length`: Length of the sparse data.
- **Returns**: Array of integer SRP hash codes.

#### `~SparseRandomProjection()`
- **Destructor**: Frees allocated memory.


## 7. MurmurHash (MurmurHash.h)

### Overview
`MurmurHash` is not an LSH algorithm itself but a fast, non-cryptographic hash function used as a building block within some of the LSH implementations in SLIDE (e.g., WTAHash, DensifiedWtaHash). It provides efficient hashing of data to integers, which are then used in the LSH algorithms.

### Function Definition (C-style header)
```c
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns a murmur hash of `key' based on `seed'
 * using the MurmurHash3 algorithm
 */
uint32_t MurmurHash (const char *, uint32_t, uint32_t);

#ifdef __cplusplus
}
#endif
```

### Function

#### `uint32_t MurmurHash (const char * key, uint32_t len, uint32_t seed)`
- **Purpose**: Computes the MurmurHash3 hash of a given key.
- **Parameters**:
    - `key`: Pointer to the data to be hashed (treated as a byte array).
    - `len`: Length of the key data in bytes.
    - `seed`: Seed value for the hash function, allowing for different hash outputs for the same key with different seeds.
- **Returns**: 32-bit unsigned integer MurmurHash value.


This document provides an overview of the hashing algorithms and related classes in SLIDE. These components are essential for understanding how SLIDE achieves efficient sparse computations and high performance. Further investigation into the implementation details of each class and method is recommended for a deeper understanding.
