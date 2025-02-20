
# Config.h

## Description

The `Config.h` header file contains various macro definitions that configure the behavior of the system. These macros are used throughout the codebase to control settings such as optimization methods, hashing algorithms, bucket sizes, and more.

### Key Features

- **Optimization Settings**:
  - ADAM: Enables the Adam optimization algorithm.
  - BETA1, BETA2, EPS: Parameters for the Adam optimizer.
  
- **Hashing Algorithms**:
  - HASH_FUNCTION_* constants define different hashing methods (WTA, Densified WTA, Topk Minhash, Simhash).
  - HashFunction macro specifies which hashing algorithm to use (currently set to Densified WTA).

- **System Configuration**:
  - BUCKETSIZE: Size of buckets used in data grouping.
  - TOPK: Number of top elements considered in some hashing methods.
  - Ratio: Used in Simhash for determining bit settings.
  - binsize: Determines how data is partitioned in WTA/Densified WTA.

- **Mode and Thresholds**:
  - Mode defines processing method (Sampling or Topk thresholding).
  - THRESH: A threshold value for certain conditions.

- **Flags**:
  - FIFO: Indicates whether to use FIFO mechanism.
  - LOADWEIGHT: Flag for loading pre-trained weights.

- **Other Configuration**:
  - MAPLEN: Length of a map structure used elsewhere in the code.

## Usage

This file is included in other source files to access these configurations. It allows developers to easily modify system behavior by changing macro definitions without altering core logic.

---

# Bucket.h

## Description

`Bucket.h` defines a bucket data structure, likely used for grouping items or managing hash tables. Buckets are essential for efficiently organizing and accessing data, especially in hashing algorithms.

### Key Features

- **Data Grouping**: Organizes data into buckets to facilitate quick access.
- **Hashing Support**: May be used with LSH or other hashing techniques to store hashed values.

### Usage

Incorporated into hashing implementations (e.g., WTA, Densified WTA) to manage data storage and retrieval efficiently.

---

# Layer.cpp

## Description

`Layer.cpp` implements the `Layer` class, representing a neural network layer. It handles activation functions, forward propagation, backpropagation, and other layer-specific operations.

### Key Features

- **Forward Propagation**: Computes outputs for each neuron in the layer.
- **Backpropagation**: Calculates gradients for weight updates during training.
- **Activation Functions**: Implements ReLU or similar activation methods.

### Usage

Used within the `Network` class to build multi-layer neural networks. Each layer processes data sequentially, contributing to the network's overall functionality.

---

# Network.cpp

## Description

The `Network.cpp` file implements the `Network` class, managing multiple layers and handling training, inference, and optimization tasks.

### Key Features

- **Forward Pass**: Executes forward propagation across all layers.
- **Backward Pass**: Coordinates backpropagation for gradient calculation.
- **Training Loop**: Manages epochs, mini-batches, and updates model parameters.
- **Optimization**: Uses Adam optimizer to update weights based on gradients.

### Usage

Serves as the main interface for training and using neural networks. It initializes layers, processes inputs, computes outputs, and trains the model.

---

# Node.cpp

## Description

The `Node.cpp` file implements the `Node` class, representing individual neurons in a neural network layer. Nodes handle activation, weight management, and gradient calculations.

### Key Features

- **Activation Calculation**: Computes node output based on inputs and weights.
- **Backpropagation Support**: Calculates gradients for weights and deltas.
- **Weight Management**: Stores and updates weights using ADAM optimization.
- **Utility Methods**: Includes functions for setting activations, debugging, etc.

### Usage

Nodes are aggregated into layers to form the network. Each node processes input data, computes outputs, and contributes to gradient calculations during training.

---

# LSH.cpp

## Description

`LSH.cpp` implements Locality-Sensitive Hashing (LSH) techniques. It provides methods for hashing high-dimensional data points to enable efficient similarity searches.

### Key Features

- **Hashing Methods**: Implements various LSH algorithms.
- **Similarity Search**: Quickly finds similar items using hashed values.

### Usage

Used in applications requiring fast nearest neighbor searches, such as recommendation systems or databases with large datasets.

---

# WtaHash.cpp

## Description

`WtaHash.cpp` implements the Winner-Take-All (WTA) hashing algorithm. It selects top elements from data bins to generate hash values.

### Key Features

- **WTA Hashing**: For each bin, finds the maximum value and its index.
- **Efficiency**: Quickly identifies top elements for hashing purposes.

### Usage

Used in scenarios where efficient hashing of high-dimensional data is needed, particularly in neural network applications for sparse representations.

---

# DensifiedWtaHash.cpp

## Description

`DensifiedWtaHash.cpp` implements an optimized version of WTA hashing, enhancing it with densification to handle sparse data effectively.

### Key Features

- **Permutation and Binning**: Uses permutations to map indices into bins.
- **Densification**: Ensures robustness by filling in gaps when some bins lack sufficient data.
- **Efficient Hashing**: Quickly finds top elements while handling sparse inputs gracefully.

### Usage

Ideal for applications with sparse data, providing reliable hashing even when data distribution is uneven.

---

# main.cpp

## Description

`main.cpp` serves as the entry point of the application. It initializes the system, sets up configurations, trains the network, and processes input data.

### Key Features

- **Initialization**: Sets up neural network layers, training parameters.
- **Training Loop**: Manages epochs and mini-batches for model training.
- **Inference**: Processes new inputs using the trained model.
- **Output Handling**: Manages results and output generation.

### Usage

Runs the entire application workflow, from setup to execution. It's where all components are orchestrated together to perform the intended tasks.
