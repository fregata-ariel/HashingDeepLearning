# Configuration Parameters in SLIDE (Config.h)

The `Config.h` file in SLIDE defines various configuration parameters that control the behavior of the system. These parameters include settings for hash functions, sampling modes, learning rates, and more. Understanding these parameters is crucial for configuring and tuning SLIDE for different datasets and experiments.

## Core Configuration Macros

### Optimization and Algorithm Choices

- **`#define ADAM 1`**: Enables or disables the Adam optimizer. (1 for enabled, 0 for disabled - if other optimizers are available, they would be selected with different values).
- **`#define BETA1 0.9`**: Beta1 parameter for Adam optimizer (exponential decay rate for first moment estimates).
- **`#define BETA2 0.999`**: Beta2 parameter for Adam optimizer (exponential decay rate for second moment estimates).
- **`#define EPS 0.00000001`**: Epsilon value (small constant) used in Adam optimizer to prevent division by zero.

### Hash Function Selection

- **`#define HASH_FUNCTION_WTA 1`**: Identifier for Winner-Takes-All Hash function.
- **`#define HASH_FUNCTION_DWTA 2`**: Identifier for Densified Winner-Takes-All Hash function.
- **`#define HASH_FUNCTION_TOPK_MIN 3`**: Identifier for Top-K Minhash function.
- **`#define HASH_FUNCTION_SIMHASH 4`**: Identifier for SimHash function.
- **`#define HashFunction 2`**:  **Active Hash Function Choice**: Selects the hash function to be used by SLIDE. Set to one of the `HASH_FUNCTION_` defines (e.g., `2` for DWTA Hash).

### LSH Parameters

- **`#define BUCKETSIZE 128`**: Size of buckets in the LSH hash tables. This limits the number of neuron IDs stored in each bucket.
- **`#define TOPK 30`**: Parameter specific to Top-K Minhash (needs clarification - likely related to the 'K' in Top-K).
- **`#define Ratio 3`**: Ratio parameter for SimHash (needs clarification - likely related to sparsity ratio in random projections).
- **`#define binsize 8`**: Bin size parameter for WTA/DWTA Hash functions (needs clarification - likely related to segment size in permutation-based hashing).

### Sampling Mode

- **`#define MODE_TOPK_THRESHOLD 1`**: Identifier for Top-K Thresholding sampling mode.
- **`#define MODE_SAMPLING 4`**: Identifier for Vanilla Sampling mode.
- **`#define Mode 4`**: **Active Sampling Mode Choice**: Selects the sampling strategy for neuron activation retrieval. Set to one of the `MODE_` defines (e.g., `4` for Vanilla Sampling).
- **`#define THRESH 2`**: Threshold parameter for Top-K Thresholding mode (needs clarification - likely minimum frequency for neuron selection).

### Other Configuration

- **`#define FIFO 1`**:  FIFO (First-In-First-Out) policy for bucket management (if `1`, FIFO is used; otherwise, another policy like Reservoir Sampling might be used, though not explicitly defined here).
- **`#define LOADWEIGHT 0`**: Flag to enable/disable loading pre-trained weights from file (1 for load weights, 0 for train from scratch).
- **`#define MAPLEN 325056`**: Parameter `MAPLEN` (purpose needs clarification - likely related to memory mapping or data loading).


## Runtime Configuration Variables (main.cpp)

These variables, while not defined in `Config.h`, are configured at runtime, often through command-line arguments or configuration files parsed in `main.cpp`. They control aspects like network architecture, dataset paths, and training hyperparameters.

- **`int *RangePow;`**: Array defining the range exponent for each layer (purpose needs clarification - likely related to hash function range).
- **`int *K;`**: Array defining the 'K' parameter (number of hash functions per table) for LSH in each layer.
- **`int *L;`**: Array defining the 'L' parameter (number of hash tables) for LSH in each layer.
- **`float *Sparsity;`**: Array defining sparsity levels for each layer (or pairs of layers - needs clarification).
- **`int Batchsize = 1000;`**: Batch size for training.
- **`int Rehash = 1000;`**: Frequency of LSH table rehashing (in iterations).
- **`int Rebuild = 1000;`**: Frequency of LSH table rebuilding (in iterations).
- **`int InputDim = 784;`**: Dimensionality of the input data.
- **`int totRecords = 60000;`**: Total number of training records.
- **`int totRecordsTest = 10000;`**: Total number of test records.
- **`float Lr = 0.0001;`**: Learning rate for training.
- **`int Epoch = 5;`**: Number of training epochs.
- **`int Stepsize = 20;`**: Step size for evaluation frequency (evaluation performed every `Stepsize` iterations).
- **`int *sizesOfLayers;`**: Array defining the size (number of nodes) of each layer in the network.
- **`int numLayer = 3;`**: Number of layers in the network.
- **`string trainData = "";`**: Path to the training data file.
- **`string testData = "";`**: Path to the test data file.
- **`string Weights = "";`**: Path to the file for loading pre-trained weights.
- **`string savedWeights = "";`**: Path to the file for saving trained weights.
- **`string logFile = "";`**: Path to the log file for outputting training and evaluation metrics.


## Notes

- **Clarification Needed**: Several parameters (e.g., `RangePow`, `Ratio`, `binsize`, `THRESH`, `MAPLEN`, `RangePow` arrays, `Sparsity` array) require further investigation of the source code or related documentation to fully understand their purpose and optimal settings.
- **Tuning**: The optimal configuration parameters often depend on the specific dataset, network architecture, and computational resources. Experimentation and validation are essential for tuning SLIDE for best performance.
- **Flexibility**: While `Config.h` provides a static configuration through macros, runtime variables offer flexibility to adjust settings without recompiling, making experiments and deployment more convenient.


This documentation outlines the configuration parameters available in `Config.h` and runtime variables in `main.cpp` for the SLIDE project. It serves as a starting point for users to understand and customize SLIDE for their specific deep learning tasks.
