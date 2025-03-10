# SLIDE Usage Guide

This guide provides instructions on how to use SLIDE for training and evaluating deep learning models. It covers setting up the environment, configuring experiments, running the SLIDE executable, and interpreting the results.

## 1. Prerequisites

Before using SLIDE, ensure you have the following prerequisites installed:

- **CMake (>= 3.0):**  Required for building the SLIDE project.
- **C++ Compiler (C++11 compatible):**  GCC or Clang is recommended.
- **OpenMP:**  Required for multi-threading support. Ensure OpenMP is enabled in your compiler.
- **Linux Environment (Recommended):** While SLIDE might be adaptable to other systems, it is primarily developed and tested on Linux (Ubuntu 16.04+).
- **Transparent Huge Pages (Linux, Optional but Recommended):** For optimal performance, especially with large datasets, enable Transparent Huge Pages as described in the SLIDE paper and `README.md`.

## 2. Building SLIDE

1. **Clone the Repository:** If you haven't already, clone the SLIDE repository:
   ```bash
   git clone [repository_url]
   cd HashingDeepLearning 
   ```

2. **Create a Build Directory:**
   ```bash
   mkdir bin
   cd bin
   ```

3. **Configure with CMake:**
   ```bash
   cmake ..
   ```
   - You can specify the generator and other CMake options if needed (e.g., `cmake -G "Unix Makefiles" ..`).

4. **Build the Project:**
   ```bash
   make 
   ```
   - This command compiles the SLIDE library and the `runme` executable.

## 3. Configuration

SLIDE's behavior is controlled by two main types of configuration:

### 3.1. `Config.h` Macros

- **Static Configuration:**  `Config.h` defines C++ macros that are set at compile time. Modify this file to change core settings like:
    - Hash function type (`HashFunction`)
    - Sampling mode (`Mode`)
    - LSH parameters (`BUCKETSIZE`, `TOPK`, `Ratio`, `binsize`)
    - Optimizer settings (`ADAM`, `BETA1`, `BETA2`, `EPS`)
    - Bucket management policy (`FIFO`)
    - Weight loading (`LOADWEIGHT`)

- **Recompilation Required:** After modifying `Config.h`, you **must** recompile the project using `make` in the `bin` directory for changes to take effect.

### 3.2. Runtime Configuration (Command-line Arguments and Config Files)

- **Dynamic Configuration:** Runtime parameters are set when you run the `runme` executable. These are typically parsed from a configuration file (e.g., `Config_amz.csv`) specified as a command-line argument.
- **Key Runtime Parameters:**
    - Network architecture (`sizesOfLayers`, `numLayer`)
    - LSH layer configuration (`RangePow`, `K`, `L`, `Sparsity`)
    - Training hyperparameters (`Batchsize`, `Lr`, `Epoch`, `Stepsize`)
    - Data paths (`trainData`, `testData`, `logFile`, `Weights`, `savedWeights`)

- **Example Configuration File (`SLIDE/Config_amz.csv`):**
   ```
   # Configuration for Amazon-670K dataset experiment
   RangePow=20,20,20
   K=9,9,9
   L=50,50,50
   Sparsity=0.1,0.1,0.1,0.1,0.1,0.1
   Batchsize=256
   Rehash=10000000
   Rebuild=10000000
   InputDim=135909
   totRecords=490449
   totRecordsTest=153025
   Epoch=20
   Lr=0.00001
   Stepsize=20
   numLayer=3
   sizesOfLayers=128,670091
   trainData=dataset/Amazon/train.txt
   testData=dataset/Amazon/test.txt
   weight=output/Amazon_weights
   savedweight=output/Amazon_weights
   logFile=output/log_amazon.txt
   ```

## 4. Running SLIDE

To run SLIDE, execute the `runme` executable from the `bin` directory, providing the path to your configuration file as a command-line argument:

```bash
cd bin
./runme ../SLIDE/Config_amz.csv 
```

- **Configuration File Path:** Ensure the path to the configuration file is correct relative to your `bin` directory.
- **Output:** SLIDE will output training progress, evaluation metrics, and logs to the console and the specified `logFile`. Weights will be saved to the `savedWeights` path.

## 5. Input Data Format

SLIDE expects input data in a sparse format, suitable for extreme classification datasets. The data format is described in `main.cpp` (see `ReadDataSVM` and `EvalDataSVM` functions), but generally consists of:

- **Text-based files:**  Training and testing data are read from text files specified in the configuration.
- **Sparse Features:** Each line in the data file represents a data instance. Features are represented in a sparse format (index:value pairs).
- **Labels:**  Labels are also included in each line, following the feature representation.

Refer to the example datasets (Delicious-200K, Amazon-670K) and the data loading code in `main.cpp` for precise format details.

## 6. Interpreting Results and Logs

SLIDE outputs the following information during training and evaluation:

- **Console Output:** Real-time updates on training iterations, loss, and evaluation metrics (accuracy).
- **Log File (`logFile`):** Detailed logs are written to the file specified by the `logFile` parameter in the configuration. This file typically includes:
    - Epoch-wise training progress
    - Iteration-wise metrics
    - Evaluation results on the test dataset

- **Weight Files (`savedWeights`):** Trained weights are saved to the file specified by `savedWeights` in `.npz` format (using the `cnpy` library).

Analyze the console output and log files to monitor training progress, evaluate model performance, and diagnose any issues.

## 7. Performance Optimization Tips

- **Enable Transparent Huge Pages:**  For Linux systems, enabling Transparent Huge Pages can significantly reduce memory overhead and improve performance, especially for large models and datasets. Follow the instructions in the SLIDE paper and `README.md`.
- **Tune LSH Parameters (K, L, Bucket Size):** Experiment with different LSH parameter settings to find the optimal balance between sparsity and accuracy for your specific task.
- **Adjust Sparsity Levels:**  The `Sparsity` parameter controls the target sparsity in each layer. Tuning this parameter can impact both performance and model accuracy.
- **Experiment with Hash Functions and Sampling Modes:** SLIDE supports different hash functions (WTAHash, DWTAHash, SimHash, DensifiedMinhash) and sampling modes (Vanilla Sampling, Top-K Thresholding). Experiment with these options to find the best combination for your data.
- **Optimize Batch Size:**  Adjust the `Batchsize` parameter to maximize parallel processing efficiency without compromising generalization performance.
- **CPU Core Utilization:** SLIDE is designed to scale well with multiple CPU cores. Ensure your system is effectively utilizing available cores during training.

## 8. Troubleshooting

- **Compilation Errors:**  Ensure you have all prerequisites installed (CMake, C++ compiler, OpenMP). Check CMake output for specific error messages.
- **Runtime Errors:**  Examine console output and log files for error messages. Double-check configuration file paths and data formats.
- **Performance Issues:** If SLIDE is running slower than expected, review the performance optimization tips above. Verify that Transparent Huge Pages are enabled (if applicable) and that CPU cores are being utilized.

For further assistance or to report issues, please refer to the SLIDE project repository or contact the authors of the SLIDE paper.
