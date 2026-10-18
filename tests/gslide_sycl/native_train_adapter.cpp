// Included around the exact maintained Network::train host body. Runtime and
// layer operations below are explicit mocks; device arithmetic is tested by
// test_network.cpp with the real SYCL CPU. This target checks adapter wiring.
#include "TrainingControl.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
static std::string trace;
static void check(bool x) {
  if (!x)
    throw std::runtime_error("native train adapter contract");
}
static int resets = 0;
static int cudaMemset(void *pointer, int byte, std::size_t bytes) {
  ++resets;
  std::memset(pointer, byte, bytes);
  trace += 'R';
  return 0;
}
static int cudaDeviceSynchronize() {
  trace += 'C';
  return 0;
}
#define CUDA_CHECK(call) check((call) == 0)
struct CscActNodes {
  int node_capacity = 5;
  void extract_from(const std::vector<int> &, const std::vector<float> &,
                    const std::vector<int> &) {
    trace += 'I';
  }
  void extract_from(const std::vector<int> &, const std::vector<int> &) {
    trace += 'L';
  }
};
struct ReluLayer {
  int id;
  CscActNodes csc_acts;
  float delta[5];
  float *d_cmprs_bp_deltas = delta;
  ReluLayer(int id) : id(id) { std::fill_n(delta, 5, 123.f); }
  void forward(CscActNodes &, int, int, int) { trace += 'a'; }
  void forward(ReluLayer &, int, int, int) { trace += 'b'; }
  void bp_first_layer(CscActNodes &, int, int, int) { trace += 'f'; }
  void bp(ReluLayer &previous, int, int, int) {
    check(previous.delta[0] == 0);
    previous.delta[0] += 2;
    previous.delta[0] += 3;
    trace += 'e';
  }
  void update_weights(int, float) { trace += char('g' + id * 2); }
  void update_biases(int, float) { trace += char('h' + id * 2); }
};
struct SoftmaxLayer {
  void forward(ReluLayer &, CscActNodes &, int, int, int, int, int) {
    trace += 'F';
  }
  void bp(ReluLayer &previous, int, int, int, int) {
    for (float x : previous.delta)
      check(x == 0);
    previous.delta[0] += 1;
    previous.delta[0] += 4;
    trace += 'D';
  }
  void update_weights(int, float) { trace += 'k'; }
  void update_biases(int, float) { trace += 'l'; }
  void rebuild(bool shuffle) { trace += shuffle ? 'S' : 'B'; }
};
struct Network {
  std::vector<std::shared_ptr<ReluLayer>> relu_layers{
      std::make_shared<ReluLayer>(0), std::make_shared<ReluLayer>(1)};
  std::shared_ptr<SoftmaxLayer> softmax_layer =
      std::make_shared<SoftmaxLayer>();
  CscActNodes csc_inputs, cmprs_labels;
  void train(const std::vector<int> &, const std::vector<float> &,
             const std::vector<int> &, const std::vector<int> &,
             const std::vector<int> &, const std::vector<int> &, int, float,
             int, int, bool, bool);
};
