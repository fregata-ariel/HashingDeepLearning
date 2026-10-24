#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <limits>

#include "GPUTimer.h"
#include "Network.h"
#include "TrainingControl.h"

// CUDA 12.0 Thrust max_element copies a tuple<float, ptrdiff_t> with padding.
// Use a fully initialized eight-byte reduction value for device diagnostics.
namespace {
struct IndexedActivation { float value; int index; };
static_assert(sizeof(IndexedActivation) == 8, "argmax storage must have no padding");
struct ReadActivation {
  const float *values;
  __host__ __device__ IndexedActivation operator()(int index) const {
    return {values[index], index};
  }
};
struct ActivationMaximum {
  __host__ __device__ IndexedActivation operator()(IndexedActivation a,
                                                  IndexedActivation b) const {
    return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
  }
};
} // namespace

Network::Network(const std::vector<int> &node_num_per_layer,
                 const std::vector<int> &node_capacity_per_layer,
                 const int input_size, const int max_batch_size,
                 const int input_capacity, const int label_capacity,
                 const int K, const int L, const int bin_size,
                 const int bucket_num_per_tbl, const int bucket_capacity,
                 const int threshold, const int min_softmax_act_num,
                 const int tbl_num_per_tile, const int tbl_num_per_thread,
                 const int linked_bucket_num_per_tbl,
                 const int linked_pool_size)
    : csc_inputs(max_batch_size, input_capacity),
      cmprs_labels(max_batch_size, label_capacity, false),
      layer_num(node_num_per_layer.size()) {
  assert(layer_num >= 2);
  assert(layer_num == node_capacity_per_layer.size());

  relu_layers = std::vector<std::shared_ptr<ReluLayer>>(layer_num - 1);
  for (int l = 0; l < layer_num; ++l) {
    if (l == 0) {
      relu_layers[l] = std::make_shared<ReluLayer>(
          input_size, node_num_per_layer[l], max_batch_size,
          node_capacity_per_layer[l]);
    } else if (l + 1 == layer_num) {
      softmax_layer = std::make_shared<SoftmaxLayer>(
          node_num_per_layer[l - 1], node_num_per_layer[l], max_batch_size,
          node_capacity_per_layer[l], K, L, bin_size, bucket_num_per_tbl,
          bucket_capacity, threshold, min_softmax_act_num, tbl_num_per_tile,
          tbl_num_per_thread, linked_bucket_num_per_tbl, linked_pool_size);
    } else {
      relu_layers[l] = std::make_shared<ReluLayer>(
          node_num_per_layer[l - 1], node_num_per_layer[l], max_batch_size,
          node_capacity_per_layer[l]);
    }
  }
}

Network::~Network() {
  csc_inputs.free();
  cmprs_labels.free();
}

int Network::eval(const std::vector<int> &h_cmprs_input_nodes,
                  const std::vector<float> &h_cmprs_input_vals,
                  const std::vector<int> &h_cmprs_input_offsets,
                  const std::vector<int> &h_cmprs_label_nodes,
                  const std::vector<int> &h_cmprs_label_offsets,
                  const int batch_size, const int thread_num) {
  // forward
  csc_inputs.extract_from(h_cmprs_input_nodes, h_cmprs_input_vals,
                          h_cmprs_input_offsets);

  for (int l = 0; l < relu_layers.size(); ++l) {
    const int node_num = relu_layers[l]->node_num;
    if (l == 0) {
      relu_layers[l]->forward(csc_inputs, batch_size, thread_num, node_num);
    } else {
      relu_layers[l]->forward(*relu_layers[l - 1], batch_size, thread_num,
                              node_num);
    }
  }
  softmax_layer->forward_dense(*relu_layers.back(), batch_size);

  CUDA_CHECK(cudaDeviceSynchronize());

  int correct_cnt = 0;
  for (int b = 0; b < batch_size; b++) {
    const float *begin =
        softmax_layer->d_dense_activations + b * softmax_layer->node_num;
    const int max_node = thrust::transform_reduce(
        thrust::device, thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(softmax_layer->node_num),
        ReadActivation{begin}, IndexedActivation{-std::numeric_limits<float>::infinity(), 0},
        ActivationMaximum{}).index;

    const int label_begin = h_cmprs_label_offsets[b];
    const int label_end = h_cmprs_label_offsets[b + 1];
    if (std::find(h_cmprs_label_nodes.begin() + label_begin,
                  h_cmprs_label_nodes.begin() + label_end,
                  max_node) != h_cmprs_label_nodes.begin() + label_end) {
      correct_cnt++;
    }
  }

  return correct_cnt;
}

/**
 * @brief Run native transfers, selection, reset, backward, Adam and rebuild.
 * @par Relation Support for Section 4.4; operation ordering is shared with CPU.
 * @par Preconditions Valid layer/batch/CSC capacities and caller-corrected lr.
 * Hidden compressed delta capacity is cleared before each backward phase;
 * legitimate accumulation within the phase is retained. Call completes device
 * work before returning. Network/layers own their buffers and native handles.
 * Native acceptance covers a bounded deterministic two-update FP64 oracle,
 * changed ReLU gates, spare-capacity reset and separate stale compressed IDs.
 * Checkpoint serialization, sparse training quality and performance are pending.
 * TRACE_TEST_ID: GSLIDE-TRAINING-CUDA-43
 */
void Network::train(const std::vector<int> &h_cmprs_input_nodes,
                    const std::vector<float> &h_cmprs_input_vals,
                    const std::vector<int> &h_cmprs_input_offsets,
                    const std::vector<int> &h_cmprs_label_nodes,
                    const std::vector<int> &h_cmprs_label_offsets,
                    const std::vector<int> &max_act_nums, const int batch_size,
                    const float lr, const int max_label_num,
                    const int thread_num, const bool rebuild,
                    const bool reshuffle) {
  gslide::TrainingOperations ops;
  ops.hidden_layers = static_cast<int>(relu_layers.size());
  ops.transfer = [&]() {
    csc_inputs.extract_from(h_cmprs_input_nodes, h_cmprs_input_vals,
                            h_cmprs_input_offsets);
    cmprs_labels.extract_from(h_cmprs_label_nodes, h_cmprs_label_offsets);
  };
  ops.hidden_forward = [&](int l) {
    if (l == 0)
      relu_layers[l]->forward(csc_inputs, batch_size, thread_num, max_act_nums[l]);
    else
      relu_layers[l]->forward(*relu_layers[l - 1], batch_size, thread_num,
                              max_act_nums[l]);
  };
  ops.output_forward = [&]() {
    softmax_layer->forward(*relu_layers.back(), cmprs_labels, batch_size,
                           thread_num, *(max_act_nums.end() - 2),
                           max_act_nums.back(), max_label_num);
  };
  ops.reset_hidden_deltas = [&]() {
    for (const auto &layer : relu_layers)
      CUDA_CHECK(cudaMemset(layer->d_cmprs_bp_deltas, 0,
                             sizeof(float) * layer->csc_acts.node_capacity));
  };
  ops.output_backward = [&]() {
    softmax_layer->bp(*relu_layers.back(), batch_size, thread_num,
                      *(max_act_nums.end() - 2), max_act_nums.back());
  };
  ops.hidden_backward = [&](int l) {
    if (l == 0)
      relu_layers[l]->bp_first_layer(csc_inputs, batch_size, thread_num,
                                     max_act_nums[l]);
    else
      relu_layers[l]->bp(*relu_layers[l - 1], batch_size, thread_num,
                         max_act_nums[l]);
  };
  ops.hidden_update = [&](int l) {
    relu_layers[l]->update_weights(thread_num, lr);
    relu_layers[l]->update_biases(thread_num, lr);
  };
  ops.output_update = [&]() {
    softmax_layer->update_weights(thread_num, lr);
    softmax_layer->update_biases(thread_num, lr);
  };
  ops.rebuild_index = [&](bool shuffle) { softmax_layer->rebuild(shuffle); };
  ops.complete = [&]() { CUDA_CHECK(cudaDeviceSynchronize()); };
  gslide::train_step(ops, rebuild, reshuffle);
}
