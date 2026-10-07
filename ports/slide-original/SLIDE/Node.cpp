#include "Node.h"
#include <random>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <sys/mman.h>
#include "Config.h"

using namespace std;

Node::Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel)
{
	_dim = dim;
	_IDinLayer = nodeID;
	_type = type;
	_layerNum = layerID;
    _currentBatchsize = batchsize;

	if (ADAM)
	{
		_adamAvgMom = adamAvgMom;
		_adamAvgVel = adamAvgVel;
		_t = new float[_dim]();

	}

	_train = new train[_currentBatchsize];
	_activeInputs = 0;

    _weights = weights;
    _bias = bias;
	_mirrorbias = _bias;

}

void Node::Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, train* train_blob)
{
    _dim = dim;
    _IDinLayer = nodeID;
    _type = type;
    _layerNum = layerID;
    _currentBatchsize = batchsize;

    if (ADAM)
    {
        _adamAvgMom = adamAvgMom;
        _adamAvgVel = adamAvgVel;
        _t = new float[_dim]();

    }

    _train = train_blob + nodeID * batchsize;
    _activeInputs = 0;

    _weights = weights;
    _bias = bias;
    _mirrorbias = _bias;

}

float Node::getLastActivation(int inputID)
{
	if(_train[inputID]._ActiveinputIds != 1)
		return 0.0;
	return _train[inputID]._lastActivations;
}


void Node::incrementDelta(int inputID, float incrementValue)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	if (_train[inputID]._lastActivations > 0)
	    _train[inputID]._lastDeltaforBPs += incrementValue;
}

bool Node::getInputActive(int inputID)
{
    return _train[inputID]._ActiveinputIds == 1;
}

bool Node::getActiveInputs(void)
{
    return _activeInputs > 0;
}

/**
 * @brief Compute one selected neuron's activation from a sparse input vector.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 3.1 sparse feed-forward pass and Algorithm 1.
 * Only the supplied active input coordinates participate in the dot product.
 *
 * @par Preconditions / side effects
 * indices/values are borrowed arrays of length 'length'; every index must be
 * within this Node's weight dimension. The function marks inputID active and
 * mutates the Layer-owned train record referenced by _train.
 *
 * @par Traceability
 * TRACE_TEST_ID: SLIDE2020-SPARSE-NODE-MATH.
 */
float Node::getActivation(int* indices, float* values, int length, int inputID)
{
	assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));

	//FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
	if (_train[inputID]._ActiveinputIds != 1) {
	    _train[inputID]._ActiveinputIds = 1; //activate input
	    _activeInputs++;
	}

	_train[inputID]._lastActivations = 0;
	for (int i = 0; i < length; i++)
	{
	    _train[inputID]._lastActivations += _weights[indices[i]] * values[i];
	}
	_train[inputID]._lastActivations += _bias;

	switch (_type)
	{
	case NodeType::ReLU:
		if (_train[inputID]._lastActivations < 0) {
		    _train[inputID]._lastActivations = 0;
		    _train[inputID]._lastGradients = 1;
		    _train[inputID]._lastDeltaforBPs = 0;

        }else{
            _train[inputID]._lastGradients = 0;
		}
		break;
	case NodeType::Softmax:

		break;
	default:
		cout << "Invalid Node type from Constructor" <<endl;
		break;
	}

	return _train[inputID]._lastActivations;
}


/**
 * @brief Normalize one selected output activation and form its training delta.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 3.1 output-layer work in the sparse training
 * path. Softmax itself is a neural-network prerequisite rather than a SLIDE
 * contribution.
 *
 * @par Implementation note
 * The stored delta follows the released update sign convention:
 * target_probability - predicted_probability, divided by batch size.
 *
 * @par Traceability
 * TRACE_TEST_ID: SLIDE2020-SOFTMAX-GRADIENT.
 */
void Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds ==1));

	_train[inputID]._lastActivations /= normalizationConstant + 0.0000001;

	//TODO:check  gradient
	_train[inputID]._lastGradients = 1;
	if (find (label, label+labelsize, _IDinLayer)!= label+labelsize) {
	    _train[inputID]._lastDeltaforBPs = (1.0/labelsize - _train[inputID]._lastActivations) / _currentBatchsize;
	}
	else {
	    _train[inputID]._lastDeltaforBPs = (-_train[inputID]._lastActivations) / _currentBatchsize;
	}
}


/**
 * @brief Backpropagate through the selected previous-layer nodes.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 3.1 sparse backpropagation and Algorithm 1.
 *
 * @par Implementation note
 * Delta propagation is ReLU-gated by the previous node's stored activation.
 * previousNodes and previousLayerActiveNodeIds are borrowed; this Node does
 * not acquire ownership. With ADAM enabled, this routine accumulates selected
 * weight gradients in _t; the later Network optimizer traversal is layer-wide.
 * The active input state is consumed/reset as a side effect.
 *
 * @par Traceability
 * TRACE_TEST_ID: SLIDE2020-TWO-LAYER-BACKWARD.
 */
void Node::backPropagate(Node* previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
	    Node* prev_node = &(previousNodes[previousLayerActiveNodeIds[i]]);
	    prev_node->incrementDelta(inputID, _train[inputID]._lastDeltaforBPs * _weights[previousLayerActiveNodeIds[i]]);

		float grad_t = _train[inputID]._lastDeltaforBPs * prev_node->getLastActivation(inputID);

		if (ADAM)
		{
			_t[previousLayerActiveNodeIds[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _train[inputID]._lastDeltaforBPs;
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
    {
        _mirrorbias += learningRate * _train[inputID]._lastDeltaforBPs;
    }

	_train[inputID]._ActiveinputIds = 0;
	_train[inputID]._lastDeltaforBPs = 0;
	_train[inputID]._lastActivations = 0;
	_activeInputs--;

}


/**
 * @brief Accumulate first-layer gradients on the active sparse coordinates.
 *
 * @par Paper mapping
 * SLIDE (MLSys 2020), Section 3.1 sparse backpropagation / gradient update.
 *
 * @par Implementation note
 * nnzindices/nnzvalues are borrowed arrays of length nnzSize. With ADAM
 * enabled this routine accumulates d*x in _t and d in _tbias.
 * Network::ProcessInput subsequently applies Adam over the layer arrays, so
 * sparse gradient accumulation must not be described as a sparse optimizer
 * state update. The input's active state is consumed/reset.
 *
 * @par Traceability
 * TRACE_TEST_ID: SLIDE2020-SPARSE-NODE-MATH.
 */
void Node::backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	for (int i = 0; i < nnzSize; i++)
	{
		float grad_t = _train[inputID]._lastDeltaforBPs * nnzvalues[i];
		float grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[nnzindices[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[nnzindices[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _train[inputID]._lastDeltaforBPs;
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
	{
		_mirrorbias += learningRate * _train[inputID]._lastDeltaforBPs;
	}

	_train[inputID]._ActiveinputIds = 0;//deactivate inputIDs
	_train[inputID]._lastDeltaforBPs = 0;
	_train[inputID]._lastActivations = 0;
    _activeInputs--;
}

void Node::SetlastActivation(int inputID, float realActivation)
{
    _train[inputID]._lastActivations = realActivation;
}

Node::~Node()
{

	delete[] _indicesInTables;
	delete[] _indicesInBuckets;

	if (ADAM)
	{
		// _adamAvgMom/_adamAvgVel are borrowed slices of Layer-owned arrays.
		// Deleting them here frees interior pointers and corrupts teardown (#3).
		delete[] _t;
	}
}


// for debugging gradients.
float Node::purturbWeight(int weightid, float delta)
{
	_weights[weightid] += delta;
	return _weights[weightid];
}


float Node::getGradient(int weightid, int inputID, float InputVal)
{
	return -_train[inputID]._lastDeltaforBPs * InputVal;
}
