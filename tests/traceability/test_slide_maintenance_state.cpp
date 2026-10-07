#include <cassert>
#include <iostream>
#include "../../ports/slide-original/SLIDE/Layer.h"

// TRACE_TEST_ID: SLIDE2020-MAINTENANCE-STATE

static void verifyNodeIndexed(Layer& layer, int nodeId, int tables) {
    Node* node = layer.getNodebyID(nodeId);
    assert(node->_indicesInTables != nullptr);
    assert(node->_indicesInBuckets != nullptr);
    for (int table = 0; table < tables; ++table) {
        const int bucket = node->_indicesInBuckets[table];
        assert(bucket >= 0);
        assert(layer._hashTables->retrieve(
                   table, node->_indicesInTables[table], bucket) == nodeId + 1);
    }
}

int main() {
    constexpr int nodes = 4;
    constexpr int dim = 32;
    constexpr int tables = 2;
    Layer layer(
        nodes, dim, 0, NodeType::ReLU, 1,
        /*K=*/2, /*L=*/tables, /*RangePow=*/6, /*Sparsity=*/0.5f);

    for (int id = 0; id < nodes; ++id)
        verifyNodeIndexed(layer, id, tables);

    const std::size_t initialGeneration = layer.getHashGeneration();

    layer.refreshHashIndex(false);
    assert(layer.getHashGeneration() == initialGeneration);
    for (int id = 0; id < nodes; ++id)
        verifyNodeIndexed(layer, id, tables);

    layer.refreshHashIndex(true);
    assert(layer.getHashGeneration() == initialGeneration + 1);
    for (int id = 0; id < nodes; ++id)
        verifyNodeIndexed(layer, id, tables);

    std::cout << "TRACE_TEST_PASS SLIDE2020-MAINTENANCE-STATE\n";
}
