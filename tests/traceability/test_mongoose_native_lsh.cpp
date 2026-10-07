#include <cassert>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../../ports/mongoose/lsh_lib/LSH.h"

// TRACE_TEST_ID: MONGOOSE-NATIVE-LSH-SET

static void expectSet(
    const std::unordered_set<int>& actual,
    std::initializer_list<int> expected) {
    std::unordered_set<int> e(expected);
    assert(actual == e);
}

int main() {
    constexpr int L = 2;
    LSH lsh(/*K=*/1, /*L=*/L, /*THREADS=*/1);

    int fp0[L] = {1, 10};
    int fp1[L] = {1, 20};
    int fp2[L] = {2, 10};
    lsh.insert(fp0, 0);
    lsh.insert(fp1, 1);
    lsh.insert(fp2, 2);

    // Query returns the union of matching buckets across tables.
    int q0[L] = {1, 10};
    expectSet(lsh.query(q0), {0, 1, 2});

    // Removal leaves an internal tombstone but it must not escape query().
    lsh.remove(fp1, 1);
    int q1[L] = {1, 20};
    expectSet(lsh.query(q1), {0});

    int queries[2 * L] = {2, 10, 1, 20};
    expectSet(lsh.query_multi(queries, 2), {0, 2});

    auto per_query = lsh.query_multiset(queries, 2);
    assert(per_query.size() == 2);
    expectSet(per_query[0], {0, 2});
    expectSet(per_query[1], {0});

    float mask[2 * 3] = {};
    lsh.query_multi_mask(queries, mask, /*M=*/2, /*N=*/3);
    assert(mask[0 * 3 + 0] == 1.0f);
    assert(mask[0 * 3 + 2] == 1.0f);
    assert(mask[1 * 3 + 0] == 1.0f);
    assert(mask[1 * 3 + 1] == 0.0f);

    lsh.clear();
    expectSet(lsh.query(q0), {});

    // Batch insertion uses rows of L fingerprints and item ids 0..N-1.
    LSH batched(/*K=*/1, /*L=*/L, /*THREADS=*/1);
    int batch_fp[3 * L] = {1, 10, 1, 20, 2, 10};
    batched.insert_multi(batch_fp, 3);
    expectSet(batched.query(q0), {0, 1, 2});

    std::cout << "TRACE_TEST_PASS MONGOOSE-NATIVE-LSH-SET\n";
    return 0;
}
