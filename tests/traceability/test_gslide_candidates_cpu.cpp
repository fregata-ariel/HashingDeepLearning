// Included after the maintained CUDA bodies by check_gslide_cpu.py.
#include <map>
#include <set>

static std::map<int, int> table_counts(const OwnedTable& table, int sample) {
  const int stride = table.bucket_num_per_tbl + table.pool_size;
  std::map<int, int> result;
  for (int i = 0; i < stride; ++i) {
    const int key = table.d_multi_tbl_keys[sample * stride + i];
    if (key != -1) assert(result.emplace(key, table.d_multi_tbl_vals[sample * stride + i]).second);
    else assert(table.d_multi_tbl_vals[sample * stride + i] == 0);
  }
  return result;
}

// Host adapters for inclusive_scan and copy_if traversal. These invoke the
// maintained filter predicate but do not execute Thrust or get_act_nodes.
static void check_filtered(const OwnedTable& table,
                           const std::vector<std::set<int>>& expected,
                           const std::vector<int>& expected_offsets) {
  GPUMultiLinkedHashTable::filter predicate(table.threshold);
  const int stride = table.bucket_num_per_tbl + table.pool_size;
  std::vector<int> offsets(1, 0), nodes;
  for (int sample = 0; sample < table.max_tbl_num; ++sample) {
    offsets.push_back(offsets.back() + table.d_multi_tbl_sizes[sample]);
    std::set<int> selected;
    for (int i = 0; i < stride; ++i) {
      const int entry = sample * stride + i;
      if (predicate(table.d_multi_tbl_vals[entry])) {
        assert(table.d_multi_tbl_keys[entry] >= 0);
        nodes.push_back(table.d_multi_tbl_keys[entry]);
        assert(selected.insert(table.d_multi_tbl_keys[entry]).second);
      }
    }
    assert(selected == expected[sample]);
    assert(static_cast<int>(selected.size()) == table.d_multi_tbl_sizes[sample]);
    assert(static_cast<int>(nodes.size()) == offsets.back());
  }
  assert(offsets == expected_offsets);
  assert(static_cast<int>(nodes.size()) == expected_offsets.back());
}

// TRACE_TEST_ID: GSLIDE-CANDIDATE-FILTER-CPU
static void test_filter_labels_isolation_and_recreated_query() {
  GPUMultiLinkedHashTable::filter filter(2);
  assert(!filter(0) && !filter(1) && filter(2) && filter(3));
  // Twelve overflow entries exceed the number of distinct keys before insert;
  // one primary bucket forces linked collisions without pool exhaustion.
  OwnedTable first_query(3, 1, 12, 2);
  const int raw[] = {2,7,2,9,7,2,4, 2,8,8,8,7};
  const int begin[] = {0,7,12}, end[] = {7,12,12};
  for (int sample = 0; sample < 3; ++sample) {
    cpu_block(sample, 3);
    first_query.d_block_reduce_cnt(raw, begin[sample], end[sample], sample);
  }
  assert((table_counts(first_query, 0) == std::map<int,int>{{2,3},{7,2},{9,1},{4,1}}));
  assert((table_counts(first_query, 1) == std::map<int,int>{{2,1},{8,3},{7,1}}));
  assert(table_counts(first_query, 2).empty());
  check_filtered(first_query, {{2,7},{8},{}}, {0,2,3,3});
  // Check the serial collision chain separately from set membership. GPU
  // parallel insertion can choose another valid order and remains pending.
  const int stride = 13;
  assert(first_query.d_multi_tbl_pool_used_sizes[0] == 3);
  assert(first_query.d_multi_tbl_pool_used_sizes[1] == 2);
  assert(first_query.d_multi_tbl_pool_used_sizes[2] == 0);
  for (int sample = 0; sample < 2; ++sample) {
    const int occupied = sample == 0 ? 4 : 3;
    for (int i = 0; i < occupied; ++i)
      assert(first_query.d_multi_tbl_nexts[sample*stride+i] == (i+1 < occupied ? i+1 : -1));
    assert(first_query.d_multi_tbl_locks[sample] == 0);
  }
  const int labels[] = {9,6,2,9, 7};
  cpu_block(0, 1); first_query.d_activate_labels_seq(labels, 0, 4, 0);
  cpu_block(0, 1); first_query.d_activate_labels_seq(labels, 4, 5, 1);
  cpu_block(0, 1); first_query.d_activate_labels_seq(labels, 5, 5, 2);
  assert((table_counts(first_query, 0) == std::map<int,int>{{2,5},{7,2},{9,5},{4,1},{6,2}}));
  assert((table_counts(first_query, 1) == std::map<int,int>{{2,1},{8,3},{7,3}}));
  assert(table_counts(first_query, 2).empty());
  check_filtered(first_query, {{2,7,9,6},{8,7},{}}, {0,4,6,6});
  // Recreate per-query storage using the host allocation adapter. This tests
  // the required fresh-table state, not CUDA init_tbls/cudaMemset execution.
  OwnedTable second_query(3, 1, 12, 2);
  const int next_raw[] = {4,4,6};
  cpu_block(0, 1); second_query.d_block_reduce_cnt(next_raw, 0, 3, 0);
  for (int sample = 1; sample < 3; ++sample) {
    cpu_block(0, 1); second_query.d_block_reduce_cnt(next_raw, 3, 3, sample);
  }
  assert((table_counts(second_query, 0) == std::map<int,int>{{4,2},{6,1}}));
  check_filtered(second_query, {{4},{},{}}, {0,1,1,1});
  // The first query retains its own state; keys/counts never cross samples or
  // the independently allocated query objects.
  check_filtered(first_query, {{2,7,9,6},{8,7},{}}, {0,4,6,6});
}

static std::vector<int> fixed_bins() {
  const int second[] = {1,2,3,4,0,5,6,7};
  std::vector<int> bins(3*2*8);
  for (int table = 0; table < 3; ++table) for (int i = 0; i < 8; ++i) {
    bins[(table*2)*8+i] = i;
    bins[(table*2+1)*8+i] = table == 1 ? 7-i : second[i];
  }
  return bins;
}

static void build_index(const std::vector<int>& bins, const std::vector<float>& weights,
                        int capacity, int* buckets, int* sizes) {
  for (int tile = 0; tile < 2; ++tile) {
    cpu_block(tile, 2);
    init_hash_no_sw_knl(bins.data(), weights.data(), 8, weights.size()/8,
                       bins.size(), 3, 2, 8, 2, 36, capacity, buckets, sizes);
  }
}

static void query_index(const std::vector<int>& bins, const std::vector<float>& inputs,
                        const int* sizes, int capacity,
                        std::vector<int>& ids, std::vector<int>& lengths) {
  ids.assign(inputs.size()/8*3, -1); lengths.assign(ids.size(), -1);
  for (int tile = 0; tile < 2; ++tile) {
    cpu_block(tile, 2);
    get_hash_knl(bins.data(), inputs.data(), sizes, 8, bins.size(), 3, 2, 8,
                 2, inputs.size()/8, 36, capacity, ids.data(), lengths.data());
  }
}

static void gather_index(const std::vector<int>& ids, const int* buckets,
                         int capacity, OwnedCsc& gathered) {
  // Single-thread serial launches are sufficient for this independent body.
  for (int tid = 0; tid < static_cast<int>(ids.size()) + 2; ++tid) {
    cpu_block(tid, ids.size()+2);
    gather_buckets_knl(ids.data(), buckets, 3, ids.size()/3, 36, capacity, gathered);
  }
}

static void check_index(const int* buckets, const int* sizes, int capacity,
                        const std::vector<std::vector<int>>& addresses) {
  for (int table = 0; table < 3; ++table) for (int bucket = 0; bucket < 36; ++bucket) {
    std::vector<int> occupants;
    for (int node = 0; node < static_cast<int>(addresses[table].size()); ++node)
      if (addresses[table][node] == bucket) occupants.push_back(node);
    assert(sizes[table*36+bucket] == static_cast<int>(occupants.size()));
    std::vector<int> physical(capacity, -1);
    // This oracle models a bounded ring in explicit serial insertion order.
    // It does not predict which candidates survive concurrent GPU writes.
    for (int i = 0; i < static_cast<int>(occupants.size()); ++i) physical[i%capacity] = occupants[i];
    for (int i = 0; i < capacity; ++i)
      assert(buckets[(table*36+bucket)*capacity+i] == physical[i]);
  }
}

// TRACE_TEST_ID: GSLIDE-BUCKET-RING-CPU
static void test_ring_clamp_gather_and_filter() {
  const auto bins = fixed_bins();
  std::vector<float> weights(5*8, -3);
  const int winners[] = {0,1,2,0,0};
  for (int node = 0; node < 5; ++node) weights[node*8+winners[node]] = 2;
  std::vector<int> guarded_buckets(3*36*2+2, -1), sizes(3*36, 0);
  guarded_buckets.front() = guarded_buckets.back() = -777;
  int* buckets = guarded_buckets.data()+1;
  build_index(bins, weights, 2, buckets, sizes.data());
  // floor(ln(8))=2 preserves the released (0,4)/(1,0) address alias.
  check_index(buckets, sizes.data(), 2, {{4,4,9,4,4},{7,10,13,7,7},{4,4,9,4,4}});
  assert(guarded_buckets.front() == -777 && guarded_buckets.back() == -777);
  std::vector<float> inputs(3*8, -3);
  inputs[0] = 2; inputs[8+1] = 2; inputs[16+3] = 2;
  std::vector<int> ids, lengths;
  query_index(bins, inputs, sizes.data(), 2, ids, lengths);
  assert((ids == std::vector<int>{4,7,4,4,10,4,14,16,14}));
  assert((lengths == std::vector<int>{2,2,2,2,1,2,0,0,0}));
  // Host prefix scan of lengths supplies the actual gather body's offsets.
  std::vector<int> offsets(1, 0);
  for (int length : lengths) offsets.push_back(offsets.back()+length);
  assert((offsets == std::vector<int>{0,2,4,6,8,9,11,11,11,11}));
  OwnedCsc gathered(std::vector<int>(offsets.back(), -1), {}, offsets);
  gather_index(ids, buckets, 2, gathered);
  const std::vector<int> expected_nodes = {3,4,4,3,3,4, 3,4,1,3,4};
  assert(std::equal(expected_nodes.begin(), expected_nodes.end(), gathered.d_nodes));
  OwnedTable candidates(3, 1, 12, 2);
  for (int sample = 0; sample < 3; ++sample) {
    cpu_block(sample, 3);
    candidates.d_block_reduce_cnt(gathered.d_nodes, gathered.d_offsets[sample*3],
                                  gathered.d_offsets[(sample+1)*3], sample);
  }
  assert((table_counts(candidates, 0) == std::map<int,int>{{3,3},{4,3}}));
  assert((table_counts(candidates, 1) == std::map<int,int>{{1,1},{3,2},{4,2}}));
  assert(table_counts(candidates, 2).empty());
  check_filtered(candidates, {{3,4},{3,4},{}}, {0,2,4,4});
}

// TRACE_TEST_ID: GSLIDE-INDEX-REBUILD-CPU
static void test_parameter_change_requires_explicit_rebuild() {
  const auto bins = fixed_bins();
  std::vector<float> weights(3*8, -3);
  for (int node = 0; node < 3; ++node) weights[node*8+node] = 2;
  std::vector<int> buckets(3*36*3, -1), sizes(3*36, 0);
  build_index(bins, weights, 3, buckets.data(), sizes.data());
  check_index(buckets.data(), sizes.data(), 3, {{4,4,9},{7,10,13},{4,4,9}});
  std::vector<float> inputs(2*8, -3); inputs[2] = 2; inputs[8] = 2;
  std::vector<int> ids, lengths;
  query_index(bins, inputs, sizes.data(), 3, ids, lengths);
  assert((ids == std::vector<int>{9,13,9,4,7,4}));
  assert((lengths == std::vector<int>{1,1,1,2,1,2}));
  const auto old_buckets = buckets, old_sizes = sizes;
  // Explicit host parameter change: node 0's winner moves from input 0 to 2.
  // No optimizer, LSH host lifecycle or rebuild scheduling is claimed here.
  weights[0] = -3; weights[2] = 2;
  query_index(bins, inputs, sizes.data(), 3, ids, lengths);
  assert(buckets == old_buckets && sizes == old_sizes);
  assert((lengths == std::vector<int>{1,1,1,2,1,2}));
  OwnedCsc stale(std::vector<int>(8, -1), {}, {0,1,2,3,5,6,8});
  gather_index(ids, buckets.data(), 3, stale);
  const std::vector<int> stale_nodes = {2,2,2,0,1,0,0,1};
  assert(std::equal(stale_nodes.begin(), stale_nodes.end(), stale.d_nodes));
  // Rebuild uses fresh counts/buckets, matching the CUDA reset precondition.
  std::fill(buckets.begin(), buckets.end(), -1); std::fill(sizes.begin(), sizes.end(), 0);
  build_index(bins, weights, 3, buckets.data(), sizes.data());
  check_index(buckets.data(), sizes.data(), 3, {{9,4,9},{13,10,13},{9,4,9}});
  query_index(bins, inputs, sizes.data(), 3, ids, lengths);
  assert((ids == std::vector<int>{9,13,9,4,7,4}));
  assert((lengths == std::vector<int>{2,2,2,1,0,1}));
  OwnedCsc rebuilt(std::vector<int>(8, -1), {}, {0,2,4,6,7,7,8});
  gather_index(ids, buckets.data(), 3, rebuilt);
  const std::vector<int> rebuilt_nodes = {0,2,0,2,0,2,1,1};
  assert(std::equal(rebuilt_nodes.begin(), rebuilt_nodes.end(), rebuilt.d_nodes));
}

int main() {
  test_filter_labels_isolation_and_recreated_query();
  test_ring_clamp_gather_and_filter();
  test_parameter_change_requires_explicit_rebuild();
  std::cout << "GSLIDE_CANDIDATES_CPU_PASS cases=3\n";
}
