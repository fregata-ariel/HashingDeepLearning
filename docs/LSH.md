# LSH.cpp

```cpp
/**
 * @brief Initializes an LSH (Locality-Sensitive Hashing) structure with specified parameters.
 * 
 * The constructor sets up the LSH structure by allocating memory for buckets and initializing random hash values.
 *
 * @param K Number of hash functions per table, controlling the number of hashes used in each dimension.
 * @param L Number of tables, which determines how many times the hashing is performed to reduce false negatives.
 * @param RangePow Exponent used to determine the range of indices for buckets; the number of buckets per table is 2^RangePow.
 */
LSH::LSH(int K, int L, int RangePow)
```

**Step-by-Step Explanation:**

1. **Understanding the Code Structure:**
   - The constructor initializes an LSH structure with parameters K, L, and RangePow.
   - It allocates memory for buckets and random hash values.

2. **Parameter Analysis:**
   - **K:** Used in loops related to generating random numbers (size _K *_L). Indicates the number of hash functions per table.
   - **L:** Defines the number of tables (size L for _bucket array).
   - **RangePow:** Determines the number of buckets per table via 1 << _RangePow, which equals \(2^{\text{RangePow}}\).

3. **Common Notation in LSH:**
   - **K** typically refers to the number of hash functions per table.
   - **L** usually denotes the number of tables used to reduce false negatives.

4. **Alignment with Standard Practices:**
   - The roles of K, L, and RangePow align with standard LSH implementations where multiple tables (L) each use several hash functions (K), and buckets are efficiently indexed using powers of two.

5. **Verification Through Methods:**
   - Methods like `add` and `hashesToIndex` suggest operations consistent with an LSH structure using K hash functions across L tables.

**Conclusion:**

The meanings of the parameters in the LSH constructor were determined by analyzing their usage within the code and aligning them with common practices in Locality-Sensitive Hashing. Specifically:

- **K:** Number of hash functions per table.
- **L:** Number of tables to reduce false negatives.
- **RangePow:** Exponent determining the number of buckets per table as \(2^{\text{RangePow}}\).

This approach ensures efficient hashing and retrieval operations typical in LSH implementations.
