#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>

/**
 * @brief Native hash-table set store used by the released MONGOOSE SLIDE path.
 *
 * Each item has one integer fingerprint per table. Query operations return the
 * union (or one set per query) of item ids found in matching buckets. remove()
 * writes tombstones that query paths erase before exposing results.
 *
 * Ownership: the table copies integer keys and item ids into STL containers;
 * pointers supplied by the Cython boundary are borrowed only during calls and
 * are never stored. Query results are value-owned STL containers.
 *
 * Preconditions: one fingerprint row contains L integer components. Batched
 * operations receive N consecutive rows. query_multi_mask writes into a
 * caller-owned M*N float buffer and never retains it.
 *
 * Traceability relation: Support. This is native infrastructure for the
 * released MONGOOSE-SLIDE path, not a separate MONGOOSE algorithmic claim.
 * TRACE_TEST_ID: MONGOOSE-NATIVE-LSH-SET.
 */
class LSH
{
	private:
		// Members
		int counter;
		int rnd;
		const int K;
		const int L;
		const int THREADS;
		std::vector<std::unordered_map<int, std::vector<int>>> tables;

		// Functions
		void add(const int, const int, const int);
		void add_multi(const int*, const int, const int);

		void retrieve(bool*, const int, const int);
		void retrieve(std::unordered_set<int>&, const int, const int);
		void retrieve_mask(const int*, float*, const int, const int, const int);
		// void retrieve_mask(const int*, float*, float*, const int, const int, const int);

	public:
		LSH(int, int, int);
        void remove(const int*, int);
		void insert(const int*, const int);
		void insert_multi(const int*, const int);
		std::unordered_set<int> query(const int*);
		std::unordered_set<int> query_multi(const int*, const int);
		void query_multi_mask(const int*, float*, const int, const int);
		void query_multi_mask_L(const int*, float*, float*,const int, const int);
		std::vector<std::unordered_set<int>> query_multiset(const int* fp, const int N);
		void clear();
		std::vector<int> print_stats();
};
