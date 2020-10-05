#ifndef _HASHCPU_
#define _HASHCPU_

#include <utility>
#include <string>
#define THREADS_NO 1000
#define KEY_INVALID 0
#define HASH_NO 0x45d9f3b
#define LOADFACTOR_MIN 0.8
#define LOADFACTOR_MAX 0.94

#define DIE(assertion, call_description) \
    do {    \
        if (assertion) {    \
        fprintf(stderr, "(%s, %d): ",    \
        __FILE__, __LINE__);    \
        perror(call_description);    \
        exit(errno);    \
    }    \
} while (0)


struct pair {
	int key, value;
};

struct hashTable {
	pair *pairs;
	long size;
};

//
// GPU HashTable
//
class GpuHashTable {
	public:
		long cntPairs;
		hashTable hashT;
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int *values, int numKeys);
		int *getBatch(int *key, int numItems);
		float loadFactor();

		void occupancy();
		void print(std::string info);
		~GpuHashTable();
};

#endif
