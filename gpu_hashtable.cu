#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "gpu_hashtable.hpp"

/*functie hash*/
__device__ long long getHash(int x, int N) {
        if (x < 0) x = -x;
        x = ((x >> 16) ^ x) * HASH_NO;
    	x = ((x >> 16) ^ x) * HASH_NO;
    	x = (x >> 16) ^ x;
		x = x % N;
    	return x;
}

/*copie o valoare a unei chei dintr-un hashtable in altul*/
__global__ void reshape_hashT(hashTable h1, long siz1, hashTable h2, long siz2) {
	int vall, idx = blockIdx.x * blockDim.x + threadIdx.x, key1, key2;
	bool ok = false;
	if ((h1.pairs[idx].key == KEY_INVALID) || (siz1 <= idx))
		return;
	key2 = h1.pairs[idx].key;
	vall = getHash(key2, h2.size);
	for (int i = vall; i < siz2; i++) {
		key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
		if (key1 == KEY_INVALID) {
			h2.pairs[i].value = h1.pairs[idx].value;
			ok = true;
			break;
		} 
	}
	if (!ok) {
		for (int i = 0; i <vall; i++) {
			key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
			if (key1 == KEY_INVALID) {
				h2.pairs[i].value = h1.pairs[idx].value;
				break;
			}

		}
	}
}

/*se insereaza o pereche de (key, value) in hashtable, daca nu se gaseste pe pozitia
 *returnata de functia hash, se va cauta pana la maximul posibil un loc liber, apoi
 *se cauta de la 0 la valoarea functiei
 */
__global__ void insert_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, key, vall;
	
	if (k <= idx) return;
	vall = getHash(keys[idx], siz);
	for (int i = vall; i < siz; i++) {
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		if (key == KEY_INVALID || key == keys[idx]) {
			h.pairs[i].value = values[idx];
			return;
		} 
	}
	for (int i = 0; i < vall; i++) {
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		if (key == KEY_INVALID || key == keys[idx]) {
			h.pairs[i].value = values[idx];
			return;
		}
	}
}

/*cauta valoarea unei chei in hashtable: se apeleaza functia hash, si daca 
 *valoarea nu se gaseste acolo, se face asemenator cu insert: se cauta pana
 *la dimensiunea maxima, apoi de la 0 la valoarea de hash
 */
__global__ void get_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, vall;
	if (k<=idx) return;
	vall = getHash(keys[idx], siz);
	for (int i = vall; i < siz; i++) {
		if (h.pairs[i].key == keys[idx]) {
			values[idx] = h.pairs[i].value;
			return;
		} 
	}
	for (int i = 0; i < vall; i++) {
		if (h.pairs[i].key == keys[idx]) {
			values[idx] = h.pairs[i].value;
			return;
		}
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	hashT.size = size;
	cntPairs = 0;
	hashT.pairs = nullptr;
	cudaMalloc(&hashT.pairs, size * sizeof(pair));
	cudaMemset(hashT.pairs, 0, size * sizeof(pair));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashT.pairs);
}

/* RESHAPE HASH
 * cresc marimea noului table si se apeleaza reshape_hashT pt a muta valorile
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int k = hashT.size / THREADS_NO;  
	if (!(hashT.size % THREADS_NO == 0)) k = k + 1;
	hashTable newH;
	newH.size = numBucketsReshape;

	cudaMalloc(&newH.pairs, numBucketsReshape * sizeof(pair));
	cudaMemset(newH.pairs, 0, numBucketsReshape * sizeof(pair));
	reshape_hashT<<< k, THREADS_NO >>>(hashT, hashT.size, newH, newH.size);

	cudaDeviceSynchronize();
	cudaFree(hashT.pairs);
	hashT = newH;
}

/* INSERT BATCH
 * se insereaza perechile, dandu-se reshape la hashtable daca e necesar
 * prin cudaMemcpy se pun datele in VRAM
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *aKeys, *aValues, k = numKeys / THREADS_NO, nr = cntPairs + numKeys;
	if (numKeys % THREADS_NO != 0) k++;   
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	cudaMalloc(&aValues, numKeys * sizeof(int));

	if (nr / hashT.size >= LOADFACTOR_MAX) reshape((int) (nr / LOADFACTOR_MIN));

	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(aValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	insert_hash<<< k, THREADS_NO>>>(numKeys, aKeys, aValues, hashT, hashT.size);

	cudaDeviceSynchronize();
	cudaFree(aKeys);
	cudaFree(aValues);
	cntPairs += numKeys;
	return true;
}

/* GET BATCH
 * se obtine valoarea apeland get_hash pt cheile din keys in valls
 * prin cudaMemcpy se pun datele in VRAM
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *aKeys, *valls, k = numKeys / THREADS_NO;
	if (!(numKeys % THREADS_NO == 0)) k++;
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	cudaMallocManaged(&valls, numKeys * sizeof(int));

	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	get_hash<<<k, THREADS_NO >>>(numKeys, aKeys, valls, hashT, hashT.size);
	cudaDeviceSynchronize();
	cudaFree(aKeys);
	return valls;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	if (hashT.size == 0) return 0;
	return (float(cntPairs) / hashT.size);
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
