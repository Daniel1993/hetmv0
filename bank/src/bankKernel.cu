#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_timer.h"
#include <time.h>
#include <math.h>

#include "bankKernel.cuh"
#include "bitmap.h"

#include "pr-stm-wrapper.cuh" // enables the granularity
#include "pr-stm-internal.cuh"

//versions in global memory	10 bits(version),0 bits (padding),20 bits(owner threadID),1 bit(LOCKED),1 bit(pre-locked)
#define	offVers	22
#define offOwn	2
#define offLock	1

#define getVersion(x)     ( ((x) >> offVers) )
#define checkPrelock(x)   ( (x) & 0b1)
#define checkLock(x)      ( ((x) >> offLock) & 0b1)
#define getOwner(x)       ( ((x) >> offOwn) & 0xfffff)
#define maskVersion       0xffc00000

#define finalIdx          (threadIdx.x+blockIdx.x*blockDim.x)
#define newLock(x,y,z)    ( ((x) << offVers) | ((y) << offOwn) | (z))

/****************************************************************************
 *	GLOBALS
 ****************************************************************************/

__constant__ __device__ long  size;
// __constant__ __device__ const int BANK_NB_TRANSFERS;
__constant__ __device__ int   hashNum;
__constant__ __device__ int   num_ways; // TODO: init
__constant__ __device__ int   txsPerGPUThread;
__constant__ __device__ int   txsInKernel;
__constant__ __device__ int   hprob;
__constant__ __device__ float hmult;

/****************************************************************************
 *	KERNELS
 ****************************************************************************/

//   Memory access layout
// +------------+--------------------+
// | NOT ACCESS |      CPU_PART      |
// +------------+--------------------+
// +------------------+--------------+
// |     GPU_PART     |  NOT ACCESS  |
// +------------------+--------------+
//
// random Function random several different numbers and store them into idx(Local array which stores idx of every source in Global memory).
__device__ void random_Kernel(PR_txCallDefArgs, int *idx, int size, int is_intersection)
{
	int i = 0;

#if BANK_PART == 1
	// break the dataset in CPU/GPU
	int is_intersect = is_intersection; //IS_INTERSECT_HIT(PR_rand(100000));
#endif

	int randVal = PR_rand(INT_MAX);
#if BANK_PART == 1
	if (is_intersect) {
		idx[i] = INTERSECT_ACCESS_GPU(randVal, size-BANK_NB_TRANSFERS);
	} else {
		idx[i] = GPU_ACCESS(randVal, size-BANK_NB_TRANSFERS);
	}
#elif BANK_PART == 2
	int is_hot = IS_ACCESS_H(randVal, hprob);
	randVal = PR_rand(INT_MAX);
	if (is_hot) {
		idx[i] = GPU_ACCESS_H(randVal, hmult, size);
	} else {
		idx[i] = GPU_ACCESS_M(randVal, 3*hmult, size);
	}
#else
	idx[i] = INTERSECT_ACCESS_GPU(randVal, size-BANK_NB_TRANSFERS);
#endif

	// TODO: accounts are consecutive
	// generates the target accounts for the transaction
	for (i = 1; i < BANK_NB_TRANSFERS; i++) {
		idx[i] = (idx[i-1] + i) % GPU_TOP_IDX(size);
	}
}

#define COMPUTE_TRANSFER(val) \
	val // TODO: do math that does not kill the final result

/*********************************
 *	bankTx()
 *
 *  Main PR-STM transaction kernel
 **********************************/
/*
* Isto e' o que interessa
*/
__global__ void bankTx(PR_globalKernelArgs)
{
	PR_enterKernel();

	int i = 0, j;	//how many transactions one thread need to commit
	int target;
	PR_GRANULE_T nval;
	int idx[BANK_NB_TRANSFERS];
	PR_GRANULE_T reads[BANK_NB_TRANSFERS];
	HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)input->accounts;
	int is_intersection = input->is_intersection;
	size_t nbAccounts = input->nbAccounts;
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;

  // TODO: it was txsPerGPUThread * iterations
	while (i++ < txsPerGPUThread) { // each thread need to commit x transactions

		// before the transaction pick the accounts (needs PR-STM stuff)
		random_Kernel(PR_txCallArgs, idx, nbAccounts, is_intersection); //get random index

#ifndef BANK_DISABLE_PRSTM
		PR_txBegin();
#endif

		// reads the accounts first, then mutates the locations
		for (j = 0; j < BANK_NB_TRANSFERS; j++)	{
			if (idx[j] < 0 || idx[j] >= nbAccounts) {
	      break;
	    }
			// TODO: make the GPU slower
#ifndef BANK_DISABLE_PRSTM
			reads[j] = PR_read(accounts + idx[j]);
			if (pr_args.is_abort) break; // PR_txBegin is a simple while loop
#else /* PR-STM disabled */
			reads[j] = accounts[idx[j]];
#endif
		}

#ifndef BANK_DISABLE_PRSTM
		if (pr_args.is_abort) { PR_txRestart(); } // optimization
#endif

		for (j = 0; j < BANK_NB_TRANSFERS / 2; ++j) {
			// TODO: make the GPU slower
			target = j*2;
			if (idx[j] < 0 || idx[j] >= nbAccounts || idx[target] < 0 || idx[target] >= nbAccounts) {
				break;
			}
			__syncthreads();
			nval = COMPUTE_TRANSFER(reads[target] - 1); // -money
#ifndef BANK_DISABLE_PRSTM
			PR_write(accounts + idx[target], nval); //write changes to write set
			if (pr_args.is_abort) break;
#else /* PR-STM disabled */
			accounts[idx[target]] = nval; //write changes to write set
#endif

			target = j*2+1;
			__syncthreads();
			nval = COMPUTE_TRANSFER(reads[target] + 1); // +money
#ifndef BANK_DISABLE_PRSTM
			PR_write(accounts + idx[target], nval); //write changes to write set
			if (pr_args.is_abort) break;
#else /* PR-STM disabled */
			accounts[idx[target]] = nval; //write changes to write set
#endif
		}
#ifndef BANK_DISABLE_PRSTM
		if (pr_args.is_abort) { PR_txRestart(); } // optimization
		PR_txCommit();
#else
		if (args.nbCommits != NULL) args.nbCommits[PR_THREAD_IDX]++;
#endif
	}

	PR_exitKernel();
}

// TODO: MemcachedGPU part
// TODO: R/W-set size is NUMBER_WAYS here
// ----------------------

/*********************************
 *	readKernelTransaction()
 *
 *  Main PR-STM transaction kernel
 **********************************/
__global__
__launch_bounds__(1024, 1)
void memcdReadTx(PR_globalKernelArgs)
{
	PR_enterKernel();

	int idx[NUMBER_WAYS];
	int reads[NUMBER_WAYS];
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int j;
	int goal = 0;

	// if (id == 0) printf("Enter memcdReadTx\n");

	HeTM_memcdTx_input_s *input = (HeTM_memcdTx_input_s*)args.inBuf;

	// if (id == 0) printf(" -------------------- \n");

	idx[0] = input->tx_queue[id] * num_ways;

	for (j = 1; j < num_ways; j++) {
		idx[j] = idx[0] + j;
	}

	goal = input->tx_queue[id] & 0xffff;
	goal = (goal % num_ways);

	PR_txBegin();

	for (j = 0; j < goal; j++) {
		reads[j] = PR_read(&input->accounts[idx[j]]);
		// printf("[%2i] read %6i j=%i goal=%i is_abort=%i\n", id, idx[j], j, goal, pr_args.is_abort);
		if (pr_args.is_abort) break;
	}
  //
	if (pr_args.is_abort) { PR_txRestart(); }

	PR_txCommit();

	input->output[id].key    = reads[goal];
	input->output[id].index  = idx[goal];
	input->ts_vec[idx[goal]] = input->clock_value;

	// if (id == 0) printf("Exit memcdReadTx\n");
	PR_exitKernel();
}

// TODO: IMPORTANT ---> set PR_MAX_RWSET_SIZE to number of ways

/*********************************
 *	writeKernelTransaction()
 *
 *  Main PR-STM transaction kernel
 **********************************/
__global__ void memcdWriteTx(PR_globalKernelArgs)
{
	PR_enterKernel();

	int idx[NUMBER_WAYS];
	int reads[NUMBER_WAYS];
	int i = 0;
	int min_val = -1, min_pos = 0;
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int goal = 0;
	int nbAborts = 0;

	HeTM_memcdTx_input_s *input = (HeTM_memcdTx_input_s*)args.inBuf;

	// TODO: always 0 (CPU must tell where to write)
	// input->tx_queue[id] = (id/num_ways) % input->nbAccounts;

	idx[0] = input->tx_queue[id] * num_ways;

	for (int j = 1; j < num_ways; j++) {
		idx[j] = (idx[0] + j) % input->nbAccounts; // TODO: what is going on!?!?
	}

	goal = input->tx_queue[id] & 0xffff;

	for (i = 0; i < txsInKernel; ++i) { // each thread need to commit x transactions

		PR_txBegin();
		nbAborts++;
		if (nbAborts > 10) break; // TODO: some deadlock here (WTF is this doing?)
		min_val = -1;
		min_pos = 0;

		/* Search hash table for an empty spot or an entry to evict */
		for (int j = 0; j < num_ways; j++) {
			reads[j] = PR_read(&input->accounts[idx[j]]);
			if (pr_args.is_abort) { PR_txRestart(); }

			// Check if it is free or if it is the same value
			if (reads[j] == goal || input->ts_vec[idx[j]] == 0) {
				min_pos = j;
				break;
			} else {
				if (min_val == -1 || min_val > input->ts_vec[idx[j]]) {
					min_val = input->ts_vec[idx[j]];
					min_pos = j;
				}
			}
		}
		PR_write(&input->accounts[idx[min_pos]], goal);
		if (pr_args.is_abort) { PR_txRestart(); }

		PR_txCommit();

		input->output[id].key = goal;
		input->output[id].index = idx[min_pos];
		input->ts_vec[idx[min_pos]] = input->clock_value;
	}

	// if (id == 0) printf("Exit memcdWriteTx\n");
	PR_exitKernel();
}
// ----------------------

/****************************************************************************
 *	FUNCTIONS
/****************************************************************************/

extern "C"
cuda_config cuda_configInit(long size, int trans, int hash, int tx, int bl, int hprob, float hmult)
{
	cuda_config c;

	c.size = size;
	c.TransEachThread = trans > 0 ? trans : ( DEFAULT_TransEachThread << 1) / BANK_NB_TRANSFERS;
	c.hashNum = hash > 0 ? hash : DEFAULT_hashNum;
	c.threadNum = tx > 0 ? tx : DEFAULT_threadNum;
	c.blockNum = bl > 0 ? bl : DEFAULT_blockNum;
	c.hprob = hprob > 0 ? hprob : DEFAULT_HPROB;
	c.hmult = hmult > 0 ? hmult : DEFAULT_HMULT;
	// c.BANK_NB_TRANSFERS = (BANK_NB_TRANSFERS > 1 && ((BANK_NB_TRANSFERS & 1) == 0)) ? BANK_NB_TRANSFERS : 2; // DEFAULT_BANK_NB_TRANSFERS

	return c;
}

extern "C"
cudaError_t cuda_configCpyMemcd(cuda_t *c)
{
	cudaError_t cudaStatus;
	int err = 1;

	while (err) {
		err = 0;
		cudaStatus = cudaMemcpyToSymbol(num_ways, &c->num_ways, sizeof(int), 0, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy to device failed for size!");
			continue;
		}
	}
	return cudaStatus;
}

extern "C"
void cuda_configCpy(cuda_config c)
{
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(size, &c.size, sizeof(long), 0, cudaMemcpyHostToDevice ), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(hashNum, &c.hashNum, sizeof(int), 0, cudaMemcpyHostToDevice ), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(txsInKernel, &c.TransEachThread, sizeof(int), 0, cudaMemcpyHostToDevice), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(txsPerGPUThread, &c.TransEachThread, sizeof(int), 0, cudaMemcpyHostToDevice), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(hprob, &c.hprob, sizeof(int), 0, cudaMemcpyHostToDevice), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(hmult, &c.hmult, sizeof(float), 0, cudaMemcpyHostToDevice), "");
	// TODO: find a way of passing BANK_NB_TRANSFERS
};
