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

#include "bankKernel.cuh"
#include "bitmap.h"

// __constant__ __device__ long dev_basePoint; // TODO: check how global variables are linked in CUDA

// TODO: implement

#ifdef PR_ARGS_S_EXT
#undef PR_ARGS_S_EXT
#endif

#if CMP_TYPE == CMP_EXPLICIT
/* TODO: need logPos (do every thread commit the same number of transactions?) */
#define HeTM_GPU_log_explicit_s \
	unsigned explicitLogBlock; \
	unsigned *explicitLogOffThr; \
//
#define HeTM_GPU_log_explicit_prepare \
	size_t explicitLogCounter = PR_threadNum*PR_blockNum; \
	GPU_log->explicitLogOffThr = (unsigned*)memman_ad_hoc_alloc(NULL, NULL, explicitLogCounter*sizeof(unsigned)); \
	memman_ad_hoc_zero(NULL); \
	GPU_log->explicitLogBlock = EXPLICIT_LOG_BLOCK; \
//
#define HeTM_GPU_log_explicit_before_reads \
	int tid_ = blockIdx.x*blockDim.x + threadIdx.x; \
	int explicitLogOffset = tid_ * GPU_log->explicitLogBlock; \
//
#define HeTM_GPU_log_explicit_after_reads \
	GPU_log->explicitLogOffThr[tid_] += BANK_NB_TRANSFERS; \
//
#define HeTM_GPU_log_explicit_teardown \
	memman_ad_hoc_free(NULL); \
//
#elif CMP_TYPE == CMP_COMPRESSED
#define HeTM_GPU_log_explicit_s                  /* empty */
#define HeTM_GPU_log_explicit_prepare            /* empty */
#define HeTM_GPU_log_explicit_before_reads       /* empty */
#define HeTM_GPU_log_explicit_after_reads        /* empty */
#define HeTM_GPU_log_explicit_teardown           /* empty */
#else
// error or disabled
#define HeTM_GPU_log_explicit_s                  /* empty */
#define HeTM_GPU_log_explicit_prepare            /* empty */
#define HeTM_GPU_log_explicit_before_reads       /* empty */
#define HeTM_GPU_log_explicit_after_reads        /* empty */
#define HeTM_GPU_log_explicit_teardown           /* empty */
#endif

#define PR_ARGS_S_EXT \
	typedef struct { \
		void *dev_rset; \
		int *onIntersect; \
		curandState *state; \
		long CPUAccountsBasePtr; \
		HeTM_GPU_log_explicit_s /* Explicit log only */ \
	} HeTM_GPU_log_s \

#ifdef PR_DEV_BUFF_S_EXT
#undef PR_DEV_BUFF_S_EXT
#endif

#define PR_DEV_BUFF_S_EXT \
	typedef struct { \
		HeTM_GPU_log_s gpuLog; \
		curandState *state; \
	} HeTM_GPU_dbuf_log_s \

#ifdef PR_BEFORE_RUN_EXT
#undef PR_BEFORE_RUN_EXT
#endif

#define PR_BEFORE_RUN_EXT(args) ({ \
	HeTM_GPU_log_s *GPU_log; \
	memman_alloc_dual("HeTM_gpuLog", sizeof(HeTM_GPU_log_s), MEMMAN_THRLC); \
	GPU_log = (HeTM_GPU_log_s*)memman_get_cpu(NULL); \
	/* TODO: explicit log only */ \
	HeTM_GPU_log_explicit_prepare \
	/* ---------------------- */ \
	GPU_log->CPUAccountsBasePtr = (long)args->host.inBuf; \
	memman_select("HeTM_dev_rset"); \
	GPU_log->dev_rset = memman_get_gpu(NULL); \
	memman_select("Stats_OnIntersect"); \
	GPU_log->onIntersect = (int*)memman_get_gpu(NULL); \
	GPU_log->state = (curandState*)parsedData.cd->devStates; \
	args->host.pr_args_ext = (void*)GPU_log; \
	memman_select("HeTM_gpuLog"); \
	args->dev.pr_args_ext = memman_get_gpu(NULL); \
	memman_cpy_to_gpu(NULL); \
}) \

// TODO: no need of coping back the log

#ifdef PR_AFTER_RUN_EXT
#undef PR_AFTER_RUN_EXT
#endif

#define PR_AFTER_RUN_EXT(args) ({ \
	HeTM_GPU_log_explicit_teardown; \
	CPY_BACK_DEBUG(); \
	memman_select("HeTM_gpuLog"); \
	memman_free_dual(); \
}) \

#ifdef HETM_DEB
#define CPY_BACK_DEBUG() \
	memman_select("Stats_OnIntersect"); \
	memman_cpy_to_cpu(NULL); \
	memman_select("HeTM_dev_rset"); \
	memman_cpy_to_cpu(NULL) \
//
#else /* HETM_DEB */
#define CPY_BACK_DEBUG() /* empty */
#endif /* HETM_DEB */

#ifdef PR_AFTER_VAL_LOCKS_EXT
#undef PR_AFTER_VAL_LOCKS_EXT
#endif

// TODO: is not the addr but the index in the account array (TODO: subtract base addr)!

// Logs the read-set after acquiring the locks
// TODO: check write/write conflicts

#if CMP_TYPE == CMP_EXPLICIT
/* TODO: need logPos (do every thread commit the same number of transactions?) */
#define SET_ON_LOG(addr) \
	int *explicitLog = (int*)GPU_log->dev_rset; \
	unsigned logPos = explicitLogOffset + GPU_log->explicitLogOffThr[tid_]; \
	uintptr_t rsetAddr = (uintptr_t)(addr); \
	HeTM_bankTx_input_s *ON_LOG_input = (HeTM_bankTx_input_s*)args->inBuf; \
	uintptr_t devBAddr = (uintptr_t)ON_LOG_input->accounts; \
	uintptr_t pos = (rsetAddr - devBAddr) >> PR_LOCK_GRAN_BITS; /* stores the index instead of the address */ \
	explicitLog[logPos + i] = pos+1 /* 0 is NULL */ \
//
/*if (GPU_log->explicitLogOffThr[tid_]==98) printf("[%i] explicitLogOffset=%i, explicitLogOffThr=%i, i=%i\n", (int)tid_,\
(int)explicitLogOffset, (int)GPU_log->explicitLogOffThr[tid_], i);*/ \
#elif CMP_TYPE == CMP_COMPRESSED
#define SET_ON_LOG(addr) \
	uintptr_t rsetAddr = (uintptr_t)(addr); \
	HeTM_bankTx_input_s *ON_LOG_input = (HeTM_bankTx_input_s*)args->inBuf; \
	uintptr_t devBAddr = (uintptr_t)ON_LOG_input->accounts; \
	uintptr_t pos = (rsetAddr - devBAddr) >> PR_LOCK_GRAN_BITS; \
	unsigned short *RSetBitmap = (unsigned short*)GPU_log->dev_rset; \
	ByteM_SET_POS(pos, RSetBitmap) \
//
#else
// error or disabled
#define SET_ON_LOG(addr) /* empty */
#endif

#define PR_AFTER_VAL_LOCKS_EXT(args) ({ \
  int i; \
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args->pr_args_ext; \
	/* TODO: explicit log only */ \
	HeTM_GPU_log_explicit_before_reads \
	/* ---------------------- */ \
	for (i = 0; i < args->rset.size; i++) { \
		SET_ON_LOG(args->rset.addrs[i]); /* add read to devLogR */ \
	} \
	/* TODO: explicit logOnly */ \
	HeTM_GPU_log_explicit_after_reads /* offset of the next transaction */ \
	/* ---------------------- */ \
}) \

#ifdef PR_AFTER_WRITEBACK_EXT
#undef PR_AFTER_WRITEBACK_EXT
#endif

// Logs the write-set after acquiring the locks (TODO: it is the same in PR_AFTER_VAL_LOCKS_EXT)
#define PR_AFTER_WRITEBACK_EXT(args, i, addr, val) ({ \
	/* HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args->pr_args_ext; */ \
	/* SET_ON_LOG(addr); TODO: add write to BM */ \
}) \

#include "pr-stm-wrapper.cuh" // enables the granularity
#include "pr-stm-internal.cuh"

// --------------------
__constant__ __device__ unsigned PR_seed = 1234; // TODO: set seed

__global__ void setupKernel(void *args)
{
	curandState *state = (curandState*)args;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(PR_seed, id, 0, &state[id]);
}

__device__ unsigned PR_i_rand(pr_tx_args_dev_host_s args, unsigned n)
{
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	curandState *state = GPU_log->state;
	int id = PR_THREAD_IDX;
	int x = 0;
	curandState localState = state[id];
	x = curand(&localState);
	state[id] = localState;
	return x % n;
}
// --------------------

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

__constant__ __device__ long size;
__constant__ __device__ int  TransEachThread;
// __constant__ __device__ const int BANK_NB_TRANSFERS;
__constant__ __device__ int  hashNum;
__constant__ __device__ int  num_ways; // TODO: init

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
__device__ void random_Kernel(PR_txCallDefArgs, int *idx, curandState* state, int size, int is_intersection)
{
	int i, j;
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	int id = threadIdx.x+blockDim.x*blockIdx.x;

	// generates the target accounts for the transaction
	for (i = 0; i < BANK_NB_TRANSFERS; i++) {
		int m = 0;
		int is_intersect = is_intersection; //IS_INTERSECT_HIT(PR_rand(100000));

		// accounts must be different
		while (m < 1) {
			int randVal = PR_rand(INT_MAX);
			// TODO: the size becomes useless
			if (is_intersect) {
				GPU_log->onIntersect[id]++;
				idx[i] = INTERSECT_ACCESS(randVal, size);
			} else {
				idx[i] = GPU_ACCESS(randVal, size);
			}
			bool hasEqual = 0;

			// idx array is traveled to check repeated accesses
			for (j = 0; j < i; j++)	{
				if (idx[i] == idx[j]) {
					hasEqual = 1;
					break;
				}
			}
			if (hasEqual != 1) {
				// if it is not repeated goto i++ in the outer for loop
				m++; // break while (m < 1)
			}
		}
	}
	/*idx[0] = generate_kernel(state,100)%size;
	for (int i = 0; i < BANK_NB_TRANSFERS; i++)
	{
	idx[i] = (idx[0]+i)%size;
	}*/
}


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

	random_Kernel(PR_txCallArgs, idx, GPU_log->state, nbAccounts, is_intersection);	//get random index

  // TODO: it was TransEachThread * iterations
	while (i++ < TransEachThread) { // each thread need to commit x transactions
		PR_txBegin();

		// reads the accounts first, then mutates the locations
		for (j = 0; j < BANK_NB_TRANSFERS; j++)	{
			reads[j] = PR_read(accounts + idx[j]);
			if (pr_args.is_abort) break; // PR_txBegin is a simple while loop
		}

		if (pr_args.is_abort) { PR_txRestart(); } // optimization

		for (j = 0; j < BANK_NB_TRANSFERS / 2; j++) {
			target = j*2;
			nval = reads[target] - 1; // -money
			PR_write(accounts + idx[target], nval); //write changes to write set
			if (pr_args.is_abort) break;

			target = j*2+1;
			nval = reads[target] + 1; // +money
			PR_write(accounts + idx[target], nval); //write changes to write set
			if (pr_args.is_abort) break;
		}
		if (pr_args.is_abort) { PR_txRestart(); } // optimization
		PR_txCommit();
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

	HeTM_memcdTx_input_s *input = (HeTM_memcdTx_input_s*)args.inBuf;

	int goal = 0;

	idx[0] = input->tx_queue[id] * num_ways;

	for (int j = 1; j < num_ways; j++) {
		idx[j] = idx[0] + j;
	}

	goal =  input->tx_queue[id] & 0xffff;
	goal = (goal % num_ways);

	PR_txBegin();

	for (int j = 0; j <= goal; j++) {
		reads[j] = PR_read(input->accounts + idx[j]);
		if (pr_args.is_abort) break;
	}

	if (pr_args.is_abort) { PR_txRestart(); }

	PR_txCommit();

	input->output[id].key    = reads[goal];
	input->output[id].index  = idx[goal];
	input->ts_vec[idx[goal]] = input->clock_value;

	PR_exitKernel();
}

/*********************************
 *	writeKernelTransaction()
 *
 *  Main PR-STM transaction kernel
 **********************************/
__global__ void writeKernelTransaction (PR_globalKernelArgs)
{
	PR_enterKernel();

	int idx[NUMBER_WAYS];
	int reads[NUMBER_WAYS];
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int i = 0;
	int min_val = -1, min_pos=0;

	HeTM_memcdTx_input_s *input = (HeTM_memcdTx_input_s*)args.inBuf;

	int goal = 0;

	idx[0] = input->tx_queue[id] * num_ways;
	for (int j = 1; j < num_ways; j++) {
		idx[j] = idx[0] + j;
	}

	goal = input->tx_queue[id] & 0xffff;

	while (i < TransEachThread) { //each thread need to commit x transactions

		PR_txBegin();

		min_val = -1;
		min_pos=0;

		/*Search hash table for an empty spot or an entry to evict*/
		for (int j = 0; j < num_ways; j++) {
			reads[j] = PR_read(input->accounts + idx[j]);

			//Check if it is free or if it is the same value
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
		if (pr_args.is_abort) { PR_txRestart(); }
		PR_write(input->accounts + idx[min_pos], goal);
		PR_txCommit();

		input->output[id].key = goal;
		input->output[id].index = idx[min_pos];
		input->ts_vec[idx[min_pos]] = input->clock_value;
		i++;
	}

	PR_exitKernel();
}
// ----------------------



/****************************************************************************
 *	FUNCTIONS
/****************************************************************************/

extern "C"
cuda_config cuda_configInit(int size, int trans, int hash, int tx, int bl) {
	cuda_config c;

	c.size = size;
	c.TransEachThread = trans > 0 ? trans : ( DEFAULT_TransEachThread << 1) / BANK_NB_TRANSFERS;
	c.hashNum = hash > 0 ? hash : DEFAULT_hashNum;
	c.threadNum = tx > 0 ? tx : DEFAULT_threadNum;
	c.blockNum = bl > 0 ? bl : DEFAULT_blockNum;
	// c.BANK_NB_TRANSFERS = (BANK_NB_TRANSFERS > 1 && ((BANK_NB_TRANSFERS & 1) == 0)) ? BANK_NB_TRANSFERS : 2; // DEFAULT_BANK_NB_TRANSFERS

	return c;
}

extern "C"
cudaError_t cuda_configCpy(cuda_config c) {
	cudaError_t cudaStatus;
	int err = 1;

	while (err) {
		err=0;
		void * point1 = &c.size;
		cudaStatus = cudaMemcpyToSymbol(size, point1, sizeof(long), 0, cudaMemcpyHostToDevice );
		if ( cudaStatus  != cudaSuccess) {
			printf("cudaMemcpy to device failed for size!");
			continue;
		}
		void * point2 = &c.TransEachThread;
		cudaStatus = cudaMemcpyToSymbol(TransEachThread, point2, sizeof(int), 0, cudaMemcpyHostToDevice );
		if ( cudaStatus  != cudaSuccess) {
			printf("cudaMemcpy to device failed for TransEachThread!");
			continue;
		}
		void * point3 = &c.hashNum;
		cudaStatus = cudaMemcpyToSymbol(hashNum, point3, sizeof(int), 0, cudaMemcpyHostToDevice );
		if ( cudaStatus  != cudaSuccess) {
			printf("cudaMemcpy to device failed for hashNum!");
			continue;
		}
		// TODO: find a way of passing BANK_NB_TRANSFERS
		// void * point4 = &c.BANK_NB_TRANSFERS;
		// cudaStatus = cudaMemcpyToSymbol(BANK_NB_TRANSFERS, point4, sizeof(int),0, cudaMemcpyHostToDevice );
		// if ( cudaStatus  != cudaSuccess) {
		// 	printf("cudaMemcpy to device failed for BANK_NB_TRANSFERS!");
		// 	continue;
		// }
	}
	/*cudaMemcpyFromSymbol(&c.hashNum,"hashNum",sizeof(int));
	printf("hashNum: %d\n",c.hashNum);*/

	return cudaStatus;
};
