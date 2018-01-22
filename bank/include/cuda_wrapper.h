#ifndef CUDA_WRAPPER_H_GUARD_
#define CUDA_WRAPPER_H_GUARD_

#include <cuda.h>
#include "log.h"
#include "cuda_defines.h"
#include "hetm-types.h"

#include <cuda_runtime.h>
// #include <curand.h>
// #include <curand_kernel.h>
// #include "helper_timer.h"

#include "pr-stm-wrapper.cuh"
#include "shared.h"

#define PADDING              0

// TODO: Number of comparisons sent to the GPU before restarting the kernel
#define NB_CMP_RETRIES       2

#define STM_LOG_BUFFER_SIZE  16 // multiplied by LOG_SIZE

/****************************************************************************
 *	STRUCTURES
 ****************************************************************************/

typedef enum {
	READ,   // Executa transações
	STREAM, // Inicia streams de cmp e Executa transações
	FINAL,  // Bloqueia para acabar as cmp
	END     // Bloqueia para acabar a sync (receber dados)
} cmp_status;

// TODO
// typedef struct HeTM_shared_ {
//
// } HeTM_shared_s;

typedef struct cuda_info {
	account_t *host_a;     //host values array
  account_t *dev_a;      //values array
	account_t *dev_b;      //device copy of host values array
	account_t *dev_bckp;   //device copy of host values array
	account_t *dev_zc;     //zero_copy array

	// TODO: not in use anymore
	int *dev_bm;				   //compressed log array
	int bm_size;

	// TODO: delete
	int *dev_LogR;			   //Read Log array
	int *dev_LogW;			   //Write Log array

	HeTM_CPULogEntry *host_log;  //Host log array
	int *dev_flag;         //Comparison flag
	void *devStates;       //randseed array

	int size;
	int threadNum;
	int blockNum;
	int TransEachThread;
	int set_percent;
	int num_ways;
	int clock;

	// TODO: this is memcd
  queue_t *q;
	unsigned int *gpu_queue;
} cuda_t;

typedef struct stream_info {
	int id;
	int count;
	cmp_status status;
	int isCmpDone;
	int isThreshold;
	int maxC;
	int isCudaError;
	cudaStream_t st;
	pthread_mutex_t mutex;
	HeTM_CPULogEntry *host_log;       //Host log array
	HeTM_CPULogEntry *stream_log;
} stream_t;

/****************************************************************************
 *	MACROS
 ****************************************************************************/

#define EXPECTED_COMMITS_GPU(d)			d->TransEachThread*d->blockNum*d->threadNum

/****************************************************************************
 *	FUNCTIONS
 ****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

// TODO: put GRANULE_T or account_t
cuda_t * jobWithCuda_init(account_t *base, int size, int trans, int hash, int tx, int bl);
void jobWithCuda_initMemcd(cuda_t *cd, int ways, int wr, int sr); // memcd

stream_t * jobWithCuda_initStream(cuda_t *d, int id, int count);

int jobWithCuda_run(cuda_t *d, account_t *a);

void jobWithCuda_wait();

int jobWithCuda_checkStream(cuda_t *d, stream_t *st, size_t size_stm);

account_t* jobWithCuda_swap(cuda_t *d);

int jobWithCuda_cpyDatasetToGPU(cuda_t *d, account_t *b);

int jobWithCuda_resetGPUSTMState(cuda_t *d);

int jobWithCuda_hupd(cuda_t *d, account_t *a, int *bm);

void jobWithCuda_getStats(cuda_t *d, long *ab, long *com);

void jobWithCuda_exit(cuda_t * d);

long* CudaMallocWrapper(size_t s);

long* CudaZeroCopyWrapper(long *p);

void CudaFreeWrapper(void *p);

int jobWithCuda_bm(cuda_t *d, int *bm);

void jobWithCuda_threadCheck(cuda_t *d, stream_t *s);

// return the current CPU log node (i.e., that did not fit into the buffer), or NULL
HeTM_CPULogNode_t* jobWithCuda_mergeLog(cudaStream_t stream, HeTM_CPULogNode_t *t, size_t *size, int isBlock);

// returns if a conflict was found or not
int jobWithCuda_checkStreamFinal(cuda_t *d, stream_t * st, int n);

void jobWithCuda_exitStream(stream_t * s);

void jobWithCuda_backup(cuda_t * d);

void jobWithCuda_backupRestore(cuda_t * d);

// TODO: memcd
unsigned int* readQueue(queue_t *q, int size, int queue_id);
void queue_Delete(queue_t *q);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_WRAPPER_H_GUARD_ */
