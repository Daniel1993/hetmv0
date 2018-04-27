#ifndef _BANK_KERNEL
#define _BANK_KERNEL

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_defines.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_timer.h"
#include <time.h>

// TODO: extend PR-STM here in order to have the logs...
#include "cmp_kernels.cuh"
#include "bank.h"
#include "tx_queue.cuh" // TODO: memcd

/****************************************************************************
 *	STRUCTURES
 ****************************************************************************/

 typedef struct {
	long 	size;
	int  	TransEachThread;
	int 	hashNum;
	int 	threadNum;
	int 	blockNum;
	int 	num_ways;
	float	hmult;
	int 	hprob;
} cuda_config;

/****************************************************************************
 *	KERNELS
 ****************************************************************************/

typedef struct HeTM_knl_bankTx_ {
	cuda_t *d;
  account_t *a;
} HeTM_knl_bankTx_s;

typedef struct HeTM_bankTx_ {
	HeTM_knl_bankTx_s knlArgs;
	stream_t *clbkArgs;
} HeTM_bankTx_s;

typedef struct HeTM_bankTx_input_ {
	PR_GRANULE_T *accounts;
  size_t nbAccounts;
	int is_intersection;
} HeTM_bankTx_input_s;

typedef struct HeTM_memcdTx_input_ {
	PR_GRANULE_T *accounts;  /* values in global memory */
  int *ts_vec;             /* timestamp array         */
  cuda_output_t *output;   /* produced results        */
  size_t nbAccounts;       /* size of the data-set    */
	int clock_value;         /* sync timestamp          */
	unsigned int *tx_queue;  /* transaction queue       */
} HeTM_memcdTx_input_s;

/*********************************
 *	bankTx()
 *
 *  Main PR-STM transaction kernel
 *  implementing TinySTM's bank micro-benchmark
 **********************************/
__global__ void bankTx(PR_globalKernelArgs);
// __global__ void setup_kernel(curandState *state, unsigned long seed);

__global__ void memcdWriteTx(PR_globalKernelArgs);

__global__
__launch_bounds__(1024, 1)
void memcdReadTx(PR_globalKernelArgs);

/****************************************************************************
 *	FUNCTIONS
 ****************************************************************************/

extern "C"
cuda_config cuda_configInit(long size, int trans, int hash, int tx, int bl, int hprob, float hmult);

extern "C"
void cuda_configCpy(cuda_config c);

extern "C"
cudaError_t cuda_configCpyMemcd(cuda_t *c);

#endif /* _BANK_KERNEL */
