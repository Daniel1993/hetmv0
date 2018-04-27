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

#include "memman.h"
#include "knlman.h"

#include "setupKernels.cuh"

extern "C" {
#include "cuda_wrapper.h"
}

#include "bankKernel.cuh"

//#define	DEBUG_CUDA
//#define	DEBUG_CUDA2

//Support for the lazylog implementation
// moved to cmp_kernels.cuh
// #define EXPLICIT_LOG_BLOCK     (TransEachThread * BANK_NB_TRANSFERS)
// #define EXPLICIT_LOG_SIZE      (blockNum * threadNum * EXPLICIT_LOG_BLOCK)	//size of the lazy lock

//Support for the compressed log implementation
#define readMask         0b01	//Set entry to write
#define writeMask        0b10	//Set entry to read

//versions in global memory	10 bits(version),0 bits (padding),20 bits(owner threadID),1 bit(LOCKED),1 bit(pre-locked)
#define	offVers	22
#define offOwn	2
#define offLock	1

#define finalIdx          (threadIdx.x+blockIdx.x*blockDim.x)
#define newLock(x,y,z)    ( ((x) << offVers) | ((y) << offOwn) | (z))

#define uint_64				int

#define CHECK_ERROR_CONTINUE(cuda_call) \
  cudaStatus = cuda_call; \
  if (cudaStatus != cudaSuccess) { \
    printf("Error " #cuda_call " \n" __FILE__ ":%i\n   > %s\n", \
      __LINE__, cudaGetErrorString(cudaStatus)); \
    /*goto Error;*/ \
    continue; \
  } \
//

void CUDART_CB checkCallback(cudaStream_t stream, cudaError_t status, void *data)
{
  stream_t *st = (stream_t*)data;
  int i;

  if(status != cudaSuccess) {
    st->count = st->maxC;
    st->isCudaError = -1;
    printf("Comparison kernel crashed. Error code is: %s.\n", cudaGetErrorString(status));
  }

  // clears the ad-hoc memory
  memman_ad_hoc_free((void*)stream);

  pthread_mutex_lock(&st->mutex);
  i = st->count;
  if(i>=st->maxC)
    printf("BUG: st->count = %d\n",i);
  st->count++;
  pthread_mutex_unlock(&st->mutex);
}


/* ################################################################################################# *
 * HOST CODE
 * ################################################################################################# */

/****************************************
 *	jobWithCuda_init(size,hostLogMax)
 *
 *	Description:	Initialize the GPU, by allocating all necessary memory,
 *					transferring static data and running the setup kernel.
 *
 *	Args:
 *		int size		: Size (in integers) to allocate for working set data
 *		long ** accounts: Pointer to host array pointer
 *      int hostLogMax	: Maximum number of entries the host transaction log can contain
 *		long * b:		:(Optional) Host log address, for use with zero copy
 *
 *	Returns:
 *		cuda_t: 	Custom structure containing all essential CUDA pointers/data
 *					or null in case of failure
 *
 ****************************************/
 // TODO: put GRANULE_T or account_t
extern "C"
cuda_t * jobWithCuda_init(account_t *accounts, int size, int trans, int hash, int tx, int bl)
{
  //int *a = (int *)malloc(size * sizeof(int));
  size_t accountsSize, cuRandSize, nbAccounts, nbGPUThreads;
  int *dev_bm = 0;         //compressed log array
  int *dev_LogR = 0, *dev_LogW = 0;	//Log array
  cuda_config cuda_info;    //Cuda config info
  bool err = 1;	// Replacement for the goto (TODO: why loop on error? Just exit!)
  cuda_t *c_data;
  cudaError_t cudaStatus;
  curandState *devStates;	//randseed array

  //cudaProfilerStop();							//Stop unnecessary  profiling

  cuda_info = cuda_configInit(size, trans, hash, tx, bl);

  PR_init(); // inits PR-STM mutex array

  nbAccounts   = cuda_info.size;
  nbGPUThreads = cuda_info.blockNum * cuda_info.threadNum;
  cuRandSize   = nbGPUThreads * sizeof(curandState);
  accountsSize = nbAccounts * sizeof(account_t);

  queue_t * q = NULL;

  // Init check Tx kernel
  // TODO: init EXPLICIT
  HeTM_setup_checkTxCompressed();
  HeTM_setup_checkTxExplicit();
  HeTM_setup_finalTxLog2();
  HeTM_setup_bankTx();

  while (err) {
    err = false;

    // Choose which GPU to run on, change this on a multi-GPU system.
    CHECK_ERROR_CONTINUE(cudaSetDevice(DEVICE_ID));

    CHECK_ERROR_CONTINUE(cudaMalloc((void **)&devStates, cuRandSize));
    setupKernel <<< cuda_info.blockNum, cuda_info.threadNum >>>(devStates); /* setups PR_rand on GPU */
  	cudaThreadSynchronize();

    memman_alloc_gpu("HeTM_accounts_a", accountsSize, accounts, 0);
    memman_zero_gpu(NULL); // copy to GPU
    memman_alloc_gpu("HeTM_accounts_b", accountsSize, accounts, 0);
    memman_zero_gpu(NULL);
    memman_alloc_gpu("HeTM_accounts_bckp", accountsSize, accounts, 0);


    // TODO
    memman_alloc_gpu("HeTM_accounts_ts", accountsSize, accounts, 0);
    memman_zero_gpu(NULL);

    memman_alloc_dual("HeTM_cpu_versions", nbAccounts * sizeof(long), 0);
    memman_zero_gpu(NULL);

    // TODO: use alloc_gpu
    // memman_alloc_dual("HeTM_CPU_wset_log", LOG_SIZE * sizeof(HeTM_CPULogEntry), 0);
    // CHECK_ERROR_CONTINUE(cudaMalloc((void **)&stm_log, LOG_SIZE * sizeof(HeTM_CPULogEntry)));

    //Comparison flag allocation
    memman_alloc_dual("HeTM_flag_inter_conflict", sizeof(int), 0);

    CHECK_ERROR_CONTINUE(cudaMalloc((void **)&dev_LogR, nbAccounts * sizeof(int)));
    CHECK_ERROR_CONTINUE(cudaMemset(dev_LogR, 0, nbAccounts * sizeof(int)));

    CHECK_ERROR_CONTINUE(cudaMalloc((void **)&dev_bm, nbAccounts * sizeof(long)));

    // 1 bit per accounts, i.e., allocates ceilling(nbAccounts/ bitsInByte)
#if HETM_CMP_TYPE == HETM_CMP_EXPLICIT
    memman_alloc_dual("HeTM_dev_rset", EXPLICIT_LOG_SIZE(cuda_info.blockNum, cuda_info.threadNum)*sizeof(int), 0);
    memman_zero_gpu(NULL);
#elif HETM_CMP_TYPE == HETM_CMP_COMPRESSED
    memman_alloc_dual("HeTM_dev_rset", nbAccounts / 8 + 1, 0);
    memman_zero_gpu(NULL);
#else
    // error or disabled
#endif

    CHECK_ERROR_CONTINUE(cuda_configCpy(cuda_info));

    time_t t;
    time(&t);
  }

  if (cudaStatus != cudaSuccess) {
    printf("\nSetup: Error code is: %s\n", cudaGetErrorString(cudaStatus));
    memman_select("HeTM_flag_inter_conflict");
    memman_free_dual();
    cudaFree(devStates);
    // memman_select("HeTM_CPU_wset_log");
    // memman_free_dual(); // TODO: use GPU only
    // cudaFree(stm_log);
    cudaFree(dev_LogR);
    cudaFree(dev_LogW);
    cudaFree(dev_bm);
    memman_select("Stats_OnIntersect");
    memman_free_dual();
    memman_select("HeTM_dev_rset");
    memman_free_dual();
    memman_select("HeTM_cpu_versions");
    memman_free_dual();
    memman_select("HeTM_accounts_a");
    memman_free_gpu();
    memman_select("HeTM_accounts_b");
    memman_free_gpu();
    PR_teardown();
    c_data = NULL;
  } else {
    //Save cuda pointers
    c_data = (cuda_t *)malloc( sizeof(cuda_t) );

    c_data->host_a = accounts;
    memman_select("HeTM_accounts_a");
    c_data->dev_a = (account_t*)memman_get_gpu(NULL);
    memman_select("HeTM_accounts_b");
    c_data->dev_b = (account_t*)memman_get_gpu(NULL);
    memman_select("HeTM_accounts_bckp");
    c_data->dev_bckp = (account_t*)memman_get_gpu(NULL);
    c_data->dev_LogR = dev_LogR;
    c_data->dev_LogW = dev_LogW;
    c_data->devStates = (void*)devStates;
    c_data->size = cuda_info.size;
    c_data->dev_bm = dev_bm; // TODO
    // memman_select("HeTM_CPU_wset_log");
    // c_data->host_log = (HeTM_CPULogEntry*)memman_get_gpu(NULL);
    memman_select("HeTM_flag_inter_conflict");
    c_data->dev_flag = (int*)memman_get_gpu(NULL);
    c_data->dev_zc = NULL;
    c_data->bm_size = (cuda_info.size>>BM_HASH)+1;
    c_data->threadNum = cuda_info.threadNum;
    c_data->blockNum = cuda_info.blockNum;
    c_data->TransEachThread=cuda_info.TransEachThread;
  }

  return c_data;
}

/****************************************
 *	jobWithCuda_initStream()
 *
 *	Description:
 *
 *
 *	Args:

 *
 *	Returns:
 *		stream_t:
 *
 *
 ****************************************/
extern "C"
stream_t *jobWithCuda_initStream(cuda_t *d, int id, int count) {
  stream_t *stream_data = NULL;
  // cudaError_t cudaStatus;
  // HeTM_CPULogEntry * host_point = 0, * stream_point = 0;
  // size_t logSize = LOG_SIZE * sizeof(HeTM_CPULogEntry);
  int n = 0;

  bool err = 1;	//Replacement for the goto

  // per thread
  stream_data = (stream_t*)malloc( sizeof(stream_t) );

  // Each CPU thread allocates some space in the GPU to dump
  // its thread local STM log.
  // memman_alloc_gpu("HeTM_CPU_wset_log", logSize, NULL, MEMMAN_THRLC);
  // stream_data->host_log = (HeTM_CPULogEntry*)memman_get_gpu(NULL);

  //Save pointers
  stream_data->st = NULL; //(cudaStream_t *)malloc( sizeof(cudaStream_t)*count );
  stream_data->isCudaError = 0;
  stream_data->id = id;
  stream_data->maxC = count;
  stream_data->count = 0;
  pthread_mutex_init(&stream_data->mutex, NULL);

  for (n = 0; n < count; n++) {
    // cudaStreamCreate(&stream_data->st[n]);
    knlman_add_stream();
  }

  return stream_data;
}

/****************************************
 *	jobWithCuda_run(d,a)
 *
 *	Description:	Update working set data and run transaction kernel.
 *					Failures are only detected on subsequent calls to jobWithCuda_wait()
 *
 *	Args:
 *		cuda_t * d		: Custom structure containing all essential transaction kernel CUDA pointers/data
 *      long * a		: Working set data
 *
 *	Returns:
 *		int:		1 in case of success, 0 otherwise
 *
 ****************************************/
extern "C"
int jobWithCuda_run(cuda_t *d, account_t *a) {
  bool err = 1;
  cudaError_t cudaStatus;

  while (err) {
    err = 0;

    CHECK_ERROR_CONTINUE(cudaSetDevice(DEVICE_ID));

    knlman_select("HeTM_bankTx");
    knlman_set_nb_blocks(d->blockNum, 1, 1);
    knlman_set_nb_threads(d->threadNum, 1, 1);

    HeTM_bankTx_s bankTx_args = {
      .knlArgs = {
        .d = d,
        .a = a,
      },
      .clbkArgs = NULL
    };
    knlman_set_entry_object(&bankTx_args);
    knlman_run();

    //Check for errors
    cudaStatus = cudaGetLastError();
  }

  if (cudaStatus != cudaSuccess) {
    printf("\nTransaction kernel launch failed. Error code: %s.\n", cudaGetErrorString(cudaStatus));
    return 0;
  }

  return 1;
}

/****************************************
 *	jobWithCuda_wait()
 *
 *	Description:	Wait for cuda execution to conclude.
 *
 *	Args: 			(none)
 *
 *	Returns:		(none)
 *
 ****************************************/
extern "C"
void jobWithCuda_wait(){

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaError_t cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize returned error code: %d\n.", cudaStatus);
  }
}

/****************************************
 *	jobWithCuda_checkStream(d,vec,size_stm,id,streamtime,trf)
 *
 *	Description:	Copy host log data to device, configure and launch comparison kernel
 *					to detect conflicts between host and device transactions using the
 *					designated stream.
 *
 *	Args:
 *		cuda_t d	: Custom structure containing all essential CUDA pointers/data
 *		HeTM_CPULogEntry * vec	: Array containing host log
 *		int size_stm: Number of valid entries in the host log array
 *		float * time: (Optional) Pointer to store duration of comparison kernel
 *		float * trf : (Optional) Pointer to store duration of comparison kernel
 *		int id		: Id of the launching thread
 *		cudaStream_t stream: Cuda stream to launch to
 *
 *	Returns:
 *		int:		Result of the comparison: 0 if comparison detected no conflicts,
 *					1 otherwise or -1 in case of error.
 *
 ****************************************/
extern "C"
int jobWithCuda_checkStream(cuda_t d, stream_t *st, HeTM_CPULogEntry *vec, int size_stm, int n) {
  cudaError_t cudaStatus;
  bool err = 1;
  int ret = 1; //Return value
  HeTM_CPULogEntry * streamHLog; // on device
  cudaStream_t stream;
  size_t logSize = size_stm * sizeof(HeTM_CPULogEntry);

  if (size_stm==0) {
    __sync_add_and_fetch(&st->count, 1);
    return 0;
  }
  //printf("Comparison size: %d\n", size_stm);
  while (err) {
    err=0;

    knlman_choose_stream(n);
    // knlman_choose_next_stream();
    // knlman_sync_stream(); // synchronize threads
    stream = (cudaStream_t)knlman_get_current_stream();

    streamHLog = (HeTM_CPULogEntry*)memman_ad_hoc_alloc((void*)stream, (void*)vec, logSize);
    memman_ad_hoc_cpy((void*)stream);

    memman_select("HeTM_dev_rset");
    void *rset = memman_get_gpu(NULL);
    memman_select("HeTM_cpu_versions");
    long *vers  = (long*)memman_get_gpu(NULL);

#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
    // -----------------------------------------------
    //Calc number of blocks
    int bo = (size_stm + 31) / (32);
    dim3 blocksCheck(bo); // partition the stm_log by the different blocks
    dim3 threadsPerBlock(32); // each block has 32 threads

    knlman_select("HeTM_checkTxCompressed");
    knlman_set_nb_blocks(bo, 1, 1);
    knlman_set_nb_threads(32, 1, 1);

    // Memory region of the entry object
    HeTM_checkTxCompressed_s checkTxCompressed_args = {
      .knlArgs = {
        .dev_flag = d.dev_flag,
        .stm_log  = streamHLog,
        .size_stm = size_stm,
        .size_log = d.size,
        .mutex    = PR_lockTableDev,
        .devLogR  = (int*)rset,
        .a        = d.dev_a,
        .b        = d.dev_b,
        .vers     = vers,
      },
      .clbkArgs = st
    };
    knlman_set_entry_object(&checkTxCompressed_args);

    knlman_run();
    // -----------------------------------------------
#elif HETM_CMP_TYPE == HETM_CMP_EXPLICIT
    int xThrs = CMP_EXPLICIT_THRS_PER_RSET / CMP_EXPLICIT_THRS_PER_WSET;
    int yThrs = CMP_EXPLICIT_THRS_PER_WSET;

    int hasRemainderXThrs = EXPLICIT_LOG_SIZE(d.blockNum, d.threadNum) % xThrs;

    int xBlocks = EXPLICIT_LOG_SIZE(d.blockNum, d.threadNum) / xThrs;
    int yBlocks = size_stm / CMP_EXPLICIT_THRS_PER_WSET;

    int hasRemainderYBlocks = size_stm % CMP_EXPLICIT_THRS_PER_WSET;

    if (hasRemainderXThrs) xBlocks++;
    if (hasRemainderYBlocks) yBlocks++;

    knlman_select("HeTM_checkTxExplicit");
    knlman_set_nb_blocks(xBlocks, yBlocks, 1);
    knlman_set_nb_threads(xThrs, yThrs, 1);

    HeTM_checkTxExplicit_s checkTxExplicit_args = {
      .knlArgs = {
        .dev_flag = d.dev_flag,
        .stm_log  = streamHLog,
        .size_stm = size_stm,
        .size_logR= EXPLICIT_LOG_SIZE(d.blockNum, d.threadNum),
        .devLogR  = (int*)rset,
        .mutex    = PR_lockTableDev,
        .a        = d.dev_a,
        .b        = d.dev_b,
        .vers     = vers,
      },
      .clbkArgs = st
    };
    knlman_set_entry_object(&checkTxExplicit_args);

    knlman_run();
#else
  printf("Error! no compare method selected!\n");
#endif

    CHECK_ERROR_CONTINUE(cudaGetLastError());
  }

  // Check for errors
  cudaStatus = cudaGetLastError(); //synchronize threads
  if (cudaStatus != cudaSuccess) {
    printf("Stream CMP[%d]>> Error code is: %s.\n", st->id, cudaGetErrorString(cudaStatus));
    ret = -1;
  }

  return ret;
}


/****************************************
 *	jobWithCuda_swap(d,a)
 *
 *	Description:	Overwrites devices working set with the hosts
 *
 *	Args:
 *		cuda_t  d		: Custom structure containing all essential transaction kernel CUDA pointers/data
 *
 *	Returns:
 *		long *:			0 in case of failure, a pointer otherwise
 *
 ****************************************/
extern "C"
account_t* jobWithCuda_swap(cuda_t d){
  return d.host_a;
}

/****************************************
 *	jobWithCuda_dupd(d,b)
 *
 *	Description:	Transfers host values to device.
 *
 *	Args:
 *		cuda_t d		: Custom structure containing all essential transaction kernel CUDA pointers/data
 *      long * b		: Host working set data
 *
 *	Returns:
 *		int:		1 in case of success, 0 otherwise
 *
 ****************************************/
extern "C"
int jobWithCuda_dupd(cuda_t d, account_t *b)
{
  cudaError_t cudaStatus;

  // Transfer comparison flag to device
  memman_select("HeTM_flag_inter_conflict");
  memman_zero_gpu(NULL); //copy new vector from CPU to GPU
  // cudaStatus = cudaMemset(d.dev_flag, 0, sizeof(int));
  // if (cudaStatus != cudaSuccess) {
  //   printf("cudaMemcpy to device failed for dev_flag!\n");
  //   return 0;
  // }

  // Transfer comparison flag to device
  // cudaStatus = cudaMemset(PR_lockTableDev, 0, PR_LOCK_TABLE_SIZE*sizeof(int));	//copy new vector from CPU to GPU
  if (cudaStatus != cudaSuccess) {
    printf("cudaMemcpy to device failed for PR_lockTableDev!\n");
    return 0;
  }

  return 1;
}

/****************************************
 *	jobWithCuda_hupd(d,vec,size_stm,time,ab.com)
 *
 *	Description:	Copy results produced by device to host
 *
 *	Args:
 *		cuda_t * d	: Custom structure containing all essential CUDA pointers/data
 *		long * a	: Device updated working set
 *		int * bm	: Bitmap indicating updated memory positions
 *
 *	Returns:
 *		int:		Result of the comparison: 0 if comparison detected no conflicts,
 *					1 otherwise or in case of error.
 *
 ****************************************/
extern "C"
int jobWithCuda_hupd(cuda_t *d, account_t *a, int *bm)
{
  // TODO: check what bm is doing
  cudaError_t cudaStatus;
  int err = 1;

  while(err) {
    err = 0;

    CHECK_ERROR_CONTINUE(cudaThreadSynchronize());

    // TODO: manter este
    //Transfer bitmap
    if (bm == NULL) {
      //Transfer data
      CHECK_ERROR_CONTINUE(cudaMemcpy(a, d->dev_a, d->size*sizeof(long), cudaMemcpyDeviceToHost)); //copy results from GPU to CPU
    } else {
      int i,j;
      for(i = 0; i< d->bm_size-1; i++) {
        if(bm[i]==1) {
          j=i<<BM_HASH;
          cudaStatus = cudaMemcpyAsync(&a[j], &d->dev_a[j], BM_HASH_SIZE*sizeof(long), cudaMemcpyDeviceToHost, 0);	//copy compressed log from GPU to CPU
          if (cudaStatus != cudaSuccess) {
            printf("cudaMemcpy failed for bm!");
            break;
          }
        }
      }
      j=i<<BM_HASH;
      if(bm[i]==1){
        CHECK_ERROR_CONTINUE(cudaMemcpyAsync(&a[j], &d->dev_a[j], (d->size - j)*sizeof(long), cudaMemcpyDeviceToHost, 0));	//copy compressed log from GPU to CPU
      }
    }
  }

  cudaStatus = cudaThreadSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("\nCpyBack: Error code is: %s.\n", cudaGetErrorString(cudaStatus));
    //printf("\nError code is: %s\n", cudaGetErrorString(cudaStatus));
    return 0;
  }

  //These will be swaped again on kernel launch, need to fix this (TODO
  account_t *buff = d->dev_a;
  d->dev_a = d->dev_b;
  d->dev_b = buff;

  return 1;
}

/****************************************
 *	jobWithCuda_getStats(cd,ab,com)
 *
 *	Description:	Get cuda stats
 *
 *	Args:
 *		cuda_t * d	: Custom structure containing all essential CUDA pointers/data
 *		int * ab	: (Optional) Pointer to store tx kernel abort counter
 *		int * com	: (Optional) Pointer to store tx kernel commit counter
 *
 *	Returns:
 *		(None)
 *
 ****************************************/
extern "C"
void jobWithCuda_getStats(cuda_t d, long *ab, long *com) {
  cudaError_t cudaStatus;
  int err = 1;

  while(err) {
    err = 0;

    CHECK_ERROR_CONTINUE(cudaThreadSynchronize());

    //Transfer aborts
    // TODO: WTF is this !!!!
    if (ab != NULL) {
      *ab = PR_nbAborts;
    }

    //Transfer commits
    if (com != NULL) {
      *com = PR_nbCommits;
    }
  }

  if (cudaStatus != cudaSuccess) {
    printf("\nStats: Error code is: %s.\n", cudaGetErrorString(cudaStatus));
    return;
  }
}

/****************************************
 *	jobWithCuda_exit(d)
 *
 *	Description:	Finish Cuda execution, free device memory and reset device.
 *
 *	Args:
 *		cuda_t d	: Custom structure containing all essential CUDA pointers/data
 *
 *	Returns:		(none)
 *
 ****************************************/
extern "C"
void jobWithCuda_exit(cuda_t * d)
{
  cudaError_t cudaStatus;

  cudaStatus = cudaSetDevice(DEVICE_ID);
  if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
  }

  if(d != NULL) {
    memman_select("HeTM_dev_rset");
    memman_free_dual();
    memman_select("HeTM_accounts_a");
    memman_free_gpu();
    memman_select("HeTM_accounts_b");
    memman_free_gpu();
    memman_select("HeTM_accounts_bckp");
    memman_free_gpu();
    // TODO:
    // memman_select("HeTM_CPU_wset_log");
    // memman_free_gpu();
    memman_select("HeTM_flag_inter_conflict");
    memman_free_dual();

    // cudaFree(d->dev_a);
    // cudaFree(d->dev_b);
    cudaFree(d->dev_bm);
    cudaFree(d->devStates);
    // cudaFree(d->host_log);
    // cudaFree(d->dev_flag);
    cudaFree(d->dev_LogW);
    cudaFree(d->dev_LogR);
    PR_teardown();
  }

  HeTM_teardown_checkTxCompressed();
  HeTM_teardown_bankTx();
  HeTM_teardown_finalTxLog2();
  HeTM_teardown_checkTxExplicit();

  // cudaDeviceReset(); // This is crashing on CUDA 9.0

  return;
}

/****************************************
 *	CudaMallocWrapper(s,type)
 *
 *	Description:	CudaHostAlloc wrapper function
 *
 *	Args:
 *		size_t s	: Size of the memory to allocate
 *
 *	Returns:		Pointer to allocd memory
 *
 ****************************************/
extern "C"
long * CudaMallocWrapper(size_t s) {
  long * ret = 0;

  cudaError_t cudaStatus = cudaHostAlloc((long **)&ret, s, cudaHostAllocPortable);
  if (cudaStatus != cudaSuccess) {
    printf("cudaHostAlloc returned error code: %d.\n", cudaStatus);
    return 0;
  }
/*#else
  ret = (long *)malloc(s);*/

  return ret;
}

/****************************************
 *	CudaZeroCopyWrapper(p)
 *
 *	Description:	cudaHostGetDevicePointer wrapper function
 *
 *	Args:
 *		void *p		: Pointer to memory to be transfered
 *
 *	Returns:		Pointer to device allocd memory
 *
 ****************************************/
extern "C"
long * CudaZeroCopyWrapper(long * p) {
  long * ret = 0;

  ret=p;
  return ret;
}

/****************************************
 *	CudaFreeWrapper(*p)
 *
 *	Description:	CudaFreeAlloc wrapper function
 *
 *	Args:
 *		void *p		: Pointer to memory to be free'd
 *
 *	Returns:		(Nothing)
 *
 ****************************************/
extern "C"
void CudaFreeWrapper(void * p) {

//#if ZERO_CPY==1
  cudaFreeHost(p);

  return;
}

extern "C"
int jobWithCuda_bm(cuda_t d, int *bm)
{
  cudaError_t cudaStatus;
  int err=1;

  while(err) {
    err = 0;

    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
      printf("jobWithCuda_bm() detected previous error!");
      //goto Error;
      continue;
    }

    knlman_select("HeTM_finalTxLog2");

    /* Define Kernel Size */
    knlman_set_nb_blocks((d.size+1023)/(1024), 1, 1);
    knlman_set_nb_threads(1024, 1, 1);

    HeTM_knl_finalTxLog2_s finalTxLog2_args = {
      .global_bitmap = d.dev_bm,
      .size          = d.size,
      .devLogR       = d.dev_LogR
    };
    knlman_set_entry_object(&finalTxLog2_args);
    knlman_run();
    cudaThreadSynchronize();

    //Transfer result
    cudaStatus = cudaMemcpy(bm, d.dev_bm, d.bm_size*sizeof(int), cudaMemcpyDeviceToHost);	//copy abortcounts
    if (cudaStatus != cudaSuccess) {
      printf("cudaMemcpy failed for bm!");
      continue;
    }

  }

  if (cudaStatus != cudaSuccess) {
    printf("\nBM: Error code is: %s.\n", cudaGetErrorString(cudaStatus));
    //printf("\nError code is: %s\n", cudaGetErrorString(cudaStatus));
    return 0;
  }
  return 1;
}

extern "C"
int jobWithCuda_checkStreamFinal(cuda_t d, stream_t *st, int n)
{
  cudaError_t cudaStatus;
  int *isConflict;

  st->count = 0;
  __sync_synchronize();

  memman_select("HeTM_flag_inter_conflict");
  memman_cpy_to_cpu(NULL); /* returns whether the comparison was successful */
  isConflict = (int*)memman_get_cpu(NULL);

  //Synchronize
  cudaStatus = cudaStreamSynchronize(0);
  if (cudaStatus != cudaSuccess) {
    printf("\nFinal Stream: Error code is: %s.\n", cudaGetErrorString(cudaStatus));
    return 0;
  }

  return *isConflict;
}

extern "C"
void jobWithCuda_exitStream(stream_t * s)
{
  // Destroy streams
  knlman_destroy_streams();

  knlman_destroy_thread();

  // Free everything else
  // memman_select("HeTM_CPU_wset_log");
  // memman_free_gpu();

  pthread_mutex_destroy(&s->mutex);
  free(s);

  return;
}

/****************************************
 *	jobWithCuda_backup(d)
 *
 *	Description:	Backup GPU produced results
 *
 *	Args:
 *		cuda_t d	: Custom structure containing all essential CUDA pointers/data
 *
 *	Returns:		(none)
 *
 ****************************************/
extern "C"
void jobWithCuda_backup(cuda_t * d) {

  cudaError_t cudaStatus = cudaMemcpy(d->dev_bckp, d->dev_a, d->size * sizeof(account_t), cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    printf("Backup: cudaMemcpy failed for dev_bckp. Error code %d: %s.\n", cudaStatus, cudaGetErrorString(cudaStatus));
  }
}

/****************************************
 *	jobWithCuda_backupRestore(d)
 *
 *	Description:	Restore GPU backup of produced results
 *
 *	Args:
 *		cuda_t d	: Custom structure containing all essential CUDA pointers/data
 *
 *	Returns:		(none)
 *
 ****************************************/
extern "C"
void jobWithCuda_backupRestore(cuda_t * d) {

  cudaError_t cudaStatus = cudaMemcpy(d->dev_a, d->dev_bckp, d->size * sizeof(account_t), cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    printf("BRestore: cudaMemcpy failed for dev_a. Error code %d: %s.\n", cudaStatus, cudaGetErrorString(cudaStatus));
  }
}
