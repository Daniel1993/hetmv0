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
#include "pr-stm-wrapper.cuh"

extern "C" {
#include "cuda_wrapper.h"
}

#include "bankKernel.cuh"

//#define	DEBUG_CUDA
//#define	DEBUG_CUDA2

#define ARCH             FERMI
#define DEVICE_ID        0

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

  if(status != cudaSuccess) {
    st->isCudaError = -1;
    printf("Comparison kernel crashed. Error code is: %s.\n", cudaGetErrorString(status));
  }

  st->isCmpDone = 1;
  __sync_synchronize();
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
  size_t accountsSize, cuRandSize, nbAccounts, nbGPUThreads, sizeCPUBuf;
  size_t sizeGPULog;
  cuda_config cuda_info;    //Cuda config info
  bool err = 1;	// Replacement for the goto (TODO: why loop on error? Just exit!)
  cuda_t *c_data;
  cudaError_t cudaStatus;
  curandState *devStates;	//randseed array

  //cudaProfilerStop();							//Stop unnecessary  profiling

  cuda_info = cuda_configInit(size, trans, hash, tx, bl);

  nbAccounts   = cuda_info.size;
  nbGPUThreads = cuda_info.blockNum * cuda_info.threadNum;
  cuRandSize   = nbGPUThreads * sizeof(curandState);
  accountsSize = nbAccounts * sizeof(account_t);
  sizeCPUBuf   = STM_LOG_BUFFER_SIZE*LOG_SIZE*sizeof(HeTM_CPULogEntry);

  while (err) {
    err = false;

    // Choose which GPU to run on, change this on a multi-GPU system.
    CHECK_ERROR_CONTINUE(cudaSetDevice(DEVICE_ID));

    // ----------------
    PR_init(); // inits PR-STM mutex array
    // ----------------

    // Init check Tx kernel
    // TODO: init EXPLICIT
    HeTM_setup_checkTxCompressed();
    HeTM_setup_checkTxExplicit();
    HeTM_setup_finalTxLog2();
    HeTM_setup_bankTx(cuda_info.blockNum, cuda_info.threadNum);

    CHECK_ERROR_CONTINUE(cudaMalloc((void **)&devStates, cuRandSize));
    setupKernel<<< cuda_info.blockNum, cuda_info.threadNum >>>(devStates); /* setups PR_rand on GPU */
  	cudaThreadSynchronize();

    memman_alloc_gpu("HeTM_accounts_a", accountsSize, accounts, 0);
    memman_zero_gpu(NULL); // copy to GPU
    memman_alloc_gpu("HeTM_accounts_b", accountsSize, accounts, 0);
    memman_zero_gpu(NULL);
    memman_alloc_gpu("HeTM_accounts_bckp", accountsSize, accounts, 0);

    // statistics on how often the GPU accessed the shared region in the dataset
    memman_alloc_dual("Stats_OnIntersect", nbGPUThreads * sizeof(int), 0);
    memman_zero_gpu(NULL);

    // the current STM version of the dataset granule
    memman_alloc_dual("HeTM_cpu_versions", nbAccounts * sizeof(long), 0);
    memman_zero_gpu(NULL);

    // buffer where the STM log is copy into, no host-side buffer needed
    memman_alloc_gpu("HeTM_CPU_wset_buffer", sizeCPUBuf, NULL, 0);
    memman_zero_gpu(NULL);

    //Comparison flag allocation
    memman_alloc_dual("HeTM_flag_inter_conflict", sizeof(int), 0);
    memman_zero_gpu(NULL);

#if CMP_TYPE == CMP_EXPLICIT
    // TODO: removed 1 bit per accounts, i.e., allocates ceilling(nbAccounts/ bitsInByte),
    // because it needs the use of atomicOr (not big changes in performance though)
    sizeGPULog = EXPLICIT_LOG_SIZE(cuda_info.blockNum, cuda_info.threadNum)*sizeof(int);
#elif CMP_TYPE == CMP_COMPRESSED
    // Using 1 byte per account, set the byte to 1 to say
    // that GPU read it
    sizeGPULog = nbAccounts;
#else
    // error or disabled
#endif

    // allocs the GPU log depending on the scheme
    memman_alloc_dual("HeTM_dev_rset", sizeGPULog, 0);
    memman_zero_gpu(NULL);

    CHECK_ERROR_CONTINUE(cuda_configCpy(cuda_info));
    long bpt = (long) ((void*)&accounts[0]);
    CHECK_ERROR_CONTINUE(copyPointer(bpt));

    time_t t;
    time(&t);
  }

  if (cudaStatus != cudaSuccess) {
    printf("\nSetup: Error code is: %s\n", cudaGetErrorString(cudaStatus));
    memman_select("HeTM_flag_inter_conflict");
    memman_free_dual();
    cudaFree(devStates);
    memman_select("Stats_OnIntersect");
    memman_free_dual();
    memman_select("HeTM_CPU_wset_buffer");
    memman_free_gpu();
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
    // TODO: Do we really need the pointers in this struct?

    //Save cuda pointers
    c_data = (cuda_t *)malloc( sizeof(cuda_t) );

    c_data->host_a = accounts;
    memman_select("HeTM_accounts_a");
    c_data->dev_a = (account_t*)memman_get_gpu(NULL);
    memman_select("HeTM_accounts_b");
    c_data->dev_b = (account_t*)memman_get_gpu(NULL);
    memman_select("HeTM_accounts_bckp");
    c_data->dev_bckp = (account_t*)memman_get_gpu(NULL);
    memman_select("HeTM_flag_inter_conflict");
    c_data->dev_flag = (int*)memman_get_gpu(NULL);

    c_data->devStates = (void*)devStates;
    c_data->size = cuda_info.size;
    c_data->dev_zc = NULL;
    c_data->threadNum = cuda_info.threadNum;
    c_data->blockNum = cuda_info.blockNum;
    c_data->TransEachThread = cuda_info.TransEachThread;
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
stream_t *jobWithCuda_initStream(cuda_t *d, int id, int count)
{
  stream_t *stream_data = NULL;

  // per thread
  stream_data = (stream_t*)malloc( sizeof(stream_t) );

  // Each CPU thread allocates some space in the GPU to dump
  // its thread local STM log.
  // memman_alloc_gpu("HeTM_CPU_wset_log", logSize, NULL, MEMMAN_THRLC);
  // stream_data->host_log = (HeTM_CPULogEntry*)memman_get_gpu(NULL);

  //Save pointers
  stream_data->isCudaError = 0;
  stream_data->isCmpDone = 0;
  stream_data->id = id;
  stream_data->maxC = count;
  stream_data->count = 0;
  stream_data->status = READ;
  pthread_mutex_init(&stream_data->mutex, NULL);

  // THERE IS 1 SINGLE STREAM PER THREAD
  knlman_add_stream();
  stream_data->st = (cudaStream_t)knlman_get_current_stream();

  return stream_data;
}

// TODO: cuda_stop CANNOT be a module static variable --> HeTM global struct needed
extern "C"
void jobWithCuda_threadCheck(cuda_t *d, stream_t *s)
{
  HeTM_CPULogNode_t *auxFree;
  size_t bufSize = STM_LOG_BUFFER_SIZE * LOG_SIZE, curBuf;
  int i;
  thread_local static HeTM_CPULogNode_t *initNode = NULL;
  thread_local static HeTM_CPULogNode_t *curNode  = NULL;

  // TODO: set this d->dthreads[d->id].HeTM_CPULog

  // did the GPU finished the batch?
  if (HeTM_get_GPU_status() != HETM_BATCH_DONE) return;
  if (s->status != READ) return; // not ready yet

  // TODO: this is probably bogus: GETing the log cleans it! I only want to
  // remove some chunks in the beginning of the log
  if (initNode == NULL || (curNode != NULL && curNode->nbNodes == 0)) {
    if (initNode != NULL) {
      TM_GET_LOG(curNode); // Never enters here (fetches more log before threshold)
    } else {
      TM_GET_LOG(initNode);
      curNode = initNode;
    }
  }

	s->isCmpDone = 0;
	__sync_synchronize();
	if (s->isThreshold) {
		/* stop sending comparison kernels to the GPU */
		s->count = 0;
    s->isThreshold = 0;
		s->status = READ;

    if (!HeTM_is_interconflict()) {
      // must block
      i = 0;
      while (curNode != NULL) {
        s->isCmpDone = 0;
        if (curNode->curPos == 0) break;
        curNode = jobWithCuda_mergeLog(s->st, curNode, &curBuf, 1);
        jobWithCuda_checkStream(d, s, curBuf);
        while (!s->isCmpDone) pthread_yield(); // block
        if (HeTM_is_interconflict()) break;
        i++;
      }
    }
    // else --> already got a conflict --> no need to keep checking

    curNode = initNode;
		while (curNode != NULL) { // TODO: freeLog() function
			auxFree = curNode;
			curNode = curNode->next;
      stm_log_node_free(auxFree); // does not free an empty node
		}

    TM_GET_LOG(initNode); // destroys the log
		initNode = curNode = NULL;
		HeTM_GPU_wait(); // /* Wake up GPU controller thread */
		HeTM_GPU_wait(); // /* wait to set the cuda_stop flag to 0 */
	} else {
		s->status = STREAM;

    curNode = jobWithCuda_mergeLog(s->st, curNode, &curBuf, 0);
    jobWithCuda_checkStream(d, s, curBuf);

    // TODO: what if the CPU generates log FASTER than what the GPU consumes?
    // -----> add threashold
    i = 0; // get the remaining size of the log (TODO: logSize() function)
    auxFree = curNode;
    while (auxFree != NULL) {
      i++;
			auxFree = auxFree->next;
		}
    // ---

		if (curBuf < bufSize || i < STM_LOG_BUFFER_SIZE) {
      // all comparison fit in the buffer OR stop in next iteration
      s->isThreshold = 1;
    }
	}
  s->count++;
}

/* Merges all the logs in one contiguos chunk in the GPU */
extern "C"
HeTM_CPULogNode_t* jobWithCuda_mergeLog(cudaStream_t stream, HeTM_CPULogNode_t *t, size_t *size, int isBlock)
{
  void *res;
  HeTM_CPULogEntry *resAux;
  size_t sizeRes = 0, sizeToCpy, sizeBuffer;
  HeTM_CPULogNode_t *logAux;

  sizeBuffer = STM_LOG_BUFFER_SIZE * LOG_SIZE;

  memman_select("HeTM_CPU_wset_buffer");
  res = memman_get_gpu(NULL);

  resAux = (HeTM_CPULogEntry*)res;

  logAux = t;
  while (logAux != NULL) {
    sizeToCpy = logAux->curPos * sizeof(HeTM_CPULogEntry);
    CUDA_CPY_TO_DEV_ASYNC(resAux, logAux->array, sizeToCpy, stream);
    resAux += logAux->curPos; // move ahead of the copied position
    sizeRes += logAux->curPos;

    // TODO: use logAux->isLast
    if (logAux->nbNodes > 1 && logAux->next != NULL) { // not the last block
      logAux->next->nbNodes = logAux->nbNodes - 1;
    }

    if (logAux->next == NULL) { // may happen if isBlock == 1
      logAux = logAux->next;
      break;
    }

    // --- detects log chain ending
    if ((logAux->nbNodes <= 1 && !isBlock) || sizeRes + logAux->next->curPos > sizeBuffer) {
      // next node goes in a different batch
      logAux = logAux->next;
      break;
    }
    // ---


    logAux = logAux->next;
  }

  if (size != NULL) *size = sizeRes;
  return logAux;
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
int jobWithCuda_checkStream(cuda_t *d, stream_t *st, size_t size_stm)
{
  cudaError_t cudaStatus;
  bool err = 1;
  int ret = 1; //Return value
  HeTM_CPULogEntry *vecDev;

  memman_select("HeTM_CPU_wset_buffer");
  vecDev = (HeTM_CPULogEntry*)memman_get_gpu(NULL);
  // size_t logSize = size_stm * sizeof(HeTM_CPULogEntry);

  if (size_stm == 0) {
    st->isCmpDone = 1;
    __sync_synchronize();
    return 0;
  }

  while (err) {
    err=0;

    knlman_choose_next_stream();
    // knlman_sync_stream(); // synchronize threads
    // cudaStream_t stream = (cudaStream_t)knlman_get_current_stream();

    memman_select("HeTM_dev_rset");
    void *rset = memman_get_gpu(NULL);
    memman_select("HeTM_cpu_versions");
    long *vers  = (long*)memman_get_gpu(NULL);

#if CMP_TYPE == CMP_COMPRESSED
    // -----------------------------------------------
    //Calc number of blocks
    int nbThreadsX = 128;
    int bo = (size_stm + nbThreadsX-1) / (nbThreadsX);
    dim3 blocksCheck(bo); // partition the stm_log by the different blocks
    dim3 threadsPerBlock(nbThreadsX); // each block has nbThreadsX threads

    knlman_select("HeTM_checkTxCompressed");
    knlman_set_nb_blocks(bo, 1, 1);
    knlman_set_nb_threads(nbThreadsX, 1, 1);

    // Memory region of the entry object
    HeTM_checkTxCompressed_s checkTxCompressed_args = {
      .knlArgs = {
        .dev_flag = d->dev_flag,
        .stm_log  = vecDev,
        .size_stm = (int)size_stm,
        .size_log = d->size,
        .mutex    = PR_lockTableDev,
        .devLogR  = (int*)rset,
        .a        = d->dev_a,
        .b        = d->dev_b,
        .vers     = vers,
      },
      .clbkArgs = st
    };
    knlman_set_entry_object(&checkTxCompressed_args);

    knlman_run();
    // -----------------------------------------------
#elif CMP_TYPE == CMP_EXPLICIT
    int xThrs = CMP_EXPLICIT_THRS_PER_RSET / CMP_EXPLICIT_THRS_PER_WSET;
    int yThrs = CMP_EXPLICIT_THRS_PER_WSET;

    int hasRemainderXThrs = EXPLICIT_LOG_SIZE(d->blockNum, d->threadNum) % xThrs;

    int xBlocks = EXPLICIT_LOG_SIZE(d->blockNum, d->threadNum) / xThrs;
    int yBlocks = size_stm / CMP_EXPLICIT_THRS_PER_WSET;

    int hasRemainderYBlocks = size_stm % CMP_EXPLICIT_THRS_PER_WSET;

    if (hasRemainderXThrs) xBlocks++;
    if (hasRemainderYBlocks) yBlocks++;

    knlman_select("HeTM_checkTxExplicit");
    knlman_set_nb_blocks(xBlocks, yBlocks, 1);
    knlman_set_nb_threads(xThrs, yThrs, 1);

    HeTM_checkTxExplicit_s checkTxExplicit_args = {
      .knlArgs = {
        .dev_flag = d->dev_flag,
        .stm_log  = vecDev,
        .size_stm = (int)size_stm,
        .size_logR= EXPLICIT_LOG_SIZE(d->blockNum, d->threadNum),
        .devLogR  = (int*)rset,
        .mutex    = PR_lockTableDev,
        .a        = d->dev_a,
        .b        = d->dev_b,
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
account_t* jobWithCuda_swap(cuda_t *d){
  return d->host_a;
}

/****************************************
 *	jobWithCuda_cpyDatasetToGPU(d,b)
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
int jobWithCuda_cpyDatasetToGPU(cuda_t *d, account_t *b)
{
  cudaError_t cudaStatus;

  cudaStatus = cudaMemcpy(d->dev_a, b, d->size*sizeof(account_t), cudaMemcpyHostToDevice); //copy new vector from CPU to GPU
  if (cudaStatus != cudaSuccess) {
    printf("cudaMemcpy to device failed for accounts!\n");
    return 0;
  }

  return 1;
}

extern "C"
int jobWithCuda_resetGPUSTMState(cuda_t *d)
{
  cudaError_t cudaStatus;

  memman_select("HeTM_flag_inter_conflict");
  memman_zero_dual(NULL);

  cudaStatus = cudaMemset(PR_lockTableDev, 0, PR_LOCK_TABLE_SIZE*sizeof(int));	//copy new vector from CPU to GPU
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

  while (err) {
    err = 0;

    CHECK_ERROR_CONTINUE(cudaThreadSynchronize());

    // TODO: bm is used to just copy the relevant part of the dataset
    // right now I'm copying it all (re-implement if needed)

    CHECK_ERROR_CONTINUE(cudaMemcpy(a, d->dev_a, d->size*sizeof(account_t), cudaMemcpyDeviceToHost)); //copy results from GPU to CPU
  }

  cudaStatus = cudaThreadSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("\nCpyBack: Error code is: %s.\n", cudaGetErrorString(cudaStatus));
    //printf("\nError code is: %s\n", cudaGetErrorString(cudaStatus));
    return 0;
  }

  //These will be swaped again on kernel launch, need to fix this (TODO)
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
void jobWithCuda_getStats(cuda_t *d, long *ab, long *com) {
  cudaError_t cudaStatus;
  int err = 1;

  while(err) {
    err = 0;

    CHECK_ERROR_CONTINUE(cudaThreadSynchronize());
    HeTM_bankTx_cpy_IO();

    //Transfer aborts
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
    memman_select("HeTM_CPU_wset_buffer");
    memman_free_gpu();

    memman_select("HeTM_flag_inter_conflict");
    memman_free_dual();

    cudaFree(d->devStates);
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


// TODO: not in use anymore --> re-implement copy back only modified dataset from GPU
extern "C"
int jobWithCuda_bm(cuda_t *d, int *bm)
{
  cudaError_t cudaStatus;
  int err = 1;

  while (err) {
    err = 0;

    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
      printf("jobWithCuda_bm() detected previous error!");
      //goto Error;
      continue;
    }

    knlman_select("HeTM_finalTxLog2");

    /* Define Kernel Size */
    knlman_set_nb_blocks((d->size+1023)/(1024), 1, 1);
    knlman_set_nb_threads(1024, 1, 1);

    HeTM_knl_finalTxLog2_s finalTxLog2_args = {
      .global_bitmap = d->dev_bm,
      .size          = d->size,
      .devLogR       = d->dev_LogR
    };
    knlman_set_entry_object(&finalTxLog2_args);
    knlman_run();
    cudaThreadSynchronize();

    //Transfer result
    cudaStatus = cudaMemcpy(bm, d->dev_bm, d->bm_size*sizeof(int), cudaMemcpyDeviceToHost);	//copy abortcounts
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
int jobWithCuda_checkStreamFinal(cuda_t *d, stream_t *st, int n)
{
  cudaError_t cudaStatus;
  int *isConflict;

  memman_select("HeTM_flag_inter_conflict");
  isConflict = (int*)memman_get_cpu(NULL);
  if (!*isConflict) {
    // do not need to copy again if isConflict
    memman_cpy_to_cpu(NULL); /* returns whether the comparison was successful */
  }

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
void jobWithCuda_backup(cuda_t *d) {

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
