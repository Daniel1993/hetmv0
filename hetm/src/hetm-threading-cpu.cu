#include "hetm-log.h"
#include "hetm.cuh"
#include "stm-wrapper.h"
#include "stm.h" // depends on STM
#include "knlman.h"
#include "hetm-cmp-kernels.cuh"
// #include ".h" // depends on STM

#include <list>
#include <mutex>

#define NVTX_PROF_BLOCK   0
#define NVTX_PROF_BACKOFF 1

static std::list<HeTM_callback> beforeCPU, afterCPU;
std::mutex HeTM_statsMutex; // extern in hetm-threading-gpu

#define MAXIMUM_THREADS 1024

// TODO: still too many memcpyies

static void initThread(int id, void *data);
static void exitThread(int id, void *data);

#if HETM_LOG_TYPE != HETM_BMAP_LOG
static int consecutiveFlagCpy = 0; // avoid consecutive copies

thread_local static int inBackoff = 0;

static int launchCmpKernel(HeTM_thread_s*, size_t wsetSize);
static void checkCmpDone();
static void cmpBlockApply();
static void cpyWSetToGPU();
static void asyncCpy(void *argsPtr);
static void asyncCmp(void *argsPtr);
static void asyncGetInterConflFlag(void*);
#endif

void HeTM_cpu_thread()
{
  int threadId = HeTM_thread_data->id;
  HeTM_callback callback = HeTM_thread_data->callback;
  void *clbkArgs = HeTM_thread_data->args;

  // TODO: check order
  TM_INIT_THREAD(HeTM_shared_data.hostMemPool, HeTM_shared_data.sizeMemPool);
  initThread(threadId, clbkArgs);

  HETM_DEB_THRD_CPU("starting CPU worker %i", threadId);
  // printf("starting CPU worker %i\n", threadId);
  HeTM_sync_barrier();

  if (HeTM_shared_data.isGPUEnabled == 0) {
    // GPU threads is resposible for update statistics, but is off
    while (CONTINUE_COND) {
      callback(threadId, clbkArgs); // does 1 transaction
      HeTM_thread_data->curNbTxs++;
    }
    __sync_add_and_fetch(&HeTM_stats_data.nbTxsCPU, HeTM_thread_data->curNbTxs);
    __sync_add_and_fetch(&HeTM_stats_data.nbCommittedTxsCPU, HeTM_thread_data->curNbTxs);
    // HeTM_stats_data.nbDroppedTxsCPU == 0;
  } else {
    while (CONTINUE_COND) {
#if HETM_LOG_TYPE != HETM_BMAP_LOG
      checkCmpDone();
      cpyWSetToGPU(); // Tests if ready
#else /* HETM_LOG_TYPE == HETM_BMAP_LOG */
      if (HeTM_get_GPU_status() == HETM_BATCH_DONE) {
        NVTX_PUSH_RANGE("blocked", NVTX_PROF_BLOCK);
        HeTM_sync_barrier(); // just block and let the GPU do its thing
        HeTM_sync_barrier();
        NVTX_POP_RANGE();
      }
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */
      callback(threadId, clbkArgs); // does 1 transaction
      HeTM_thread_data->curNbTxs++;
      if (HeTM_get_GPU_status() == HETM_BATCH_DONE) {
        // transaction done while comparing
        HeTM_thread_data->curNbTxsNonBlocking++;
      }
    }
    NVTX_POP_RANGE();
  }

  HETM_DEB_THRD_CPU("exiting CPU worker %i", threadId);

  exitThread(threadId, clbkArgs);
  TM_EXIT_THREAD();
}

int HeTM_before_cpu_start(HeTM_callback req)
{
  beforeCPU.push_back(req);
  return 0;
}

int HeTM_after_cpu_finish(HeTM_callback req)
{
  afterCPU.push_back(req);
  return 0;
}

static void initThread(int id, void *data)
{
  knlman_add_stream(); // each thread has its stream
  HeTM_thread_data->stream = knlman_get_current_stream();
  stm_log_init();
  HeTM_thread_data->wSetLog = stm_thread_local_log;
  cudaEventCreate(&HeTM_thread_data->cmpStartEvent);
  cudaEventCreate(&HeTM_thread_data->cmpStopEvent);
  cudaEventCreate(&HeTM_thread_data->cpyWSetStartEvent);
  cudaEventCreate(&HeTM_thread_data->cpyWSetStopEvent);
  cudaEventCreate(&HeTM_thread_data->cpyDatasetStartEvent);
  cudaEventCreate(&HeTM_thread_data->cpyDatasetStopEvent);
  for (auto it = beforeCPU.begin(); it != beforeCPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

static void exitThread(int id, void *data)
{
  HETM_DEB_THRD_CPU("Time cpy WSet = %10fms - Time cmp = %10fms \n",
    HeTM_thread_data->timeCpySum, HeTM_thread_data->timeCmpSum);

  printf("[%i] Total empty space first chunk = %zu B\n",
    HeTM_thread_data->id, HeTM_thread_data->emptySpaceFirstChunk);

  HeTM_statsMutex.lock();
  HeTM_stats_data.totalTimeCpyWSet += HeTM_thread_data->timeCpySum;
  HeTM_stats_data.totalTimeCmp += HeTM_thread_data->timeCmpSum;
  HeTM_stats_data.totalTimeCpyDataset += HeTM_thread_data->timeCpyDatasetSum;
  HeTM_stats_data.timeNonBlocking += HeTM_thread_data->timeBackoff;
  HeTM_stats_data.timeBlocking += HeTM_thread_data->timeBlocked;
  HeTM_statsMutex.unlock();

  for (auto it = afterCPU.begin(); it != afterCPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

#if HETM_LOG_TYPE != HETM_BMAP_LOG
static void asyncCpy(void *argsPtr)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)argsPtr;
#if HETM_LOG_TYPE == HETM_VERS2_LOG
  // TODO: 64 == upper bound of threads
  chunked_log_s truncated;

  threadData->curWSetSize = STM_LOG_BUFFER_SIZE*LOG_SIZE*LOG_ACTUAL_SIZE*sizeof(HeTM_CPULogEntry);
  truncated = MOD_CHUNKED_LOG_TRUNCATE(threadData->wSetLog, STM_LOG_BUFFER_SIZE);
  if (truncated.first != NULL) {
    HeTM_wset_log_cpy_to_gpu(threadData, truncated.first, &threadData->curWSetSize);
    CHUNKED_LOG_DESTROY(&truncated);
  }
  consecutiveFlagCpy = 0; // allow to cpy the flag
#else /* HETM_LOG_TYPE == HETM_VERS_LOG */
  chunked_log_node_s *truncated;
  // truncates the exact amount
  truncated = stm_log_truncate(threadData->wSetLog);
  consecutiveFlagCpy = 0; // allow to cpy the flag
  HeTM_wset_log_cpy_to_gpu(threadData, truncated, &threadData->curWSetSize);
#endif /* HETM_LOG_TYPE == HETM_VERS_LOG */
  HETM_DEB_THRD_CPU("Buffered WSet of size %zu\n", threadData->curWSetSize);
}

static void asyncCmp(void *argsPtr)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)argsPtr;

  consecutiveFlagCpy = 0; // allow to cpy the flag

  launchCmpKernel(threadData, threadData->curWSetSize);
}

static void cpyWSetToGPU()
{
  // did the GPU finished the batch?
  if (HeTM_get_GPU_status() != HETM_BATCH_DONE) return;

  if (inBackoff == 0) {
    TIMER_READ(HeTM_thread_data->backoffBegTimer);
  }

  if (!HeTM_thread_data->isCpyDone && HeTM_thread_data->statusCMP == HETM_CPY_ASYNC) {
    return; // not ready yet
  }
  if (!HeTM_thread_data->isCmpDone && HeTM_thread_data->statusCMP == HETM_CMP_ASYNC) {
    return; // not ready yet
  }

  if (HeTM_thread_data->isCpyDone && HeTM_thread_data->statusCMP == HETM_CPY_ASYNC) {
    HeTM_thread_data->statusCMP = HETM_CMP_ASYNC;
    HeTM_thread_data->isCmpDone = 0;

    CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cpyWSetStartEvent);
    CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cpyWSetStopEvent);
    CUDA_EVENT_ELAPSED_TIME(&HeTM_thread_data->timeCpy, HeTM_thread_data->cpyWSetStartEvent,
      HeTM_thread_data->cpyWSetStopEvent);
    if (HeTM_thread_data->timeCpy > 0) { // TODO: else is boggus ---> happens a lot
      HeTM_thread_data->timeCpySum += HeTM_thread_data->timeCpy;
    }
    HeTM_async_request((HeTM_async_req_s){
      .args = (void*)HeTM_thread_data,
      .fn = asyncCmp
    });
    // printf("Cpy done for worker %i\n", HeTM_thread_data->id);
    return;
  }

  if (HeTM_thread_data->isCmpDone) { // cmp completed
    // TODO: this slows down the GPU --> put into a #ifdef
    CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cmpStartEvent);
    CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cmpStopEvent);
    CUDA_EVENT_ELAPSED_TIME(&HeTM_thread_data->timeCmp, HeTM_thread_data->cmpStartEvent,
      HeTM_thread_data->cmpStopEvent);
    HeTM_thread_data->timeCmpSum += HeTM_thread_data->timeCmp;
  }

	HeTM_thread_data->isCmpDone = 0;
  __sync_synchronize();
	if (HeTM_thread_data->statusCMP == HETM_CMP_BLOCK) {
		/* stop sending comparison kernels to the GPU */
    cmpBlockApply();
	} else {

    // 1st policy --> if enough log do not block
#if HETM_LOG_TYPE == HETM_VERS2_LOG
    if (!MOD_CHUNKED_LOG_IS_FLAT(HeTM_thread_data->wSetLog)) {
#else
    int i = HeTM_thread_data->wSetLog->size;
    if (i > STM_LOG_BUFFER_SIZE) {
#endif
      // continue running the CPU
      if (!inBackoff) {
        // printf("[%i] enter backoff\n", HeTM_thread_data->id);
        NVTX_PUSH_RANGE("backoff_mode", NVTX_PROF_BACKOFF);
        inBackoff = 1;
      }
      HeTM_thread_data->statusCMP = HETM_CPY_ASYNC;
      HeTM_async_request((HeTM_async_req_s){
        .args = (void*)HeTM_thread_data,
        .fn = asyncCpy
      });

      // TODO: I'm spamming these
      HeTM_async_request((HeTM_async_req_s){
        .args = NULL,
        .fn = asyncGetInterConflFlag
      });
    } else {
      // all comparison fit in the buffer OR stop in next iteration
      HeTM_thread_data->statusCMP = HETM_CMP_BLOCK;
      cmpBlockApply();
    }
	}
  HeTM_thread_data->nbCmpLaunches++; // case that CPU is faster than GPU
}

static void cmpBlockApply()
{
  int i;
#if HETM_LOG_TYPE == HETM_VERS2_LOG
  size_t curNodeSize = !MOD_CHUNKED_LOG_IS_EMPTY(HeTM_thread_data->wSetLog);
#else
  size_t curNodeSize = HeTM_thread_data->wSetLog->size;
#endif

  // TODO: should only be called if CMP_ASYNC before
  HeTM_async_request((HeTM_async_req_s){
    .args = NULL,
    .fn = asyncGetInterConflFlag
  });

  HETM_DEB_THRD_CPU("Thread %i reachead CMP threshold WSetSize=%zu(x64k)",
    HeTM_thread_data->id, curNodeSize);
#if HETM_LOG_TYPE == HETM_VERS2_LOG
  if (MOD_CHUNKED_LOG_IS_EMPTY(HeTM_thread_data->wSetLog)) {
#else
  if (CHUNKED_LOG_IS_EMPTY(HeTM_thread_data->wSetLog)) {
#endif
    HeTM_thread_data->isCmpDone = 1;
  }
  if (inBackoff) {
    // printf("[%i] exit backoff\n", HeTM_thread_data->id);
    NVTX_POP_RANGE();
    inBackoff = 0;
  }
  NVTX_PUSH_RANGE("blocked", NVTX_PROF_BLOCK);
  TIMER_READ(HeTM_thread_data->backoffEndTimer);
  HeTM_thread_data->timeBackoff += TIMER_DIFF_SECONDS(
    HeTM_thread_data->backoffBegTimer, HeTM_thread_data->backoffEndTimer
  );

  if (curNodeSize > 0 && !HeTM_is_interconflict()) {
    // must block
    i = 0;
    while (
#if HETM_LOG_TYPE == HETM_VERS2_LOG
      !MOD_CHUNKED_LOG_IS_EMPTY(HeTM_thread_data->wSetLog)
#else
      !CHUNKED_LOG_IS_EMPTY(HeTM_thread_data->wSetLog)
#endif
    ) {
      HeTM_thread_data->isCmpDone = 0;
      __sync_synchronize();
#if HETM_LOG_TYPE == HETM_VERS_LOG
      if (HeTM_thread_data->wSetLog->first->p.pos == 0) break;
#endif

      HeTM_async_request((HeTM_async_req_s){
        .args = (void*)HeTM_thread_data,
        .fn = asyncCpy
      });

      while (!HeTM_thread_data->isCpyDone) {
        // __sync_synchronize();
        // _mm_pause();
        pthread_yield(); // block
      }

      CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cpyWSetStartEvent);
      CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cpyWSetStopEvent);
      CUDA_EVENT_ELAPSED_TIME(&HeTM_thread_data->timeCpy, HeTM_thread_data->cpyWSetStartEvent,
        HeTM_thread_data->cpyWSetStopEvent);
      if (HeTM_thread_data->timeCpy > 0) { // TODO: bug
        HeTM_thread_data->timeCpySum += HeTM_thread_data->timeCpy;
      }

      HeTM_async_request((HeTM_async_req_s){
        .args = (void*)HeTM_thread_data,
        .fn = asyncCmp
      });

      while (!HeTM_thread_data->isCmpDone) {
        // __sync_synchronize();
        // _mm_pause();
        pthread_yield(); // block
      }

      CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cmpStartEvent);
      CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cmpStopEvent);
      CUDA_EVENT_ELAPSED_TIME(&HeTM_thread_data->timeCmp, HeTM_thread_data->cmpStartEvent,
        HeTM_thread_data->cmpStopEvent);
      HeTM_thread_data->timeCmpSum += HeTM_thread_data->timeCmp;

      HeTM_async_request((HeTM_async_req_s){
        .args = NULL,
        .fn = asyncGetInterConflFlag
      });

      // wait flag?

      if (HeTM_is_interconflict()) break;
      i++;
    } /* while not empty */
  }  /* no inter-conflict */

  HeTM_sync_barrier(); // /* Wake up GPU controller thread */
  HeTM_sync_barrier(); // /* wait to set the cuda_stop flag to 0 */
  NVTX_POP_RANGE();

  HeTM_thread_data->isFirstChunk = 1;

  TIMER_READ(HeTM_thread_data->blockingEndTimer);
  HeTM_thread_data->timeBlocked += TIMER_DIFF_SECONDS(
    HeTM_thread_data->backoffEndTimer, HeTM_thread_data->blockingEndTimer
  );

  // printf("[%i] exit blocked\n", HeTM_thread_data->id);
  HeTM_thread_data->statusCMP = HETM_CMP_OFF;
  HeTM_thread_data->nbCmpLaunches = 0;
}

static int launchCmpKernel(HeTM_thread_s *threadData, size_t wsetSize)
{
  HeTM_CPULogEntry *vecDev;
  size_t sizeBuffer = STM_LOG_BUFFER_SIZE * LOG_SIZE;
  int tid = threadData->id;

  vecDev = (HeTM_CPULogEntry*)HeTM_shared_data.wsetLog;
  vecDev += tid*sizeBuffer; // each thread has a bit of the buffer

  // PROBLEM --> the wsetSize is 0 because it was already copied
  if (HeTM_is_interconflict() || wsetSize == 0) {
    HETM_DEB_THRD_CPU("Thread %i decided not to CMP (isConfl=%i, wsetSize=%i)",
      threadData->id, HeTM_is_interconflict(), wsetSize);
    threadData->isCmpDone = 1; // TODO: put global
    __sync_synchronize();
    return 0;
  }
  HETM_DEB_THRD_CPU("Thread %i decided to CMP (wsetSize=%i)", threadData->id,
    wsetSize);

#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
  // -----------------------------------------------
  //Calc number of blocks
  int nbThreadsX = 256;
  int bo = (wsetSize + nbThreadsX-1) / (nbThreadsX);

  // Memory region of the entry object
  HeTM_cmp_s checkTxCompressed_args = {
    .knlArgs = {
      .sizeWSet = (int)wsetSize,
      .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
    },
    .clbkArgs = threadData
  };

#if HETM_LOG_TYPE == HETM_VERS2_LOG
  // size_t nbGranules = HeTM_shared_data.sizeMemPool / sizeof(PR_GRANULE_T);
  size_t granPerThread = LOG_SIZE*STM_LOG_BUFFER_SIZE;
  nbThreadsX = LOG_THREADS_IN_BLOCK;
  bo = LOG_GPU_THREADS / nbThreadsX;

  // number of entries for each thread
  checkTxCompressed_args.knlArgs.sizeRSet = granPerThread;
#endif /* HETM_LOG_TYPE == HETM_VERS2_LOG */

  dim3 blocksCheck(bo); // partition the stm_log by the different blocks
  dim3 threadsPerBlock(nbThreadsX); // each block has nbThreadsX threads

  knlman_select("HeTM_checkTxCompressed");
  knlman_set_nb_blocks(bo, 1, 1);
  knlman_set_nb_threads(nbThreadsX, 1, 1);
  knlman_set_entry_object(&checkTxCompressed_args);
  knlman_run(threadData->stream);
  // -----------------------------------------------
#elif HETM_CMP_TYPE == HETM_CMP_EXPLICIT
  int xThrs = CMP_EXPLICIT_THRS_PER_RSET / CMP_EXPLICIT_THRS_PER_WSET;
  int yThrs = CMP_EXPLICIT_THRS_PER_WSET;
  int nbGPUThreads = HeTM_shared_data.nbGPUThreads;
  int nbGPUBlocks  = HeTM_shared_data.nbGPUBlocks;
  int blockSize    = HeTM_get_explicit_log_block_size();
  long explicitLogSize = nbGPUThreads * nbGPUBlocks * blockSize;

  int hasRemainderXThrs = explicitLogSize % xThrs;

  int xBlocks = explicitLogSize / xThrs;
  int yBlocks = wsetSize / CMP_EXPLICIT_THRS_PER_WSET;

  int hasRemainderYBlocks = wsetSize % CMP_EXPLICIT_THRS_PER_WSET;

  if (hasRemainderXThrs) xBlocks++;
  if (hasRemainderYBlocks) yBlocks++;

  knlman_select("HeTM_checkTxExplicit");
  knlman_set_nb_blocks(xBlocks, yBlocks, 1);
  knlman_set_nb_threads(xThrs, yThrs, 1);

  HeTM_cmp_s checkTxExplicit_args = {
    .knlArgs = {
      .sizeWSet = (int)wsetSize,
      .sizeRSet = (int)explicitLogSize,
    },
    .clbkArgs = threadData
  };
  knlman_set_entry_object(&checkTxExplicit_args);
  knlman_run(threadData->stream);
#else
  printf("Error! no compare method selected!\n");
#endif

  return 0;
}

static void checkCmpDone()
{
  if (HeTM_thread_data->isCmpDone) {
    if (HeTM_thread_data->nbCmpLaunches >= HETM_MAX_CMP_KERNEL_LAUNCHES
        || HeTM_is_interconflict()) {
      HeTM_thread_data->statusCMP = HETM_CMP_BLOCK;
      __sync_synchronize();
    }
  }
}

static void asyncGetInterConflFlag(void*)
{
  if (!consecutiveFlagCpy) {
    consecutiveFlagCpy = 1;
    HeTM_set_is_interconflict(HeTM_get_inter_confl_flag());
  }
}
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */
