#include "hetm.cuh"
#include "pr-stm-wrapper.cuh"
#include "hetm-timer.h"
#include "hetm-cmp-kernels.cuh"
#include "knlman.h"

#if HETM_LOG_TYPE == HETM_ADDR_LOG
#include "hetm-cmp-kernels.cuh" // must run apply kernel
#endif

#include <list>
#include <mutex>

#define RUN_ASYNC(_func, _args) \
  HeTM_async_request((HeTM_async_req_s){ \
    .args = (void*)_args, \
    .fn = _func, \
  }) \
//

extern std::mutex HeTM_statsMutex; // defined in hetm-threading-cpu
pr_tx_args_s HeTM_pr_args; // used only by the CUDA-control thread (TODO: PR-STM only)

static std::list<HeTM_callback> beforeGPU, afterGPU, beforeBatch, afterBatch;
static HeTM_callback choose_policy;
static int isAfterCmpDone = 0;
static int isGetStatsDone = 0;
static int isGetPRStatsDone = 0;
static int isDatasetSyncDone = 0;
static int isGPUResetDone = 0;

static void initThread(int id, void *data);
static void exitThread(int id, void *data);
static inline void prepareSyncDataset();
static inline void runBatch();
static inline void waitBatchEnd();
static inline void waitCMPEnd();
static inline void mergeDataset();
static inline void checkIsExit();
static inline void doGPUStateReset();
// static inline void waitDatasetEnd();
static inline void notifyCPUNextBatch();
static void updateStatisticsAfterBatch(void*);

static inline void runAfterGPU(int id, void *data);
static inline void runBeforeGPU(int id, void *data);

static inline void runAfterBatch(int id, void *data);
static inline void runBeforeBatch(int id, void *data);

// executed in other thread
static void beforeSyncDataset(void*);
static void offloadSyncDatasetAfterBatch(void*);
static void offloadWaitDataset(void*);
static void offloadGetPRStats(void*);
static void offloadAfterCmp(void*);
static void offloadResetGPUState(void*);

static int isWaitingCpyBack = 0;

#if HETM_LOG_TYPE == HETM_BMAP_LOG
static int launchCmpKernel(void*);
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */

#define WAIT_ON_FLAG(flag) while(!(flag)) pthread_yield(); flag = 0

#ifdef HETM_OVERLAP_CPY_BACK
static int isOverlappingKernel = 0;
static int isAfterOverlapping = 0;

void HeTM_gpu_thread()
{
  int threadId = HeTM_thread_data->id;
  void *clbkArgs = HeTM_thread_data->args;
  int isFirst = true;

  TIMER_T t1, t2, t3, t4;

  // TODO: check order
  PR_createStatistics(&HeTM_pr_args);
  initThread(threadId, clbkArgs);

  HeTM_sync_barrier();

  // first one is here, then before unlocking CPU threads
  runBeforeBatch(threadId, (void*)HeTM_thread_data);
loop_gpu_batch:
  // while (CONTINUE_COND) {
    // TODO: prepare syncBatchPolicy
    prepareSyncDataset();

loop_gpu_callback:
    if (!isAfterOverlapping) {
      doGPUStateReset();
      if (isOverlappingKernel) {
        RUN_ASYNC(beforeSyncDataset, HeTM_thread_data);
        isWaitingCpyBack = 1;
      }
      WAIT_ON_FLAG(isGPUResetDone); // forces GPU reset before start
      TIMER_READ(t1);
      if (!isFirst) {
        HeTM_stats_data.timeAfterCMP += TIMER_DIFF_SECONDS(t3, t1);
      } else {
        isFirst = true;
      }
      runBatch();
    } else {
      // -----------------------------
      // waitBatchEnd();
      // exit(0);
      // -----------------------------
      isAfterOverlapping = 0;
    }

    // TODO: right now is not working
    if (isOverlappingKernel) { // TODO: only allows 1 overlapping Kernel
      // does not overlap with the kernel
      WAIT_ON_FLAG(isWaitingCpyBack);

      RUN_ASYNC(offloadSyncDatasetAfterBatch, HeTM_thread_data);
      RUN_ASYNC(offloadWaitDataset, HeTM_thread_data);

#if HETM_LOG_TYPE == HETM_BMAP_LOG
      memman_select("HeTM_cpu_wset");
      memman_zero_cpu(NULL); // this is slow!
      memman_select("HeTM_cpu_wset_cache");
      memman_zero_cpu(NULL); // this is slow!
      memman_select("HeTM_cpu_wset_cache_confl");
      memman_zero_cpu(NULL); // this is slow!
      memman_zero_gpu(NULL); // this is slow!
#endif /* HETM_BMAP_LOG */
      WAIT_ON_FLAG(isDatasetSyncDone);
      isWaitingCpyBack = 0;
      isOverlappingKernel = 0;
      isAfterOverlapping = 1;
      goto loop_wait_on_dataset_cpy_back;
    }

    waitBatchEnd();

    TIMER_READ(t2);
    HeTM_stats_data.timePRSTM += TIMER_DIFF_SECONDS(t1, t2);

    waitCMPEnd();
    TIMER_READ(t3);
     // TODO: let the next batch begin if dataset is still not copied

    HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t2, t3);

    mergeDataset();
    checkIsExit();

    // if there is no conflict launch immediately the kernel while copying
    // assynchronously the dataset to the CPU, wake up CPU threads only after
    // copy is done

    WAIT_ON_FLAG(isAfterCmpDone);
    WAIT_ON_FLAG(isGetStatsDone); // resets the state

    isOverlappingKernel = 1;
    goto loop_gpu_callback; // waits dataset there

loop_wait_on_dataset_cpy_back:
    // waitDatasetEnd(); // waits after lauching kernel again
    runAfterBatch(threadId, (void*)HeTM_thread_data);
    runBeforeBatch(threadId, (void*)HeTM_thread_data);
    notifyCPUNextBatch();

    HETM_DEB_THRD_GPU(" --- End of batch (%li sucs, %li fail)",
      HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.nbBatchesFail);
  // }
  if (CONTINUE_COND) {
    goto loop_gpu_batch;
  }

  TIMER_READ(t4);
  HeTM_stats_data.timeAfterCMP += TIMER_DIFF_SECONDS(t3, t4);

  // TODO: this was per iteration
  // This will run sync
  RUN_ASYNC(offloadGetPRStats, HeTM_thread_data);
  RUN_ASYNC(updateStatisticsAfterBatch, -1);

  WAIT_ON_FLAG(isGetStatsDone);
  WAIT_ON_FLAG(isGetPRStatsDone);

  exitThread(threadId, clbkArgs);
}
#endif /* HETM_OVERLAP_CPY_BACK */

#ifndef HETM_OVERLAP_CPY_BACK
void HeTM_gpu_thread()
{
  int threadId = HeTM_thread_data->id;
  void *clbkArgs = HeTM_thread_data->args;

  TIMER_T t1, t2, t3;

  // TODO: check order
  PR_createStatistics(&HeTM_pr_args);
  initThread(threadId, clbkArgs);

  HeTM_sync_barrier();

  TIMER_READ(t1);

  // first one here, then before notify the CPU
  runBeforeBatch(threadId, (void*)HeTM_thread_data);
loop_gpu_batch:
  // while (CONTINUE_COND) {
    // TODO: prepare syncBatchPolicy
    prepareSyncDataset();

    doGPUStateReset();
    WAIT_ON_FLAG(isGPUResetDone); // forces GPU reset before start
    runBatch();

    waitBatchEnd();

    TIMER_READ(t2);
    // TODO: I'm taking GPU time in PR-STM
    HeTM_stats_data.timePRSTM += TIMER_DIFF_SECONDS(t1, t2);

    waitCMPEnd();
    TIMER_READ(t3);
     // TODO: let the next batch begin if dataset is still not copied

    HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t2, t3);

    mergeDataset();
    checkIsExit();

    RUN_ASYNC(beforeSyncDataset, HeTM_thread_data);
    RUN_ASYNC(offloadSyncDatasetAfterBatch, HeTM_thread_data);
    RUN_ASYNC(offloadWaitDataset, HeTM_thread_data);

    WAIT_ON_FLAG(isDatasetSyncDone);
    runAfterBatch(threadId, (void*)HeTM_thread_data);
    runBeforeBatch(threadId, (void*)HeTM_thread_data);
    notifyCPUNextBatch();

    TIMER_READ(t1);
    HeTM_stats_data.timeAfterCMP += TIMER_DIFF_SECONDS(t3, t1);

    HETM_DEB_THRD_GPU(" --- End of batch (%li sucs, %li fail)",
      HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.nbBatchesFail);
  // }
  if (CONTINUE_COND) {
    goto loop_gpu_batch;
  }

  // TODO: this was per iteration
  // This will run sync
  RUN_ASYNC(offloadGetPRStats, HeTM_thread_data);
  RUN_ASYNC(updateStatisticsAfterBatch, -1);

  WAIT_ON_FLAG(isWaitingCpyBack);
  WAIT_ON_FLAG(isAfterCmpDone); // avoids a warning in the compiler
  WAIT_ON_FLAG(isGetStatsDone);
  WAIT_ON_FLAG(isGetPRStatsDone);

  exitThread(threadId, clbkArgs);
}
#endif /* !HETM_OVERLAP_CPY_BACK */

static inline void prepareSyncDataset() { /* TODO */ }

static inline void runBatch()
{
  int threadId = HeTM_thread_data->id;
  HeTM_callback callback = HeTM_thread_data->callback;
  void *clbkArgs = HeTM_thread_data->args;
  if (HeTM_get_GPU_status() != HETM_IS_EXIT) {
    callback(threadId, clbkArgs);
  }
}

static inline void runBeforeGPU(int id, void *data)
{
  for (auto it = beforeGPU.begin(); it != beforeGPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

static inline void runAfterGPU(int id, void *data)
{
  for (auto it = afterGPU.begin(); it != afterGPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

static inline void runBeforeBatch(int id, void *data)
{
  for (auto it = beforeBatch.begin(); it != beforeBatch.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

static inline void runAfterBatch(int id, void *data)
{
  for (auto it = afterBatch.begin(); it != afterBatch.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

static inline void waitBatchEnd()
{
  PR_waitKernel();
}

static inline void waitCMPEnd()
{
  HeTM_set_GPU_status(HETM_BATCH_DONE); // notifies
  HeTM_sync_barrier(); // Blocks and waits comparison kernel to end
}

static inline void mergeDataset()
{
  RUN_ASYNC(offloadAfterCmp, HeTM_thread_data); // fetches the conflict flag
  // if (!isOverlappingKernel) {
    // RUN_ASYNC(offloadSyncDatasetAfterBatch, HeTM_thread_data);
  // }
  RUN_ASYNC(updateStatisticsAfterBatch, NULL);
}

static inline void checkIsExit()
{
  if (!HeTM_is_stop()) {
    HeTM_set_GPU_status(HETM_BATCH_RUN);
  } else { // Times up
    HeTM_set_GPU_status(HETM_IS_EXIT);
  }
}

static inline void doGPUStateReset()
{
  RUN_ASYNC(offloadResetGPUState, NULL);
}

// static inline void waitDatasetEnd()
// {
//   WAIT_ON_FLAG(isDatasetSyncDone); // only need to wait on this
// }

static inline void notifyCPUNextBatch()
{
  HeTM_sync_barrier(); // wakes the threads, GPU will re-run
}

int HeTM_choose_policy(HeTM_callback req)
{
  choose_policy = req;
  return 0;
}

int HeTM_before_gpu_start(HeTM_callback req)
{
  beforeGPU.push_back(req);
  return 0;
}

int HeTM_after_gpu_finish(HeTM_callback req)
{
  afterGPU.push_back(req);
  return 0;
}

int HeTM_before_batch(HeTM_callback req)
{
  beforeBatch.push_back(req);
  return 0;
}

int HeTM_after_batch(HeTM_callback req)
{
  afterBatch.push_back(req);
  return 0;
}

void initThread(int id, void *data)
{
  cudaEventCreate(&HeTM_thread_data->cmpStartEvent);
  cudaEventCreate(&HeTM_thread_data->cmpStopEvent);
  cudaEventCreate(&HeTM_thread_data->cpyWSetStartEvent);
  cudaEventCreate(&HeTM_thread_data->cpyWSetStopEvent);
  cudaEventCreate(&HeTM_thread_data->cpyDatasetStartEvent);
  cudaEventCreate(&HeTM_thread_data->cpyDatasetStopEvent);

  // cudaEventCreate(&HeTM_shared_data->batchStartEvent);
  // cudaEventCreate(&HeTM_shared_data->batchStopEvent);

  knlman_add_stream(); // each thread has its stream
  HeTM_thread_data->stream = knlman_get_current_stream();

  runBeforeGPU(id, data);
}

void exitThread(int id, void *data)
{
  HeTM_statsMutex.lock();
  HeTM_stats_data.totalTimeCpyWSet += HeTM_thread_data->timeCpySum;
  HeTM_stats_data.totalTimeCmp += HeTM_thread_data->timeCmpSum;
  HeTM_stats_data.totalTimeCpyDataset += HeTM_thread_data->timeCpyDatasetSum;
  HeTM_statsMutex.unlock();

  HeTM_stats_data.timeGPU = PR_kernelTime;
  runAfterGPU(id, data);
  HETM_DEB_THRD_GPU("Time copy dataset %10fms - Time cpy WSet %10fms - Time cmp %10fms\n",
    HeTM_thread_data->timeCpyDatasetSum, HeTM_thread_data->timeCpySum,
    HeTM_thread_data->timeCmpSum);

  CHUNKED_LOG_TEARDOWN(); // deletes the freed nodes from the CPU
}

static void beforeSyncDataset(void*)
{
  memman_select("HeTM_mempool");
  memman_cpy_to_cpu_buffer_bitmap();
  isWaitingCpyBack = 1;
  __sync_synchronize();
}

static void offloadSyncDatasetAfterBatch(void *args)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)args;
  size_t datasetCpySize;

  HETM_DEB_THRD_GPU("Syncing dataset ...");
  if (!HeTM_is_interconflict()) { // Successful execution
    // TODO: take statistic on this

    // Transfers the GPU WSet to CPU
    // TODO: bitmap is not in use (use bitmap to reduce the copyback size)
    CUDA_EVENT_RECORD(threadData->cpyDatasetStartEvent, (cudaStream_t)HeTM_memStream);
    HeTM_mempool_cpy_to_cpu(&datasetCpySize); // TODO: only in HETM_GPU_INV mode
    HeTM_stats_data.sizeCpyDataset += datasetCpySize;
    CUDA_EVENT_RECORD(threadData->cpyDatasetStopEvent, (cudaStream_t)HeTM_memStream);
  } else { // conflict detected

    CUDA_EVENT_RECORD(threadData->cpyDatasetStartEvent, (cudaStream_t)HeTM_memStream);
    if (HeTM_shared_data.policy == HETM_GPU_INV) {
      HeTM_mempool_cpy_to_gpu(&datasetCpySize); // ignores the data from GPU
      HeTM_stats_data.sizeCpyDataset += datasetCpySize;
    }
    // TODO: must restore CPU and GPU data --> backup on CPU memory
    // then: memcpy(mempool, backup); HeTM_mempool_cpy_to_gpu();
    if (HeTM_shared_data.policy == HETM_CPU_INV) {
      HeTM_mempool_cpy_to_cpu(&datasetCpySize); // ignores the data from CPU
      HeTM_stats_data.sizeCpyDataset += datasetCpySize;
    }
    // if (HeTM_shared_data.policy == HETM_CPU_INV)
    CUDA_EVENT_RECORD(threadData->cpyDatasetStopEvent, (cudaStream_t)HeTM_memStream);
  }
  // TODO: this is an hack for the GPU-only version
  // if (HeTM_shared_data.isCPUEnabled == 0) HeTM_mempool_cpy_to_gpu();
}

static void offloadWaitDataset(void *args)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)args;

  cudaStreamSynchronize((cudaStream_t)HeTM_memStream);
  // cudaDeviceSynchronize(); // TODO: now I'm waiting this to complete
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyDatasetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyDatasetStopEvent);

  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpyDataset, threadData->cpyDatasetStartEvent,
    threadData->cpyDatasetStopEvent);
  threadData->timeCpyDatasetSum += threadData->timeCpyDataset;

  isDatasetSyncDone = 1;
  __sync_synchronize();
}

static void offloadGetPRStats(void *argPtr)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)argPtr;

  PR_retrieveIO(&HeTM_pr_args, HeTM_memStream);
  threadData->curNbTxs = PR_nbCommits;
  HeTM_stats_data.nbAbortsGPU += PR_nbAborts;
  PR_resetStatistics(&HeTM_pr_args, HeTM_memStream);
  isGetPRStatsDone = 1;
  __sync_synchronize();
}

static void offloadAfterCmp(void *args)
{
#if HETM_LOG_TYPE == HETM_BMAP_LOG
  launchCmpKernel(args); // if BMAP blocks and run the kernel
#else /* HETM_LOG_TYPE != HETM_BMAP_LOG */

HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(HeTM_memStream));
#if HETM_LOG_TYPE == HETM_ADDR_LOG
  HeTM_thread_s *threadData = (HeTM_thread_s*)args;
  if (!HeTM_is_interconflict()) {
    size_t nbGranules = HeTM_shared_data.sizeMemPool / PR_LOCK_GRANULARITY;
    size_t wSetSize;

    int amount = 4; // TODO
    int nbThreads = 512;
    int modBlocks = nbGranules % (nbThreads * amount);
    int nbBlocks = nbGranules / (nbThreads * amount);
    if (modBlocks != 0) nbBlocks++;
    // args not needed
    cudaDeviceSynchronize(); // TODO: is this needed?
    memman_select("HeTM_mempool_backup"); // sends the CPU dataset (apply WSet)
    CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)HeTM_memStream);
    memman_cpy_to_gpu(HeTM_memStream, &wSetSize);
    HeTM_stats_data.sizeCpyWSet += wSetSize;
    CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)HeTM_memStream);
    cudaDeviceSynchronize(); // TODO: is this needed?
    CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
    CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
    CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
      threadData->cpyWSetStopEvent);
    if (threadData->timeCpy > 0) { // TODO: bug
      threadData->timeCpySum += threadData->timeCpy;
    }

    CUDA_EVENT_RECORD(threadData->cmpStartEvent, NULL);
    HeTM_knl_apply_cpu_data<<<nbBlocks, nbThreads>>>(amount, nbGranules);
    CUDA_EVENT_RECORD(threadData->cmpStopEvent, NULL);
    cudaDeviceSynchronize();
    CUDA_EVENT_SYNCHRONIZE(threadData->cmpStartEvent);
    CUDA_EVENT_SYNCHRONIZE(threadData->cmpStopEvent);
    CUDA_EVENT_ELAPSED_TIME(&threadData->timeCmp, threadData->cmpStartEvent,
      threadData->cmpStopEvent);
    threadData->timeCmpSum += threadData->timeCmp;
  }
#endif /* HETM_LOG_TYPE == HETM_ADDR_LOG */

#endif /* HETM_LOG_TYPE == HETM_BMAP_LOG */
  // gets the flag
  isAfterCmpDone = 1;
  __sync_synchronize();
}

static void updateStatisticsAfterBatch(void *arg)
{
  long committedTxsCPUBatch = 0; //, committedTxsGPUBatch = 0;
  long txsNonBlocking = 0;
  long droppedTxsCPUBatch = 0; //, droppedTxsGPUBatch = 0;
  // int idGPUThread = HeTM_shared_data.nbCPUThreads; // the last one

  if (arg == NULL) { // TODO: stupid hack to avoid transfer the commits
    HeTM_stats_data.nbBatches++;
    if (HeTM_is_interconflict()) {
      HeTM_stats_data.nbBatchesFail++;
    } else {
      HeTM_stats_data.nbBatchesSuccess++;
    }
  }

  choose_policy(0, arg);

  if (HeTM_is_interconflict() && HeTM_shared_data.policy == HETM_CPU_INV) {
    // drop CPU TXs
    for (int i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
      droppedTxsCPUBatch += HeTM_shared_data.threadsInfo[i].curNbTxs;
      HeTM_shared_data.threadsInfo[i].curNbTxs = 0;
    }
  } else {
    for (int i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
      committedTxsCPUBatch += HeTM_shared_data.threadsInfo[i].curNbTxs;
      // TODO: what if CPU INV?
      txsNonBlocking += HeTM_shared_data.threadsInfo[i].curNbTxsNonBlocking;
      HeTM_shared_data.threadsInfo[i].curNbTxs = 0;
      HeTM_shared_data.threadsInfo[i].curNbTxsNonBlocking = 0;
    }
  }

  // TODO: now this is done by choose_policy
  // if (HeTM_is_interconflict() && HeTM_shared_data.policy == HETM_GPU_INV) {
  //   // drop GPU TXs
  //   droppedTxsGPUBatch += HeTM_shared_data.threadsInfo[idGPUThread].curNbTxs;
  //   HeTM_shared_data.threadsInfo[idGPUThread].curNbTxs = 0;
  // } else {
  //   committedTxsGPUBatch += HeTM_shared_data.threadsInfo[idGPUThread].curNbTxs;
  //   HeTM_shared_data.threadsInfo[idGPUThread].curNbTxs = 0;
  // }

  HeTM_stats_data.nbTxsCPU += droppedTxsCPUBatch + committedTxsCPUBatch;
  HeTM_stats_data.nbCommittedTxsCPU += committedTxsCPUBatch;
  HeTM_stats_data.txsNonBlocking    += txsNonBlocking;
  HeTM_stats_data.nbDroppedTxsCPU   += droppedTxsCPUBatch;
  // HeTM_stats_data.nbTxsGPU += droppedTxsGPUBatch + committedTxsGPUBatch;
  // HeTM_stats_data.nbCommittedTxsGPU += committedTxsGPUBatch;
  // HeTM_stats_data.nbDroppedTxsGPU   += droppedTxsGPUBatch;
  // HeTM_stats_data.nbDroppedTxsGPU   += droppedTxsGPUBatch;

  // if (arg == (void*)-1 && HeTM_shared_data.policy == HETM_GPU_INV) { // TODO: stupid hack to avoid transfer the commits
  //   double ratioDropped = (double)HeTM_stats_data.nbBatchesFail / (double)HeTM_stats_data.nbBatches;
  //   double ratioCommitt = (double)HeTM_stats_data.nbBatchesSuccess / (double)HeTM_stats_data.nbBatches;
  //
  //   HeTM_stats_data.nbCommittedTxsGPU = HeTM_stats_data.nbTxsGPU*ratioCommitt;
  //   HeTM_stats_data.nbDroppedTxsGPU = HeTM_stats_data.nbTxsGPU*ratioDropped;
  // }

  isGetStatsDone = 1;
  __sync_synchronize();
}

static void offloadResetGPUState(void*)
{
  HeTM_reset_GPU_state(); // flags/locks
  isGPUResetDone = 1;
  __sync_synchronize();
}

#if HETM_LOG_TYPE == HETM_BMAP_LOG
static int launchCmpKernel(void *args)
{
  // TIMER_T t1, t2;
  // TIMER_READ(t1);
  HeTM_thread_s *threadData = (HeTM_thread_s*)args;
  size_t sizeCache;

  int nbThreadsX = 1024;
  int nbBlocksX;

  // --------------------------
  // COPY BMAP to GPU
#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
  nbBlocksX = (HeTM_shared_data.wsetLogSize + nbThreadsX-1) / (nbThreadsX);
  dim3 blocksCheck(nbBlocksX); // partition the stm_log by the different blocks
  dim3 threadsPerBlock(nbThreadsX); // each block has nbThreadsX threads

  unsigned char *cachedValues;
  memman_select("HeTM_cpu_wset_cache_confl");
  memman_zero_gpu(NULL);
  memman_zero_cpu(NULL);
  memman_select("HeTM_cpu_wset_cache");
  cachedValues = (unsigned char*)memman_get_cpu(NULL);
#else /* HETM_CMP_EXPLICIT */
  nbBlocksX = (HeTM_shared_data.rsetLogSize / sizeof(PR_GRANULE_T) + nbThreadsX-1) / (nbThreadsX);
  dim3 blocksCheck(nbBlocksX); // partition the stm_log by the different blocks
  dim3 threadsPerBlock(nbThreadsX); // each block has nbThreadsX threads

  memman_select("HeTM_cpu_wset");
#endif /* HETM_CMP_TYPE == HETM_CMP_COMPRESSED */

  // COPY BMAP TO GPU ////////////////////////////////////
  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)threadData->stream);
  memman_cpy_to_gpu(threadData->stream, &sizeCache);
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)threadData->stream);
  HeTM_stats_data.sizeCpyWSet += sizeCache;
  // cudaDeviceSynchronize(); // waits transfer to stop
  // COPY BMAP TO GPU ////////////////////////////////////

  // CHECK FOR CONFLICTS ////////////////////////////////////
  CUDA_EVENT_RECORD(threadData->cmpStartEvent, (cudaStream_t)threadData->stream);
#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
  int cacheNbThreads = 256;
  int cacheNbBlocks = HeTM_shared_data.wsetCacheSize / cacheNbThreads;
  if (HeTM_shared_data.wsetCacheSize % cacheNbThreads != 0) {
    cacheNbBlocks++;
  }
  HeTM_knl_checkTxBitmapCache<<<cacheNbBlocks, cacheNbThreads, 0, (cudaStream_t)threadData->stream>>>(
    (HeTM_knl_cmp_args_s){
      .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
      .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
  });
#else /* HETM_CMP_EXPLICIT */
  HeTM_knl_checkTxBitmap_Explicit<<<blocksCheck, threadsPerBlock, 0, (cudaStream_t)threadData->stream>>>(
    (HeTM_knl_cmp_args_s){
      .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
      .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
  });
#endif /* HETM_CMP_TYPE == HETM_CMP_COMPRESSED */
  CUDA_EVENT_RECORD(threadData->cmpStopEvent, (cudaStream_t)threadData->stream);

  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
    threadData->cpyWSetStopEvent);
  if (threadData->timeCpy > 0) {
    threadData->timeCpySum += threadData->timeCpy;
  }
  CUDA_EVENT_SYNCHRONIZE(threadData->cmpStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cmpStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCmp, threadData->cmpStartEvent,
    threadData->cmpStopEvent);
  threadData->timeCmpSum += threadData->timeCmp;
  // CHECK FOR CONFLICTS ////////////////////////////////////

#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
  cudaStreamSynchronize((cudaStream_t)threadData->stream);
  TIMER_T start_confl, end_confl;
  TIMER_READ(start_confl);
  HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(threadData->stream));
  if (HeTM_is_interconflict()) {
    cudaStream_t currentStream = (cudaStream_t)threadData->stream;
    cudaStream_t nextStream = (cudaStream_t)HeTM_memStream;
    cudaStream_t tmp;
    HeTM_reset_inter_confl_flag();

    // TODO: I'm here: what if the data-set is too long? it will distrube this phase

    memman_select("HeTM_cpu_wset_cache_confl");
    unsigned char *conflInGPU = (unsigned char*)memman_get_cpu(NULL);
    memman_cpy_to_cpu(currentStream, &sizeCache);
    // printf("POSSIBLE CONFLICT!!!\n");

    // memman_select("HeTM_cpu_wset");
    // memman_cpy_to_gpu(HeTM_memStream, &sizeWSet);
    // HeTM_stats_data.sizeCpyWSet += sizeWSet;
    memman_select("HeTM_cpu_wset");
    size_t sizeBmap, sizeToCpy = CACHE_GRANULE_SIZE;
    void *bitmapCPU = memman_get_cpu(&sizeBmap);
    void *bitmapGPU = memman_get_gpu(NULL);
    int startIdx;
    uintptr_t bitmapCPUptr, bitmapGPUptr;

    if (sizeToCpy > sizeBmap) {
      sizeToCpy = sizeBmap;
    }

    cudaStreamSynchronize(currentStream);
    for (int i = 0; i < sizeCache; ++i) {
      if (conflInGPU[i] != 1) continue;

      bitmapCPUptr = (uintptr_t)bitmapCPU;
      bitmapGPUptr = (uintptr_t)bitmapGPU;
      bitmapCPUptr += i * CACHE_GRANULE_SIZE;
      bitmapGPUptr += i * CACHE_GRANULE_SIZE;

      if (sizeToCpy > sizeBmap - i * CACHE_GRANULE_SIZE) {
        sizeToCpy = sizeBmap - i * CACHE_GRANULE_SIZE;
      }

      CUDA_CPY_TO_DEV_ASYNC((void*)bitmapGPUptr, (void*)bitmapCPUptr, sizeToCpy, currentStream);
      HeTM_stats_data.sizeCpyWSet += sizeToCpy;
      startIdx = i;
      break;
    }

    for (int i = startIdx; i < sizeCache; ++i) {
      if (conflInGPU[i] != 1) continue;
      // --------------------------
      // TODO: hacked!!! I'm copying only the memory pages
      // COPY BMAP to GPU (4096 positions)
      HeTM_knl_checkTxBitmap<<<16, 256, 0, currentStream>>>(
        (HeTM_knl_cmp_args_s){
          .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
          .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
        }, i * CACHE_GRANULE_SIZE);

      for (int j = i+1; j < sizeCache; ++j) {
        if (conflInGPU[j] != 1) continue;

        bitmapCPUptr = (uintptr_t)bitmapCPU;
        bitmapGPUptr = (uintptr_t)bitmapGPU;
        bitmapCPUptr += j * CACHE_GRANULE_SIZE;
        bitmapGPUptr += j * CACHE_GRANULE_SIZE;

        if (sizeToCpy > sizeBmap - j * CACHE_GRANULE_SIZE) {
          sizeToCpy = sizeBmap - j * CACHE_GRANULE_SIZE;
        }

        CUDA_CPY_TO_DEV_ASYNC((void*)bitmapGPUptr, (void*)bitmapCPUptr, sizeToCpy, nextStream);
        HeTM_stats_data.sizeCpyWSet += sizeToCpy;
        break;
      }

      HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(currentStream));
      if (HeTM_is_interconflict()) break;
      tmp = currentStream;
      currentStream = nextStream;
      nextStream = tmp;
    }
    cudaDeviceSynchronize();
    // -------------------------------------------------------------------------
  }
  TIMER_READ(end_confl);
  double timeCmpMS = TIMER_DIFF_SECONDS(start_confl, end_confl) * 1000.0f;
  threadData->timeCmpSum += timeCmpMS;
#else
  // --------------------------
  // COPY data-set to GPU (values written by the CPU)
  memman_select("HeTM_mempool_backup");
  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)HeTM_memStream);
  memman_cpy_to_gpu(HeTM_memStream, &sizeWSet); // TODO: put this in other stream
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)HeTM_memStream);
  HeTM_stats_data.sizeCpyWSet += sizeWSet;
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
    threadData->cpyWSetStopEvent);
  // --------------------------
#endif /* HETM_CMP_COMPRESSED */

  TIMER_T start_wset, end_wset;
  TIMER_READ(start_wset);
  HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(threadData->stream));
  if (!HeTM_is_interconflict()) {
    cudaDeviceSynchronize(); // waits transfer to stop
    // wait check kernel to end

#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
    // memcpy only the non-conflicting blocks (no conflicts is safe)
    unsigned char *conflInGPU;
    void *CPUDataset;
    void *GPUDataset;
    uintptr_t CPUptr, GPUptr;
    cudaStream_t currentStream = (cudaStream_t)threadData->stream;
    cudaStream_t nextStream = (cudaStream_t)HeTM_memStream;
    cudaStream_t tmp;

    memman_select("HeTM_cpu_wset_cache_confl");
    conflInGPU = (unsigned char*)memman_get_cpu(NULL);

    size_t sizeCPUMempool;
    memman_select("HeTM_mempool");
    CPUDataset = memman_get_cpu(&sizeCPUMempool);
    memman_select("HeTM_mempool_backup");
    GPUDataset = memman_get_gpu(NULL);

    // only copy the WSet where CPU and GPU conflict --> run kernel if at least 1 exists
    // merge contiguous pages
    for (int i = 0; i < sizeCache; ++i) {
      if (conflInGPU[i] != 1 || cachedValues[i] != 1) continue;
      // the two accessed the same page

      size_t size_cache_chunk = CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY;
      size_t size_to_cpy = size_cache_chunk;

      if (size_to_cpy > sizeCPUMempool) {
        size_to_cpy = sizeCPUMempool; // else copies beyond the buffer
      }

      CPUptr = (uintptr_t)CPUDataset;
      GPUptr = (uintptr_t)GPUDataset;
      CPUptr += i * size_cache_chunk;
      GPUptr += i * size_cache_chunk;

      CUDA_CPY_TO_DEV_ASYNC((void*)GPUptr, (void*)CPUptr,
        size_to_cpy, currentStream);

      // ---------
      HeTM_stats_data.sizeCpyWSet += CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY;
      // ---------

      HeTM_knl_writeTxBitmap<<<16, 256, 0, currentStream>>>(
        (HeTM_knl_cmp_args_s){
          .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
          .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
        }, i * CACHE_GRANULE_SIZE);
      tmp = currentStream;
      currentStream = nextStream;
      nextStream = tmp;

    }
    // cudaStreamSynchronize((cudaStream_t)threadData->stream);
    // override the remaining dataset

    memman_select("HeTM_mempool");
    size_t sizeOfGPUDataset;
    GPUDataset = memman_get_gpu(&sizeOfGPUDataset); // this is the right one

    for (int cont = 1, i = 0; i < sizeCache; i += cont) {
      if (cachedValues[i] != 1 || conflInGPU[i] == 1) continue; // not modified or conflict handled
      cont = 1;
      for (int j = i+1; j < sizeCache; ++j) {
        if (cachedValues[j] != 1 || conflInGPU[j] == 1) break; // done
        cont++;
      }

      CPUptr = (uintptr_t)CPUDataset;
      GPUptr = (uintptr_t)GPUDataset;
      CPUptr += i * CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY;
      GPUptr += i * CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY;

      size_t size_to_copy = cont * CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY;

      // ----------------
      HeTM_stats_data.sizeCpyWSet += size_to_copy;
      // ----------------

      if (i+cont == sizeCache) {
        size_t size_of_last = sizeOfGPUDataset % (CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY);
        size_to_copy = (cont-1) * CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY + size_of_last;
      }
      CUDA_CPY_TO_DEV_ASYNC((void*)GPUptr, (void*)CPUptr, size_to_copy, (cudaStream_t)HeTM_memStream);
    }

#else /* HETM_CMP_TYPE != HETM_CMP_COMPRESSED */
    HeTM_knl_writeTxBitmap<<<blocksCheck, threadsPerBlock, 0, (cudaStream_t)threadData->stream>>>(
      (HeTM_knl_cmp_args_s){
        .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
        .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
      }, 0);
#endif /* HETM_CMP_TYPE != HETM_CMP_COMPRESSED */
    NVTX_PUSH_RANGE("wait write", 3);
    cudaDeviceSynchronize(); // waits the kernel to stop
    TIMER_READ(end_wset);
    threadData->timeCpySum += TIMER_DIFF_SECONDS(start_wset, end_wset) * 1000.0f;

    NVTX_POP_RANGE();
  } else {
    // stops sending WSet GPU because
    // memman_stop_async_transfer();
    // TODO: instead of stop --> send the whole data-set (to replace with GPU dataset)
  }

#ifndef HETM_OVERLAP_CPY_BACK
  // TODO: zero cache set cache to 1
  // TODO: --> do a +1 on the bitmap and avoid memset to 0 (only once every 255 batches)
  if (HeTM_is_interconflict()) { // TODO: causes false conflicts
    memman_select("HeTM_cpu_wset");
    memman_zero_cpu(NULL); // this is slow!
  }
  memman_select("HeTM_cpu_wset_cache");
  memman_zero_cpu(NULL); // this is slow!
  memman_select("HeTM_cpu_wset_cache_confl");
  memman_zero_cpu(NULL); // this is slow!
  memman_zero_gpu(NULL); // this is slow!
#endif
  // HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(HeTM_memStream));
  // -----------------------------------------------
  // TIMER_READ(t2);
  // HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t1, t2); // must remove this time from the dataset
  return 0;
}
#endif /* if HETM_LOG_TYPE == HETM_BMAP_LOG */
