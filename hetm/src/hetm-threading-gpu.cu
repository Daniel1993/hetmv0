#include "hetm.cuh"
#include "pr-stm-wrapper.cuh"
#include "hetm-timer.h"
#include "hetm-cmp-kernels.cuh"
#include "knlman.h"

#include <list>
#include <mutex>

#define MINIMUM_DtD_CPY 4194304
#define MINIMUM_DtH_CPY 262144
#define MINIMUM_HtD_CPY 262144

#define RUN_ASYNC(_func, _args) \
  HeTM_async_request((HeTM_async_req_s){ \
    .args = (void*)_args, \
    .fn = _func, \
  }) \
//

extern std::mutex HeTM_statsMutex; // defined in hetm-threading-cpu
pr_tx_args_s HeTM_pr_args; // used only by the CUDA-control thread (TODO: PR-STM only)

static std::list<HeTM_callback> beforeGPU;
static std::list<HeTM_callback> afterGPU;
static std::list<HeTM_callback> beforeBatch;
static std::list<HeTM_callback> afterBatch;
static std::list<HeTM_callback> beforeKernel;
static std::list<HeTM_callback> afterKernel;
static HeTM_callback choose_policy;
static int isAfterCmpDone = 0;
static int isGetStatsDone = 0;
static int isGetPRStatsDone = 0;
static int isDatasetSyncDone = 0;
static int isGPUResetDone = 0;

static int isInterConflt = 0;

static long afterBatch_batchCount = 0;

static void initThread(int id, void *data);
static void exitThread(int id, void *data);
static inline void prepareSyncDataset();
static inline void runBatch();
static inline void waitBatchEnd();
static inline void afterWaitBatchEnd();
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

// TODO: add a runBeforeKernel and runAfterKernel
// rational: now a batch/run is a number of kernel executions
static inline void runAfterKernel(int id, void *data);
static inline void runBeforeKernel(int id, void *data);

// executed in other thread
static void beforeSyncDataset(void*);
static void offloadSyncDatasetAfterBatch(void*);
static void offloadWaitDataset(void*);
static void offloadGetPRStats(void*);
static void offloadAfterCmp(void*);
static void offloadResetGPUState(void*);

static int isWaitingCpyBack = 0;
// static thread_local HeTM_thread_s *tmp_threadData; // TODO: HETM_OVERLAP_CPY_BACK

#if HETM_LOG_TYPE == HETM_BMAP_LOG
static TIMER_T bmapBlockedStart, bmapBlockedEnd;
static int launchCmpKernel(void*);
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */

#define WAIT_ON_FLAG(flag) while(!(flag)) pthread_yield(); flag = 0

#ifdef HETM_OVERLAP_CPY_BACK
static int isOverlappingKernel = 0;

void HeTM_gpu_thread()
{
  int threadId = HeTM_thread_data->id;
  void *clbkArgs = HeTM_thread_data->args;
  // int isOtherKernelStarted = 0; // TODO: cannot overlap more than 1 kernel
  TIMER_T t1, t2, t3, t1woCpy;

  PR_createStatistics(&HeTM_pr_args);
  initThread(threadId, clbkArgs);

  HeTM_sync_barrier();

  TIMER_READ(t1);
  afterBatch_batchCount = 1;
  isInterConflt = 0;

  // first one here, then before notify the CPU
  runBeforeBatch(threadId, (void*)HeTM_thread_data);

loop_gpu_batch:

    if (isOverlappingKernel /* && HeTM_shared_data.isCPUEnabled == 1 */) {
      ///---
      // if (isInterConflt) {
      //   notifyCPUNextBatch(); // do not wait the D->D copy
      // }
      ///---

      // backups the dataset
      size_t datasetCpySize = 0;
      if (HeTM_stats_data.timeDtD == 0) {
        HeTM_stats_data.timeDtD += 1e-200;
      } else {
        float elapsedTime;
        CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cpyWSetStartEvent);
        CUDA_EVENT_SYNCHRONIZE(HeTM_thread_data->cpyWSetStopEvent);
        CUDA_EVENT_ELAPSED_TIME(&elapsedTime, HeTM_thread_data->cpyWSetStartEvent,
          HeTM_thread_data->cpyWSetStopEvent);
        HeTM_stats_data.timeDtD += elapsedTime;
      }
      CUDA_EVENT_RECORD(HeTM_thread_data->cpyWSetStartEvent, (cudaStream_t)HeTM_memStream);
      if (isInterConflt) {
        // recovers the main (D->H done with the shadow copy)
        HeTM_mempool_cpy_to_gpu_main(&datasetCpySize, afterBatch_batchCount);
      } else {
        // overides the main
        HeTM_mempool_cpy_to_gpu_backup(&datasetCpySize, afterBatch_batchCount);
      }
      CUDA_EVENT_RECORD(HeTM_thread_data->cpyWSetStopEvent, (cudaStream_t)HeTM_memStream);
      // TODO: does not count the D->D copy
      // HeTM_stats_data.sizeCpyDataset += datasetCpySize;
    }

    prepareSyncDataset();

    doGPUStateReset();
    WAIT_ON_FLAG(isGPUResetDone); // forces GPU reset before start

    // printf("before first runBatch\n");
    runBeforeKernel(threadId, (void*)HeTM_thread_data);
    cudaStreamSynchronize((cudaStream_t)HeTM_memStream); // waits D->D cpy
    runBatch(); // only the first kernel is not an overlapping kernel

    // ------------------------------------------------------------------------
    HeTM_reset_inter_confl_flag(); // TODO: where to reset this
    // ------------------------------------------------------------------------

    if (isOverlappingKernel) {
      // the next kernel must overlap with the copy of data
      RUN_ASYNC(beforeSyncDataset, HeTM_thread_data);
      RUN_ASYNC(offloadSyncDatasetAfterBatch, HeTM_thread_data);

      // wait copy and notify CPU
      RUN_ASYNC(offloadWaitDataset, HeTM_thread_data);
      // repeat kernels until time budget expired
      // do more batches while the copy is not done

      // TODO: GPU kernels are NOT overlaping with the copy
      // WAIT_ON_FLAG(isDatasetSyncDone);

      while (!isDatasetSyncDone) {
        // wait previous kernel and launch another one
        waitBatchEnd();
        runAfterKernel(threadId, (void*)HeTM_thread_data);
        runBeforeKernel(threadId, (void*)HeTM_thread_data);
        runBatch();
      }
      ///---
      // if (!isInterConflt) { // from previous batch
      ///---
        isDatasetSyncDone = 0;
        notifyCPUNextBatch();
      ///---
      // }
      ///---
      TIMER_READ(t3);
      HeTM_stats_data.timeAfterCMP += TIMER_DIFF_SECONDS(t2, t3);
    }
    TIMER_READ(t1woCpy);

    afterBatch_batchCount = *hetm_batchCount;

    // repeat kernels until time budget expired
    waitBatchEnd();
    runAfterKernel(threadId, (void*)HeTM_thread_data);
    TIMER_READ(t2);
    if(TIMER_DIFF_SECONDS(t1woCpy, t2) < HeTM_shared_data.timeBudget) {
      do {
        // TODO: this runs before EACH batch in a round
        // but the afterBatch method only runs once
        runBeforeKernel(threadId, (void*)HeTM_thread_data);
        runBatch();
        waitBatchEnd();
        runAfterKernel(threadId, (void*)HeTM_thread_data);
        TIMER_READ(t2);
      } while(TIMER_DIFF_SECONDS(t1woCpy, t2) < HeTM_shared_data.timeBudget && !isInterConflt);
    }
    if (isInterConflt) HeTM_stats_data.nbEarlyValAborts++;
    afterWaitBatchEnd();
    // isOtherKernelStarted = 0;
    // printf("after last runBatch --> %f s\n", HeTM_shared_data.timeBudget);

    TIMER_READ(t2);
    double timeLastBatch = TIMER_DIFF_SECONDS(t1, t2);
    HeTM_stats_data.timePRSTM += timeLastBatch;

    waitCMPEnd();
    isInterConflt = HeTM_is_interconflict();
    if (isInterConflt) {
      HeTM_stats_data.timeAbortedBatches += timeLastBatch;
    }
    TIMER_READ(t1);
    HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t2, t1);

// ---------------------
    mergeDataset();
// ---------------------
    checkIsExit();
// ---------------------
    runAfterBatch(threadId, (void*)HeTM_thread_data);

    HETM_DEB_THRD_GPU(" --- End of batch (%li sucs, %li fail)",
      HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.nbBatchesFail);
  // }

  isOverlappingKernel = 1; // all but the first are overlapping kernels
  if (CONTINUE_COND) {
    runBeforeBatch(threadId, (void*)HeTM_thread_data);
    // notifyCPUNextBatch(); // only after the D->H copy is done
    goto loop_gpu_batch;
  }

  // End: does the last D->H copy
  RUN_ASYNC(beforeSyncDataset, HeTM_thread_data);
  RUN_ASYNC(offloadSyncDatasetAfterBatch, HeTM_thread_data);
  // wait copy and notify CPU
  if (HeTM_shared_data.isCPUEnabled == 1) {
    RUN_ASYNC(offloadWaitDataset, HeTM_thread_data);
    WAIT_ON_FLAG(isDatasetSyncDone);
  }

  cudaDeviceSynchronize(); // before terminating the benchmark

  notifyCPUNextBatch();
  TIMER_READ(t3);
  HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t2, t3);

  // TODO: this was per iteration
  // This will run sync
  RUN_ASYNC(offloadGetPRStats, HeTM_thread_data);
  RUN_ASYNC(updateStatisticsAfterBatch, -1);

// ---------------------
  WAIT_ON_FLAG(isWaitingCpyBack);
  WAIT_ON_FLAG(isAfterCmpDone); // avoids a warning in the compiler
  WAIT_ON_FLAG(isGetStatsDone);
  WAIT_ON_FLAG(isGetPRStatsDone);
// ---------------------

  exitThread(threadId, clbkArgs);
}
#endif /* HETM_OVERLAP_CPY_BACK */

#ifndef HETM_OVERLAP_CPY_BACK
void HeTM_gpu_thread()
{
  int threadId = HeTM_thread_data->id;
  void *clbkArgs = HeTM_thread_data->args;

  TIMER_T t1, t2, t3/*, t1WCpy*/;

  // TODO: check order
  PR_createStatistics(&HeTM_pr_args);
  initThread(threadId, clbkArgs);

  HeTM_sync_barrier();

  TIMER_READ(t1);
  // TIMER_READ(t1WCpy);
  afterBatch_batchCount = 1;

  // // first one here, then before notify the CPU
  runBeforeBatch(threadId, (void*)HeTM_thread_data);

loop_gpu_batch:
  // while (CONTINUE_COND) {
    // TODO: prepare syncBatchPolicy
    prepareSyncDataset();

    doGPUStateReset();
    WAIT_ON_FLAG(isGPUResetDone); // forces GPU reset before start

    afterBatch_batchCount = *hetm_batchCount;
    HeTM_reset_inter_confl_flag(); // TODO: where to reset this
    TIMER_READ(t2);
    if(TIMER_DIFF_SECONDS(t1, t2) < HeTM_shared_data.timeBudget) {
      do {
        // TODO: this runs before EACH batch in a round
        // but the afterBatch method only runs once
        runBeforeKernel(threadId, (void*)HeTM_thread_data);
        runBatch();
        waitBatchEnd();
        runAfterKernel(threadId, (void*)HeTM_thread_data);
        TIMER_READ(t2);
        // printf("Time in seconds = %f s\n", TIMER_DIFF_SECONDS(t1, t2));
      } while(TIMER_DIFF_SECONDS(t1, t2) < HeTM_shared_data.timeBudget && !isInterConflt);
    }
    if (isInterConflt) HeTM_stats_data.nbEarlyValAborts++;
    afterWaitBatchEnd();
    TIMER_READ(t2);
    // TODO: I'm taking GPU time in PR-STM
    double timeLastBatch = TIMER_DIFF_SECONDS(t1, t2);
    HeTM_stats_data.timePRSTM += timeLastBatch;

    waitCMPEnd();
    isInterConflt = HeTM_is_interconflict();
    if (isInterConflt) {
      HeTM_stats_data.timeAbortedBatches += timeLastBatch;
    }
    TIMER_READ(t3);
     // TODO: let the next batch begin if dataset is still not copied

    HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t2, t3);

// ---------------------
    mergeDataset();
// ---------------------
    checkIsExit();

// ---------------------
    // TIMER_READ(t1WCpy);
    RUN_ASYNC(beforeSyncDataset, HeTM_thread_data);
    RUN_ASYNC(offloadSyncDatasetAfterBatch, HeTM_thread_data);
    RUN_ASYNC(offloadWaitDataset, HeTM_thread_data);
    WAIT_ON_FLAG(isDatasetSyncDone);

// ---------------------
    runAfterBatch(threadId, (void*)HeTM_thread_data);

    TIMER_READ(t1);
    HeTM_stats_data.timeAfterCMP += TIMER_DIFF_SECONDS(t3, t1);

    HETM_DEB_THRD_GPU(" --- End of batch (%li sucs, %li fail)",
      HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.nbBatchesFail);
  // }
  if (CONTINUE_COND) {
    runBeforeBatch(threadId, (void*)HeTM_thread_data);
    notifyCPUNextBatch();
    goto loop_gpu_batch;
  }

  cudaDeviceSynchronize(); // before terminating the benchmark

  notifyCPUNextBatch();

  // TODO: this was per iteration
  // This will run sync
  RUN_ASYNC(offloadGetPRStats, HeTM_thread_data);
  RUN_ASYNC(updateStatisticsAfterBatch, -1);

// ---------------------
  WAIT_ON_FLAG(isWaitingCpyBack);
  WAIT_ON_FLAG(isAfterCmpDone); // avoids a warning in the compiler
  WAIT_ON_FLAG(isGetStatsDone);
  WAIT_ON_FLAG(isGetPRStatsDone);
// ---------------------

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
  HeTM_shared_data.batchCount = 1;
  hetm_batchCount = &HeTM_shared_data.batchCount;
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
  // also setup some memoru for PR-STM
  memman_select("HeTM_mempool_bmap");
  memman_cpy_to_gpu(HeTM_memStream2, NULL, 1);
  memman_bmap_s *bmap = (memman_bmap_s*)memman_get_cpu(NULL);
  CUDA_CHECK_ERROR(cudaMemsetAsync(bmap->dev, 0, bmap->div, (cudaStream_t)HeTM_memStream2), "");

  memman_select("HeTM_mempool_backup_bmap");
	memman_cpy_to_gpu(HeTM_memStream2, NULL, 1);
  memman_bmap_s *bmapBackup = (memman_bmap_s*)memman_get_cpu(NULL);
  CUDA_CHECK_ERROR(cudaMemsetAsync(bmapBackup->dev, 0, bmapBackup->div, (cudaStream_t)HeTM_memStream2), "");

  /* Fail on multiple allocs (memman_select needed) */
	// memman_alloc_dual("HeTM_gpuLog", sizeof(HeTM_GPU_log_s), MEMMAN_THRLC);
	// memman_select("HeTM_gpuLog");
  // memman_cpy_to_gpu(HeTM_memStream, NULL, 1); // TODO: this is copied on each batch

  for (auto it = beforeBatch.begin(); it != beforeBatch.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }

  cudaStreamSynchronize((cudaStream_t)HeTM_memStream2);
}

static inline void runAfterBatch(int id, void *data)
{
  HeTM_shared_data.batchCount++;
  if ((HeTM_shared_data.batchCount & 0xff) == 0) {
    HeTM_shared_data.batchCount++;
  }
  __sync_synchronize();
  for (auto it = afterBatch.begin(); it != afterBatch.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

static inline void runBeforeKernel(int id, void *data)
{
  for (auto it = beforeKernel.begin(); it != beforeKernel.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

static inline void runAfterKernel(int id, void *data)
{
  for (auto it = afterKernel.begin(); it != afterKernel.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }

  isInterConflt = HeTM_is_interconflict();

  PR_useNextStream(&HeTM_pr_args); // swaps the current stream
}

// host is backup, dev is mempool (both are in device)
// static void backupTheMempool_fn(void *hostPtr, void *devPtr, size_t cpySize)
// {
//   // TODO: D->D GPU+CPU WS
//   cudaMemcpyAsync(hostPtr, devPtr, cpySize, cudaMemcpyDeviceToDevice,
//     (cudaStream_t)HeTM_memStream);
// }

// static void recoverTheMempool_fn(void *hostPtr, void *devPtr, size_t cpySize)
// {
//   // TODO: D->D GPU+CPU WS
//   cudaMemcpyAsync(devPtr, hostPtr, cpySize, cudaMemcpyDeviceToDevice,
//     (cudaStream_t)(tmp_threadData->stream));
// }

// TODO: HETM_OVERLAP_CPY_BACK
// static void dropGPU_fn(void *hostPtr, void *devPtr, size_t cpySize)
// {
//   // TODO: D->D GPU+CPU WS
//   CUDA_CPY_TO_DEV_ASYNC(devPtr, hostPtr, cpySize, (cudaStream_t)HeTM_memStream);
// }

// static void dropCPU_fn(void *hostPtr, void *devPtr, size_t cpySize)
// {
//   // TODO: D->D GPU+CPU WS
//   CUDA_CPY_TO_HOST_ASYNC(hostPtr, devPtr, cpySize, (cudaStream_t)HeTM_memStream);
// }

// static void cpyDtoD(void *mainPtr, void *backupPtr, size_t cpySize)
// {
//   CUDA_CPY_DtD_ASYNC(mainPtr, backupPtr, cpySize, (cudaStream_t)HeTM_memStream);
// }


static inline void waitBatchEnd()
{
  PR_waitKernel();
  // Removed deviceSync from here
}

static inline void afterWaitBatchEnd()
{
  memman_select("HeTM_mempool");
  memman_bmap_s *bMap = (memman_bmap_s*) memman_get_bmap(NULL);
  cudaMemcpy(HeTM_shared_data.devBackupMemPoolBmap, HeTM_shared_data.devMemPoolBmap,
    bMap->div * sizeof(char), cudaMemcpyDeviceToDevice);
  memman_cpy_to_cpu_buffer_bitmap(HeTM_memStream2); // gets what the GPU wrote
  memman_select("HeTM_mempool_backup");
  memman_cpy_to_cpu_buffer_bitmap(HeTM_memStream2); // gets what the GPU wrote (copy)
}

static inline void waitCMPEnd()
{
  HeTM_set_GPU_status(HETM_BATCH_DONE); // notifies
  __sync_synchronize();

  // waits threads to stop doing validation (VERS)
  do {
    COMPILER_FENCE();
  } while(HeTM_shared_data.threadsWaitingSync != HeTM_shared_data.nbCPUThreads);
  // at this point the CPU should be blocked

#if HETM_LOG_TYPE == HETM_BMAP_LOG
  TIMER_READ(bmapBlockedStart);
#endif /* HETM_LOG_TYPE == HETM_BMAP_LOG */

  // TODO: CPU_INV not working
// #if HETM_LOG_TYPE == HETM_VERS_LOG
//   if (HeTM_shared_data.policy == HETM_CPU_INV) {
//
//     memman_select("HeTM_mempool");
//     size_t sizeofMempool;
//     void *ptrMempool = memman_get_gpu(&sizeofMempool);
//     memman_select("HeTM_mempool_backup");
//     void *ptrMempoolBackup = memman_get_gpu(NULL);
//
//     memman_smart_cpy(bmapBackup->ptr, bmapBackup->gran, ptrMempoolBackup,
//       ptrMempool, sizeofMempool, backupTheMempool_fn);
//   }
// #endif /* HETM_LOG_TYPE == HETM_VERS_LOG */

  HeTM_sync_barrier(); // Blocks and waits comparison kernel to end
  HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(HeTM_memStream2, 1));
}

static inline void mergeDataset()
{
  RUN_ASYNC(offloadAfterCmp, HeTM_thread_data); // fetches the conflict flag
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

static inline void notifyCPUNextBatch()
{
  HeTM_sync_barrier(); // wakes the threads, GPU will re-run
#if HETM_LOG_TYPE == HETM_BMAP_LOG
  TIMER_READ(bmapBlockedEnd);
  HeTM_stats_data.timeBlocking += TIMER_DIFF_SECONDS(bmapBlockedStart, bmapBlockedEnd) * HeTM_shared_data.nbCPUThreads;
#endif /* HETM_LOG_TYPE == HETM_BMAP_LOG */
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

int HeTM_before_kernel(HeTM_callback req)
{
  beforeKernel.push_back(req);
  return 0;
}

int HeTM_after_kernel(HeTM_callback req)
{
  afterKernel.push_back(req);
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
  // cudaMemcpy(bmapBackup->dev, bmap->dev, bmap->div, cudaMemcpyDeviceToDevice);
  // memman_select("HeTM_mempool_backup");
  // memman_cpy_to_cpu_buffer_bitmap();
  isWaitingCpyBack = 1;
  __sync_synchronize();
}

static void offloadSyncDatasetAfterBatch(void *args)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)args;
  size_t datasetCpySize = 0;

  // tmp_threadData = threadData;

#if HETM_CMP_TYPE == HETM_CMP_DISABLED || defined (HETM_DISABLE_WS)
  // if (HeTM_shared_data.isCPUEnabled == 0) {
    // HeTM_mempool_cpy_to_cpu(&datasetCpySize);
    memman_select("HeTM_mempool");
    void *host = memman_get_cpu(&datasetCpySize);
    void *dev = memman_get_gpu(NULL);
#ifdef HETM_OVERLAP_CPY_BACK
    cudaMemcpyAsync(host, dev, datasetCpySize, cudaMemcpyDeviceToHost, (cudaStream_t)HeTM_memStream);
#else
    // sync
    cudaMemcpy(host, dev, datasetCpySize, cudaMemcpyDeviceToHost);
#endif
    HeTM_stats_data.sizeCpyDataset += datasetCpySize;
    return;
  // }
#endif /* HETM_CMP_TYPE == HETM_CMP_DISABLED */

  HETM_DEB_THRD_GPU("Syncing dataset ...");
  if (!isInterConflt) { // Successful execution

    // GPU-only enters always here

    // Transfers the GPU WSet to CPU
    CUDA_EVENT_RECORD(threadData->cpyDatasetStartEvent, (cudaStream_t)HeTM_memStream);

    HeTM_mempool_cpy_to_cpu(&datasetCpySize, afterBatch_batchCount);
    HeTM_stats_data.sizeCpyDataset += datasetCpySize;

    CUDA_EVENT_RECORD(threadData->cpyDatasetStopEvent, (cudaStream_t)HeTM_memStream);
#ifndef HETM_OVERLAP_CPY_BACK // The overlap does this before launching the next batch
    if (HeTM_shared_data.isCPUEnabled == 1) {
      if (HeTM_stats_data.timeDtD == 0) {
        HeTM_stats_data.timeDtD += 1e-200;
      } else {
        float elapsedTime;
        CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
        CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
        CUDA_EVENT_ELAPSED_TIME(&elapsedTime, threadData->cpyWSetStartEvent, threadData->cpyWSetStopEvent);
        HeTM_stats_data.timeDtD += elapsedTime;
      }
      CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)HeTM_memStream);
      HeTM_mempool_cpy_to_gpu_backup(&datasetCpySize, afterBatch_batchCount);
      CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)HeTM_memStream);
    }
#endif /* HETM_OVERLAP_CPY_BACK */

  } else { // conflict detected

    CUDA_EVENT_RECORD(threadData->cpyDatasetStartEvent, (cudaStream_t)HeTM_memStream);
    if (HeTM_shared_data.policy == HETM_GPU_INV) {
#ifndef HETM_OVERLAP_CPY_BACK // The overlap does this before launching the next batch
      // Drop GPU policy
      // overide the main with the backup
      if (HeTM_shared_data.isCPUEnabled == 1) {
        if (HeTM_stats_data.timeDtD == 0) {
          HeTM_stats_data.timeDtD += 1e-200;
        } else {
          float elapsedTime;
          CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
          CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
          CUDA_EVENT_ELAPSED_TIME(&elapsedTime, threadData->cpyWSetStartEvent, threadData->cpyWSetStopEvent);
          HeTM_stats_data.timeDtD += elapsedTime;
        }
        // backups
        CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)HeTM_memStream);
        HeTM_mempool_cpy_to_gpu_main(&datasetCpySize, afterBatch_batchCount);
        CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)HeTM_memStream);
      }
      // TODO: does not count the D->D copy
      // HeTM_stats_data.sizeCpyDataset += datasetCpySize;
#endif /* HETM_OVERLAP_CPY_BACK */
    }

    // TODO: CPU_INV not working
//     if (HeTM_shared_data.policy == HETM_CPU_INV) {
//
// #if HETM_LOG_TYPE == HETM_VERS_LOG
//       // On VERS need to roll-back applied CPU writes
//       memman_select("HeTM_mempool");
//       size_t sizeofMempool;
//       void *ptrMempool = memman_get_gpu(&sizeofMempool);
//       void *CPUmempool = memman_get_cpu(&sizeofMempool);
//       memman_select("HeTM_mempool_backup");
//       void *ptrMempoolBackup = memman_get_gpu(NULL);
//       memman_select("HeTM_mempool_backup_bmap");
//       memman_bmap_s *bmapWS = (memman_bmap_s*)memman_get_cpu(NULL);
//
//       // Restores the mempool in the GPU
//       memman_smart_cpy(bmapWS->ptr, bmapWS->gran, ptrMempoolBackup,
//         ptrMempool, sizeofMempool, recoverTheMempool_fn);
//
//       // At the same time can copy from backup to CPU (restore in the CPU)
//       datasetCpySize += memman_smart_cpy(bmapWS->ptr, bmapWS->gran, CPUmempool,
//         ptrMempoolBackup, sizeofMempool, dropCPU_fn);
//       HeTM_stats_data.sizeCpyDataset += datasetCpySize;
//
// #else /* HETM_LOG_TYPE == HETM_BMAP_LOG */
//
//       memman_select("HeTM_mempool");
//       size_t sizeofMempool;
//       void *ptrMempool = memman_get_gpu(&sizeofMempool);
//       void *CPUmempool = memman_get_cpu(&sizeofMempool);
//       memman_select("HeTM_mempool_backup_bmap");
//       memman_bmap_s *bmapWS = (memman_bmap_s*)memman_get_cpu(NULL);
//
//       datasetCpySize += memman_smart_cpy(bmapWS->ptr, bmapWS->gran, CPUmempool,
//         ptrMempool, sizeofMempool, dropCPU_fn);
//       HeTM_stats_data.sizeCpyDataset += datasetCpySize;
// #endif /* HETM_LOG_TYPE == HETM_VERS_LOG */
//
//     }
    // if (HeTM_shared_data.policy == HETM_CPU_INV)
    CUDA_EVENT_RECORD(threadData->cpyDatasetStopEvent, (cudaStream_t)HeTM_memStream);
  }

  // TODO: what was this doing?
  // memman_select("HeTM_mempool_backup_bmap");
  // memman_bmap_s *bmapBackup = (memman_bmap_s*)memman_get_cpu(NULL);
  // memset(bmapBackup->ptr, 0, bmapBackup->div);
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
  // HeTM_thread_s *threadData = (HeTM_thread_s*)argPtr;

  // PR_retrieveIO(&HeTM_pr_args);
  // threadData->curNbTxs = PR_nbCommits;
  // HeTM_stats_data.nbAbortsGPU += PR_nbAborts;
  // PR_resetStatistics(&HeTM_pr_args);
  isGetPRStatsDone = 1;
  __sync_synchronize();
}

static void offloadAfterCmp(void *args)
{
#if HETM_LOG_TYPE == HETM_BMAP_LOG
  if (HeTM_shared_data.isGPUEnabled && HeTM_shared_data.isCPUEnabled) {
    launchCmpKernel(args); // if BMAP blocks and run the kernel
  }
#else /* HETM_LOG_TYPE != HETM_BMAP_LOG */

HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(HeTM_memStream2, 1));

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
    if (isInterConflt) {
      HeTM_stats_data.nbBatchesFail++;
    } else {
      HeTM_stats_data.nbBatchesSuccess++;
    }
  }

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

  choose_policy(0, arg); // choose the policy for the next batch

  HeTM_stats_data.nbTxsCPU += droppedTxsCPUBatch + committedTxsCPUBatch;
  HeTM_stats_data.nbCommittedTxsCPU += committedTxsCPUBatch;
  HeTM_stats_data.txsNonBlocking    += txsNonBlocking;
  HeTM_stats_data.nbDroppedTxsCPU   += droppedTxsCPUBatch;

  HeTM_stats_data.nbTxsGPU += PR_nbCommitsSinceCheckpoint;
  HeTM_stats_data.nbAbortsGPU += PR_nbAbortsSinceCheckpoint;
  if (isInterConflt) {
    HeTM_stats_data.nbDroppedTxsGPU   += PR_nbCommitsSinceCheckpoint;
  } else {
    HeTM_stats_data.nbCommittedTxsGPU += PR_nbCommitsSinceCheckpoint;
  }

  PR_checkpointAbortsCommits();

  isGetStatsDone = 1;
  __sync_synchronize();
}

static void offloadResetGPUState(void*)
{
  HeTM_reset_GPU_state(afterBatch_batchCount); // flags/locks
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
  size_t sizeOfGPUDataset;

  int nbThreadsX = 1024;
  int nbBlocksX;

  int threadsPerBlockBitmapGran = 512;
  int blocksBitmapGran = DEFAULT_BITMAP_GRANULARITY / threadsPerBlockBitmapGran;

  TIMER_T start_confl, end_confl;
  TIMER_T start_wset, end_wset;

  unsigned char *conflInGPU;
  void *CPUDataset;
  void *GPUDataset;
  void *GPUDatasetBackup;
  volatile uintptr_t CPUptr;
  volatile uintptr_t GPUptr;
  cudaStream_t currentStream = (cudaStream_t)threadData->stream;
  cudaStream_t nextStream = (cudaStream_t)HeTM_memStream;
  cudaStream_t tmp;

  double timeCmpMS;

  size_t sizeCPUMempool;
  memman_select("HeTM_mempool");
  CPUDataset = memman_get_cpu(&sizeCPUMempool);
  GPUDataset = memman_get_gpu(NULL);
  memman_select("HeTM_mempool_backup");
  GPUDatasetBackup = memman_get_gpu(NULL);

  sizeOfGPUDataset = sizeCPUMempool;

  TIMER_READ(start_confl);

  unsigned char *cachedValues;
  memman_select("HeTM_cpu_wset_cache_confl"); // not zeroed now
  memman_select("HeTM_cpu_wset_cache");
  cachedValues = (unsigned char*)memman_get_cpu(&sizeCache);
  // --------------------------
  // COPY BMAP to GPU
#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
  nbBlocksX = (HeTM_shared_data.wsetLogSize + nbThreadsX-1) / (nbThreadsX);
  dim3 blocksCheck(nbBlocksX); // partition the stm_log by the different blocks
  dim3 threadsPerBlock(nbThreadsX); // each block has nbThreadsX threads
#else /* HETM_CMP_EXPLICIT */
  nbBlocksX = (HeTM_shared_data.rsetLogSize / sizeof(PR_GRANULE_T) + nbThreadsX-1) / (nbThreadsX);
  dim3 blocksCheck(nbBlocksX); // partition the stm_log by the different blocks
  dim3 threadsPerBlock(nbThreadsX); // each block has nbThreadsX threads

  memman_select("HeTM_cpu_wset");
#endif /* HETM_CMP_TYPE == HETM_CMP_COMPRESSED */

  // TODO: test ------------------------------------
  for (int i = 0; i < sizeCache; ++i) {
    ((unsigned char*)stm_wsetCPUCache)[i] = ((unsigned char*)stm_wsetCPUCache_x64)[i << 6];
  }
  // TODO: test ------------------------------------

  // COPY BMAP TO GPU ////////////////////////////////////
  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)threadData->stream);
  memman_cpy_to_gpu(threadData->stream, NULL, afterBatch_batchCount);
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)threadData->stream);
  HeTM_stats_data.sizeCpyWSet += sizeCache;
  // cudaDeviceSynchronize(); // waits transfer to stop
  // COPY BMAP TO GPU ////////////////////////////////////

  // CHECK FOR CONFLICTS ////////////////////////////////////
  // CUDA_EVENT_RECORD(threadData->cmpStartEvent, (cudaStream_t)threadData->stream);
#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
  // TODO: comparison MUST be cache with cache for best performance

  // int cacheNbThreads = 1024;
  // int cacheNbBlocks = HeTM_shared_data.wsetLogSize / cacheNbThreads;
  // if (HeTM_shared_data.wsetLogSize % cacheNbThreads != 0) {
  //   cacheNbBlocks++;
  // }

  int cacheNbThreads = 256;
  int cacheNbBlocks = HeTM_shared_data.nbChunks / cacheNbThreads;
  if (HeTM_shared_data.nbChunks % cacheNbThreads != 0) {
    cacheNbBlocks++;
  }
  // printf("Launch top level kernel (%i, %i) sizeWset=%lu\n", cacheNbBlocks, cacheNbThreads, HeTM_shared_data.wsetLogSize);
  HeTM_knl_checkTxBitmapCache<<<cacheNbBlocks, cacheNbThreads, 0, (cudaStream_t)threadData->stream>>>(
    (HeTM_knl_cmp_args_s){
      .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
      .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
      .idCPUThr = 0,
      .batchCount = (unsigned char)HeTM_shared_data.batchCount,
  });
#else /* HETM_CMP_EXPLICIT */
  HeTM_knl_checkTxBitmap_Explicit<<<blocksCheck, threadsPerBlock, 0, (cudaStream_t)threadData->stream>>>(
    (HeTM_knl_cmp_args_s){
      .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
      .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
      .idCPUThr = 0,
      .batchCount = (unsigned char)HeTM_shared_data.batchCount,
  });
#endif /* HETM_CMP_TYPE == HETM_CMP_COMPRESSED */

  // CUDA_EVENT_RECORD(threadData->cmpStopEvent, (cudaStream_t)threadData->stream);

  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
    threadData->cpyWSetStopEvent);
  if (threadData->timeCpy > 0) {
    HeTM_stats_data.timeMemCpySum += threadData->timeCpy;
  }
  // CUDA_EVENT_SYNCHRONIZE(threadData->cmpStartEvent);
  // CUDA_EVENT_SYNCHRONIZE(threadData->cmpStopEvent);
  // CUDA_EVENT_ELAPSED_TIME(&threadData->timeCmp, threadData->cmpStartEvent,
  //   threadData->cmpStopEvent);
  // threadData->timeCmpSum += threadData->timeCmp;
  // CHECK FOR CONFLICTS ////////////////////////////////////

  cudaStreamSynchronize((cudaStream_t)threadData->stream);
  HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(threadData->stream, 1));

  memman_select("HeTM_cpu_wset_cache_confl2");
  unsigned char *conflInGPUw = (unsigned char*)memman_get_cpu(NULL);
  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, NULL);
  memman_cpy_to_cpu(NULL, NULL, afterBatch_batchCount);
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, NULL);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
    threadData->cpyWSetStopEvent);
  if (threadData->timeCpy > 0) {
    HeTM_stats_data.timeMemCpySum += threadData->timeCpy;
  }

  memman_select("HeTM_cpu_wset_cache_confl");
  conflInGPU = (unsigned char*)memman_get_cpu(NULL);
  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, NULL);
  memman_cpy_to_cpu(NULL, &sizeCache, afterBatch_batchCount);
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, NULL);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
    threadData->cpyWSetStopEvent);
  if (threadData->timeCpy > 0) {
    threadData->timeMemCpySum += threadData->timeCpy;
  }

  TIMER_READ(end_confl);
  timeCmpMS = TIMER_DIFF_SECONDS(start_confl, end_confl) * 1000.0f;
  threadData->timeCmpSum += timeCmpMS;

  TIMER_READ(start_wset);

  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)HeTM_memStream);
  // Copy safe CPU data into the GPU (i.e., chunks that the GPU did not wrote)
  for (int cont = 1, i = 0; i < sizeCache; i += cont) {

    // TODO: there are repeated comparisons!

    if (conflInGPUw[i] ||
        cachedValues[i] != (unsigned char)HeTM_shared_data.batchCount
        || conflInGPU[i] == (unsigned char)HeTM_shared_data.batchCount)
      continue; // not modified or conflict handled

    const int NbitsToJump = 4;
    cont = 1;
    for (int j = i+1; j < sizeCache; j += NbitsToJump) {

      int foundAtLeastA1 = 0;
      for (int k = 0; k < NbitsToJump; ++k) {
        if (k+j >= sizeCache) {
          break; // end of the cache
        }
        if (!conflInGPUw[j+k] &&
            cachedValues[j+k] == (unsigned char)HeTM_shared_data.batchCount
            && conflInGPU[j+k] != (unsigned char)HeTM_shared_data.batchCount) {
          foundAtLeastA1 = 1;
          break; // found a 1
        }
      }

      if (!foundAtLeastA1) break;

      // if (conflInGPUw[j] ||
      //     cachedValues[j] != (unsigned char)HeTM_shared_data.batchCount
      //     || conflInGPU[j] == (unsigned char)HeTM_shared_data.batchCount) {
      //   break; // done
      // }

      cont += j+NbitsToJump > sizeCache ? sizeCache-j : NbitsToJump;
    }

    const size_t chunkSize = CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY;

    // TODO: moved cache granularity from accounts to bytes
    CPUptr = (uintptr_t)CPUDataset;
    GPUptr = (uintptr_t)GPUDataset;
    CPUptr += (uintptr_t)i * chunkSize; //* PR_LOCK_GRANULARITY;
    GPUptr += (uintptr_t)i * chunkSize; //* PR_LOCK_GRANULARITY;

    size_t size_to_copy = cont * chunkSize; //* PR_LOCK_GRANULARITY;

    // ----------------
    // HeTM_stats_data.sizeCpyWSet += size_to_copy;
    HeTM_stats_data.sizeCpyWSetCPUData += size_to_copy;
    // ----------------

    if (i+cont >= sizeCache) {
      size_t size_of_last = sizeOfGPUDataset % (chunkSize);
      size_to_copy = (cont-1) * chunkSize + size_of_last;
    }

    // assert(CPUptr >= (uintptr_t)CPUDataset && CPUptr < CPUptr+sizeCPUMempool);
    // assert(GPUptr >= (uintptr_t)GPUDataset && GPUptr < GPUptr+sizeCPUMempool);

    CUDA_CPY_TO_DEV_ASYNC(GPUptr, CPUptr, size_to_copy, (cudaStream_t)HeTM_memStream);
  }
  // TODO: gaps will enter here
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)HeTM_memStream);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
    threadData->cpyWSetStopEvent);
  if (threadData->timeCpy > 0) {
    threadData->timeMemCpySum += threadData->timeCpy;
  }

  TIMER_READ(end_wset);
  threadData->timeCpySum += TIMER_DIFF_SECONDS(start_wset, end_wset) * 1000.0f;

  TIMER_READ(start_confl);

#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
  if (HeTM_is_interconflict()) {
    cudaStream_t currentStream = (cudaStream_t)threadData->stream;
    cudaStream_t nextStream = (cudaStream_t)HeTM_memStream;
    cudaStream_t tmp;
    HeTM_reset_inter_confl_flag();

    static int possibleConflicts = 0;
    // printf(" --- POSSIBLE CONFLICT - %i!!!\n", ++possibleConflicts);

    // memman_select("HeTM_cpu_wset");
    // memman_cpy_to_gpu(HeTM_memStream, &sizeWSet);
    // HeTM_stats_data.sizeCpyWSet += sizeWSet;
    memman_select("HeTM_cpu_wset");
    size_t sizeBmap, sizeToCpy = CACHE_GRANULE_SIZE;
    void *bitmapCPU = memman_get_cpu(&sizeBmap);
    void *bitmapGPU = memman_get_gpu(NULL);
    uintptr_t bitmapCPUptr, bitmapGPUptr;

    cudaStreamSynchronize(currentStream);
    int blocksBitmapGran_coalesced; // coallesce contiguos conflicts
    for (int i = 0; i < sizeCache; ++i) {
      if (conflInGPU[i] != (unsigned char)HeTM_shared_data.batchCount) continue;

      // Discover how much to copy
      int j;
      for (j = i+1; j < sizeCache; ++j) {
        if (conflInGPU[j] != (unsigned char)HeTM_shared_data.batchCount) break;
      }

      bitmapCPUptr = (uintptr_t)bitmapCPU;
      bitmapGPUptr = (uintptr_t)bitmapGPU;
      bitmapCPUptr += i * CACHE_GRANULE_SIZE;
      bitmapGPUptr += i * CACHE_GRANULE_SIZE;

      sizeToCpy = (CACHE_GRANULE_SIZE * (j - i));

      if (sizeToCpy > sizeBmap - i * CACHE_GRANULE_SIZE) {
        sizeToCpy = sizeBmap - i * CACHE_GRANULE_SIZE;
      }

      // TODO: not easy to get this one
      CUDA_CPY_TO_DEV_ASYNC((void*)bitmapGPUptr, (void*)bitmapCPUptr, sizeToCpy, currentStream);
      HeTM_stats_data.sizeCpyWSet += sizeToCpy;
      HeTM_stats_data.nbBitmapConflicts++;

      // launch the kernel in one stream
      blocksBitmapGran_coalesced = sizeToCpy / threadsPerBlockBitmapGran;
      if (sizeToCpy % threadsPerBlockBitmapGran > 0)  blocksBitmapGran_coalesced++;

      cudaStreamSynchronize(currentStream);
      HeTM_knl_checkTxBitmap<<<blocksBitmapGran_coalesced, threadsPerBlockBitmapGran, 0, currentStream>>>(
        (HeTM_knl_cmp_args_s){
          .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
          .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
          .idCPUThr = 0,
          .batchCount = (unsigned char)HeTM_shared_data.batchCount,
        }, i * CACHE_GRANULE_SIZE);

      tmp = currentStream;
      currentStream = nextStream;
      nextStream = tmp;
      break;
    }

    cudaDeviceSynchronize();
    HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(currentStream, 1));
    // -------------------------------------------------------------------------
  }
#else
  // --------------------------
  // COPY data-set to GPU (values written by the CPU)
  size_t sizeWSet;
  memman_select("HeTM_mempool_backup");
  // CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)HeTM_memStream);
  memman_cpy_to_gpu(HeTM_memStream, &sizeWSet, afterBatch_batchCount); // TODO: put this in other stream
  // CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)HeTM_memStream);
  HeTM_stats_data.sizeCpyWSet += sizeWSet;
  // CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
  // CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
  // CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
  //   threadData->cpyWSetStopEvent);
  // --------------------------
#endif /* HETM_CMP_COMPRESSED */

  TIMER_READ(end_confl);
  timeCmpMS = TIMER_DIFF_SECONDS(start_confl, end_confl) * 1000.0f;
  threadData->timeCmpSum += timeCmpMS;

  TIMER_READ(start_wset);

  HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(threadData->stream, 1));
  if (!HeTM_is_interconflict()) {
    cudaDeviceSynchronize(); // waits transfer to stop
    // wait check kernel to end

#if HETM_CMP_TYPE == HETM_CMP_COMPRESSED
    // memcpy only the non-conflicting blocks (no conflicts is safe)

    memman_select("HeTM_cpu_wset_cache_confl");
    conflInGPU = (unsigned char*)memman_get_cpu(NULL);

    // only copy the WSet where CPU and GPU conflict --> run kernel if at least 1 exists
    // merge contiguous pages
    // for (int i = 0; i < sizeCache; ++i) {
    //   printf("%i", (int) conflInGPUw[i]);
    // }
    // printf("\n");
    for (int i = 0; i < sizeCache; ++i) {
      if (!conflInGPUw[i] ||
          (conflInGPU[i] != (unsigned char)HeTM_shared_data.batchCount ||
          cachedValues[i] != (unsigned char)HeTM_shared_data.batchCount)) continue;
      // the two accessed the same page

      size_t size_cache_chunk = CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY;
      size_t size_to_cpy = size_cache_chunk;

      if (size_to_cpy + i * size_cache_chunk > sizeCPUMempool) {
        size_to_cpy = sizeCPUMempool - i * size_cache_chunk; // else copies beyond the buffer
      }

      CPUptr = (uintptr_t)CPUDataset;
      GPUptr = (uintptr_t)GPUDatasetBackup;
      CPUptr += i * size_cache_chunk;
      GPUptr += i * size_cache_chunk;

      CUDA_CPY_TO_DEV_ASYNC((void*)GPUptr, (void*)CPUptr,
        size_to_cpy, currentStream);

      // ---------
      // HeTM_stats_data.sizeCpyWSet += size_to_cpy;
      HeTM_stats_data.sizeCpyWSetCPUData += size_to_cpy;
      HeTM_stats_data.nbCPUdataConflicts++;
      // ---------

      HeTM_knl_writeTxBitmap<<<blocksBitmapGran, threadsPerBlockBitmapGran, 0, currentStream>>>(
        (HeTM_knl_cmp_args_s){
          .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
          .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
          .idCPUThr = 0,
          .batchCount = (unsigned char)HeTM_shared_data.batchCount,
        }, i * CACHE_GRANULE_SIZE);
      tmp = currentStream;
      currentStream = nextStream;
      nextStream = tmp;
    }
    // cudaStreamSynchronize((cudaStream_t)threadData->stream);
    // override the remaining dataset

#else /* HETM_CMP_TYPE != HETM_CMP_COMPRESSED */
    HeTM_knl_writeTxBitmap<<<blocksCheck, threadsPerBlock, 0, (cudaStream_t)threadData->stream>>>(
      (HeTM_knl_cmp_args_s){
        .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
        .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
        .idCPUThr = 0,
        .batchCount = (unsigned char)HeTM_shared_data.batchCount,
      }, 0);
#endif /* HETM_CMP_TYPE != HETM_CMP_COMPRESSED */
    NVTX_PUSH_RANGE("wait write", 3);
    cudaDeviceSynchronize(); // waits the kernel to stop

    NVTX_POP_RANGE();
  } else {
    // stops sending WSet GPU because
    // memman_stop_async_transfer();
    // TODO: instead of stop --> send the whole data-set (to replace with GPU dataset)
    // --> what the GPU damaged is already copied by default
    //     TODO: use the union bitmap in HeTM_cpu_wset_cache_confl3




    // memman_select("HeTM_mempool");
    // size_t sizeMemPool;
    // void *mempoolGPU = memman_get_gpu(&sizeMemPool);
    // void *mempoolCPU = memman_get_cpu(NULL);
    // memman_select("HeTM_cpu_wset_cache");
    // cachedValues = (unsigned char*)memman_get_cpu(NULL);
    // size_t cpySize = memman_smart_cpy((char*)stm_wsetCPUCache, (char)HeTM_shared_data.batchCount,
    //   CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY, mempoolCPU, mempoolGPU, sizeMemPool, dropGPU_fn);
    // // TODO: empty cache
    // // printf("drop cpy %zu sizeMemPool=%zu\n", cpySize, sizeMemPool);
    // HeTM_stats_data.sizeCpyWSetCPUData += cpySize;



    for (int i = 0; i < sizeCache; ++i) {
      if (!conflInGPUw[i] ||
          (conflInGPU[i] != (unsigned char)HeTM_shared_data.batchCount ||
          cachedValues[i] != (unsigned char)HeTM_shared_data.batchCount)) continue;
      // the two accessed the same page

      size_t size_cache_chunk = CACHE_GRANULE_SIZE * PR_LOCK_GRANULARITY;
      size_t size_to_cpy = size_cache_chunk;

      if (size_to_cpy + i * size_cache_chunk > sizeCPUMempool) {
        size_to_cpy = sizeCPUMempool - i * size_cache_chunk; // else copies beyond the buffer
      }

      CPUptr = (uintptr_t)CPUDataset;
      GPUptr = (uintptr_t)GPUDataset;
      CPUptr += i * size_cache_chunk;
      GPUptr += i * size_cache_chunk;

      CUDA_CPY_TO_DEV_ASYNC((void*)GPUptr, (void*)CPUptr,
        size_to_cpy, currentStream); // aborted --> override

      // ---------
      // HeTM_stats_data.sizeCpyWSet += size_to_cpy;
      HeTM_stats_data.sizeCpyWSetCPUData += size_to_cpy;
      // ---------
    }
  }
  TIMER_READ(end_wset);
  threadData->timeCpySum += TIMER_DIFF_SECONDS(start_wset, end_wset) * 1000.0f;

#ifndef HETM_OVERLAP_CPY_BACK
  // TODO: zero cache set cache to 1
  // TODO: --> do a +1 on the bitmap and avoid memset to 0 (only once every 255 batches)
  // if (HeTM_is_interconflict()) { // TODO: causes false conflicts
  if ((HeTM_shared_data.batchCount & 0xff) == 0xff) { // at some round --> reset
    memman_select("HeTM_cpu_wset");
    memman_zero_cpu(NULL); // this is slow!
    memman_select("HeTM_cpu_wset_cache");
    memman_zero_cpu(NULL); // this is slow!
    memman_select("HeTM_cpu_wset_cache_confl");
    memman_zero_cpu(NULL); // this is slow!
  }
  // memman_select("HeTM_cpu_wset_cache_confl");
  // memman_zero_gpu(NULL); // this is slow!
  memman_select("HeTM_cpu_wset_cache_confl2");
  // memman_zero_cpu(NULL); // this is slow!
  memman_zero_gpu(NULL); // this is slow!
#endif
  // HeTM_set_is_interconflict(HeTM_get_inter_confl_flag(HeTM_memStream2, 1));
  // -----------------------------------------------
  // TIMER_READ(t2);
  // HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t1, t2); // must remove this time from the dataset
  return 0;
}
#endif /* if HETM_LOG_TYPE == HETM_BMAP_LOG */
