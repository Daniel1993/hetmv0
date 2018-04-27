#include "hetm.cuh"
#include "pr-stm-wrapper.cuh"
#include "hetm-timer.h"
#include "hetm-cmp-kernels.cuh"

#if HETM_LOG_TYPE == HETM_ADDR_LOG
#include "hetm-cmp-kernels.cuh" // must run apply kernel
#endif

#include <list>
#include <mutex>

extern std::mutex HeTM_statsMutex; // defined in hetm-threading-cpu
pr_tx_args_s HeTM_pr_args; // used only by the CUDA-control thread (TODO: PR-STM only)

static std::list<HeTM_callback> beforeGPU, afterGPU;
static int isAfterCmpDone = 0;
static int isGetStatsDone = 0;
static int isDatasetSyncDone = 0;

static void initThread(int id, void *data);
static void exitThread(int id, void *data);
static void prepareSyncDataset();
static void updateStatisticsAfterBatch(void*);

// executed in other thread
static void offloadSyncDatasetAfterBatch(void*);
static void offloadGetPRStats(void*);
static void offloadAfterCmp(void*);

#if HETM_LOG_TYPE == HETM_BMAP_LOG
static int launchCmpKernel(void*);
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */

#define WAIT_ON_FLAG(flag) while(!flag) pthread_yield(); flag = 0

void HeTM_gpu_thread()
{
  int threadId = HeTM_thread_data->id;
  HeTM_callback callback = HeTM_thread_data->callback;
  void *clbkArgs = HeTM_thread_data->args;

  TIMER_T t1, t2, t3;

  // TODO: check order
  PR_createStatistics(&HeTM_pr_args);
  initThread(threadId, clbkArgs);

  HeTM_sync_barrier();

  TIMER_READ(t1);

  while (CONTINUE_COND) {
    // TODO: prepare syncBatchPolicy
    prepareSyncDataset();

    if (HeTM_get_GPU_status() != HETM_IS_EXIT) {
      callback(threadId, clbkArgs);
    }

    PR_waitKernel();

    TIMER_READ(t2);
    // TODO: I'm taking GPU time in PR-STM
    HeTM_stats_data.timePRSTM += TIMER_DIFF_SECONDS(t1, t2);

    HeTM_set_GPU_status(HETM_BATCH_DONE); // notifies
    HeTM_sync_barrier(); // Blocks and waits comparison kernel to end
    TIMER_READ(t3);

    HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t2, t3);

    HeTM_async_request((HeTM_async_req_s){
      .args = HeTM_thread_data,
      .fn = offloadAfterCmp,
    });
    HeTM_async_request((HeTM_async_req_s){
      .args = HeTM_thread_data,
      .fn = offloadSyncDatasetAfterBatch,
    });
    HeTM_async_request((HeTM_async_req_s){
      .args = NULL,
      .fn = updateStatisticsAfterBatch,
    });

    if (!HeTM_is_stop()) {
      HeTM_set_GPU_status(HETM_BATCH_RUN);
    } else { // Times up
      HeTM_set_GPU_status(HETM_IS_EXIT);
    }

    // TODO: if there is no conflict launch immediately the kernel
    // while copying assynchronously the dataset to the CPU, then
    // wake up CPU threads

    // WAIT_ON_FLAG(isAfterCmpDone);

    WAIT_ON_FLAG(isDatasetSyncDone); // only need to wait on this
    HeTM_sync_barrier(); // wakes the threads, GPU will re-run

    TIMER_READ(t1);
    HeTM_stats_data.timeAfterCMP += TIMER_DIFF_SECONDS(t3, t1);

    HETM_DEB_THRD_GPU(" --- End of batch (%li sucs, %li fail)",
      HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.nbBatchesFail);
  }

  // TODO: this was per iteration
  // This will run sync
  HeTM_async_request((HeTM_async_req_s){
    .args = (void*)HeTM_thread_data,
    .fn = offloadGetPRStats,
  });
  HeTM_async_request((HeTM_async_req_s){
    .args = (void*)-1,
    .fn = updateStatisticsAfterBatch,
  });
  WAIT_ON_FLAG(isGetStatsDone);
  WAIT_ON_FLAG(isAfterCmpDone);

  exitThread(threadId, clbkArgs);
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

void initThread(int id, void *data)
{
  cudaEventCreate(&HeTM_thread_data->cmpStartEvent);
  cudaEventCreate(&HeTM_thread_data->cmpStopEvent);
  cudaEventCreate(&HeTM_thread_data->cpyWSetStartEvent);
  cudaEventCreate(&HeTM_thread_data->cpyWSetStopEvent);
  cudaEventCreate(&HeTM_thread_data->cpyDatasetStartEvent);
  cudaEventCreate(&HeTM_thread_data->cpyDatasetStopEvent);

  for (auto it = beforeGPU.begin(); it != beforeGPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

void exitThread(int id, void *data)
{
  HeTM_statsMutex.lock();
  HeTM_stats_data.totalTimeCpyWSet += HeTM_thread_data->timeCpySum;
  HeTM_stats_data.totalTimeCmp += HeTM_thread_data->timeCmpSum;
  HeTM_stats_data.totalTimeCpyDataset += HeTM_thread_data->timeCpyDatasetSum;
  HeTM_statsMutex.unlock();

  HeTM_stats_data.timeGPU = PR_kernelTime;
  for (auto it = afterGPU.begin(); it != afterGPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
  HETM_DEB_THRD_GPU("Time copy dataset %10fms - Time cpy WSet %10fms - Time cmp %10fms\n",
    HeTM_thread_data->timeCpyDatasetSum, HeTM_thread_data->timeCpySum,
    HeTM_thread_data->timeCmpSum);
}

static void prepareSyncDataset() { /* TODO */ }

static void offloadSyncDatasetAfterBatch(void *args)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)args;
  size_t datasetCpySize;

  HETM_DEB_THRD_GPU("Syncing dataset ...");
  if (!HeTM_is_interconflict()) { // Successful execution
    // TODO: take statistic on this
#if HETM_LOG_TYPE == HETM_ADDR_LOG
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
#endif

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

  cudaDeviceSynchronize(); // TODO: now I'm waiting this to complete
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyDatasetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyDatasetStopEvent);

  // TODO: get this statistic
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
  isGetStatsDone = 1;
  __sync_synchronize();
}

static void offloadAfterCmp(void *threadData)
{
#if HETM_LOG_TYPE == HETM_BMAP_LOG
  launchCmpKernel(threadData); // if BMAP blocks and run the kernel
#endif
  // gets the flag
  HeTM_set_is_interconflict(HeTM_get_inter_confl_flag());

  isAfterCmpDone = 1;
  __sync_synchronize();
}

static void updateStatisticsAfterBatch(void *arg)
{
  long committedTxsCPUBatch = 0, committedTxsGPUBatch = 0;
  long txsNonBlocking = 0;
  long droppedTxsCPUBatch = 0, droppedTxsGPUBatch = 0;
  int idGPUThread = HeTM_shared_data.nbCPUThreads; // the last one

  if (arg == NULL) { // TODO: stupid hack to avoid transfer the commits
    HeTM_stats_data.nbBatches++;
    if (HeTM_is_interconflict()) {
      HeTM_stats_data.nbBatchesFail++;
    } else {
      HeTM_stats_data.nbBatchesSuccess++;
    }
  }

  if (HeTM_is_interconflict() && HeTM_shared_data.policy == HETM_CPU_INV) {
    // drop CPU TXs
    int i;
for (i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
      droppedTxsCPUBatch += HeTM_shared_data.threadsInfo[i].curNbTxs;
      HeTM_shared_data.threadsInfo[i].curNbTxs = 0;
    }
  } else {
    int i;
for (i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
      committedTxsCPUBatch += HeTM_shared_data.threadsInfo[i].curNbTxs;
      // TODO: what if CPU INV?
      txsNonBlocking += HeTM_shared_data.threadsInfo[i].curNbTxsNonBlocking;
      HeTM_shared_data.threadsInfo[i].curNbTxs = 0;
      HeTM_shared_data.threadsInfo[i].curNbTxsNonBlocking = 0;
    }
  }

  if (HeTM_is_interconflict() && HeTM_shared_data.policy == HETM_GPU_INV) {
    // drop GPU TXs
    droppedTxsGPUBatch += HeTM_shared_data.threadsInfo[idGPUThread].curNbTxs;
    HeTM_shared_data.threadsInfo[idGPUThread].curNbTxs = 0;
  } else {
    committedTxsGPUBatch += HeTM_shared_data.threadsInfo[idGPUThread].curNbTxs;
    HeTM_shared_data.threadsInfo[idGPUThread].curNbTxs = 0;
  }

  HeTM_stats_data.nbTxsCPU += droppedTxsCPUBatch + committedTxsCPUBatch;
  HeTM_stats_data.nbCommittedTxsCPU += committedTxsCPUBatch;
  HeTM_stats_data.txsNonBlocking    += txsNonBlocking;
  HeTM_stats_data.nbDroppedTxsCPU   += droppedTxsCPUBatch;
  HeTM_stats_data.nbTxsGPU += droppedTxsGPUBatch + committedTxsGPUBatch;
  HeTM_stats_data.nbCommittedTxsGPU += committedTxsGPUBatch;
  HeTM_stats_data.nbDroppedTxsGPU   += droppedTxsGPUBatch;
  HeTM_stats_data.nbDroppedTxsGPU   += droppedTxsGPUBatch;

  if (arg == (void*)-1 && HeTM_shared_data.policy == HETM_GPU_INV) { // TODO: stupid hack to avoid transfer the commits
    double ratioDropped = (double)HeTM_stats_data.nbBatchesFail / (double)HeTM_stats_data.nbBatches;
    double ratioCommitt = (double)HeTM_stats_data.nbBatchesSuccess / (double)HeTM_stats_data.nbBatches;

    HeTM_stats_data.nbCommittedTxsGPU = HeTM_stats_data.nbTxsGPU*ratioCommitt;
    HeTM_stats_data.nbDroppedTxsGPU = HeTM_stats_data.nbTxsGPU*ratioDropped;
  }

  // IMPORTANT!!!
  HeTM_reset_GPU_state(); // flags/locks
}

#if HETM_LOG_TYPE == HETM_BMAP_LOG
static int launchCmpKernel(void *args)
{
  // TIMER_T t1, t2;
  // TIMER_READ(t1);
  HeTM_thread_s *threadData = (HeTM_thread_s*)args;
  size_t sizeWSet;
  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, (cudaStream_t)HeTM_memStream);
  memman_select("HeTM_cpu_wset");
  memman_cpy_to_gpu(HeTM_memStream, &sizeWSet);
  HeTM_stats_data.sizeCpyWSet += sizeWSet;
  memman_select("HeTM_mempool_backup");
  memman_cpy_to_gpu(HeTM_memStream, &sizeWSet); // TODO: put this in other stream
  HeTM_stats_data.sizeCpyWSet += sizeWSet;
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, (cudaStream_t)HeTM_memStream);

  cudaDeviceSynchronize(); // waits transfer to stop
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpy, threadData->cpyWSetStartEvent,
    threadData->cpyWSetStopEvent);
  if (threadData->timeCpy > 0) { // TODO: bug
    threadData->timeCpySum += threadData->timeCpy;
  }

  // -----------------------------------------------
  //Calc number of blocks
  int nbThreadsX = 1024;
  int bo = (HeTM_shared_data.wsetLogSize + nbThreadsX-1) / (nbThreadsX);
  dim3 blocksCheck(bo); // partition the stm_log by the different blocks
  dim3 threadsPerBlock(nbThreadsX); // each block has nbThreadsX threads

  // TODO: follow the API for the other kernels!
  CUDA_EVENT_RECORD(threadData->cmpStartEvent, (cudaStream_t)HeTM_memStream);
  HeTM_knl_checkTxBitmap<<<blocksCheck, threadsPerBlock>>>((HeTM_knl_cmp_args_s){
    .sizeWSet = (int)HeTM_shared_data.wsetLogSize,
    .sizeRSet = (int)HeTM_shared_data.rsetLogSize,
  });
  CUDA_EVENT_RECORD(threadData->cmpStopEvent, (cudaStream_t)HeTM_memStream);
  cudaDeviceSynchronize(); // waits the kernel to stop
  CUDA_EVENT_SYNCHRONIZE(threadData->cmpStartEvent);
  CUDA_EVENT_SYNCHRONIZE(threadData->cmpStopEvent);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCmp, threadData->cmpStartEvent,
    threadData->cmpStopEvent);
  threadData->timeCmpSum += threadData->timeCmp;
  memman_select("HeTM_cpu_wset");
  memman_zero_cpu(NULL);
  HeTM_set_is_interconflict(HeTM_get_inter_confl_flag());
  // -----------------------------------------------
  // TIMER_READ(t2);
  // HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t1, t2); // must remove this time from the dataset
  return 0;
}
#endif /* if HETM_LOG_TYPE == HETM_BMAP_LOG */
