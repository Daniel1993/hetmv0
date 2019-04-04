#include "hetm.cuh"
#include "hetm-cmp-kernels.cuh"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "cuda_util.h"
#include "memman.h"
#include "knlman.h"

// --------------------
// Variables (global)
HeTM_shared_s HeTM_shared_data;
HeTM_statistics_s HeTM_stats_data;
hetm_pc_s *HeTM_offload_pc;
void *HeTM_memStream;
void *HeTM_memStream2;
// --------------------

// --------------------
// Functions (file local)
#if HETM_LOG_TYPE != HETM_BMAP_LOG
static void run_checkTxCompressed(knlman_callback_params_s params);
static void run_earlyCheckTxCompressed(knlman_callback_params_s params);
static void run_checkTxExplicit(knlman_callback_params_s params);
static void CUDART_CB cmpCallback(cudaStream_t event, cudaError_t status, void *data);
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */
// --------------------

#define MAX_FREE_NODES       0x800
static unsigned long freeNodesPtr;
static HeTM_async_req_s reqsBuffer[MAX_FREE_NODES];

int HeTM_init(HeTM_init_s init)
{
  HeTM_shared_data.isCPUEnabled = 1;
  HeTM_shared_data.isGPUEnabled = 1;
  HeTM_shared_data.nbCPUThreads = init.nbCPUThreads;
  HeTM_shared_data.nbGPUBlocks  = init.nbGPUBlocks;
  HeTM_shared_data.nbGPUThreads = init.nbGPUThreads;
  HeTM_shared_data.timeBudget   = init.timeBudget;
  HeTM_shared_data.policy       = init.policy;

  HeTM_set_GPU_status(HETM_BATCH_RUN);
  if (!init.isCPUEnabled) {
    // disable CPU usage
    HeTM_shared_data.nbCPUThreads = 0;
    HeTM_shared_data.isCPUEnabled = 0;
    HeTM_shared_data.nbThreads    = 1;
    barrier_init(HeTM_shared_data.GPUBarrier, 1); // only the controller thread
  } else if (!init.isGPUEnabled) {
    // disable GPU usage
    HeTM_set_GPU_status(HETM_IS_EXIT);
    HeTM_shared_data.isGPUEnabled = 0;
    HeTM_shared_data.nbThreads    = init.nbCPUThreads;
    barrier_init(HeTM_shared_data.GPUBarrier, init.nbCPUThreads); // 0 controller
  } else {
    // both on
    HeTM_shared_data.nbThreads    = init.nbCPUThreads + 1;
    barrier_init(HeTM_shared_data.GPUBarrier, init.nbCPUThreads+1); // 1 controller
  }

  malloc_or_die(HeTM_shared_data.threadsInfo, HeTM_shared_data.nbThreads);
  memset(HeTM_shared_data.threadsInfo, 0, HeTM_shared_data.nbThreads*sizeof(HeTM_thread_s));

#if HETM_LOG_TYPE != HETM_BMAP_LOG
  knlman_create("HeTM_checkTxCompressed", run_checkTxCompressed, 0);
  knlman_create("HeTM_earlyCheckTxCompressed", run_earlyCheckTxCompressed, 0);
  knlman_create("HeTM_checkTxExplicit", run_checkTxExplicit, 0);
  knlman_select("HeTM_checkTxExplicit");
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */

  PR_init(2); // inits PR-STM mutex array
  // TODO: init here STM too

  HeTM_offload_pc = hetm_pc_init(HETM_PC_BUFFER_SIZE);
  knlman_add_stream();
  HeTM_memStream = knlman_get_current_stream();
  knlman_add_stream();
  HeTM_memStream2 = knlman_get_current_stream();

  return 0;
}

int HeTM_destroy()
{
  barrier_destroy(HeTM_shared_data.GPUBarrier);
  free(HeTM_shared_data.threadsInfo);
  memman_select("HeTM_interConflFlag");
  memman_free_dual();
  HeTM_set_is_stop(1);
  HeTM_async_set_is_stop(1);
  knlman_select("HeTM_checkTxCompressed");
  knlman_destroy();
  knlman_select("HeTM_earlyCheckTxCompressed");
  knlman_destroy();
  knlman_select("HeTM_checkTxExplicit");
  knlman_destroy();
  hetm_pc_destroy(HeTM_offload_pc);

  return 0;
}

int HeTM_get_inter_confl_flag(void *stream, int doSync) {
  if (*HeTM_shared_data.hostInterConflFlag) {
    // does not need to copy again (conflict detected already)
    return 1;
  }
  memman_select("HeTM_interConflFlag");
  memman_cpy_to_cpu(stream, NULL, 1);
  if (doSync) { cudaStreamSynchronize((cudaStream_t)stream); }
  return *HeTM_shared_data.hostInterConflFlag;
}

int HeTM_reset_inter_confl_flag() {
  memman_select("HeTM_interConflFlag");
  memman_zero_dual(HeTM_memStream2);
  cudaStreamSynchronize((cudaStream_t)HeTM_memStream2);
  HeTM_set_is_interconflict(0);
  return 0;
}

int HeTM_sync_barrier()
{
  barrier_cross(HeTM_shared_data.GPUBarrier);
  return 0;
}

// TODO: only works for a "sufficiently" large buffer
void HeTM_async_request(HeTM_async_req_s req)
{
  HeTM_async_req_s *m_req;
  long idx = HETM_PC_ATOMIC_INC_PTR(freeNodesPtr, MAX_FREE_NODES);
  reqsBuffer[idx].fn   = req.fn;
  reqsBuffer[idx].args = req.args;
  m_req = &reqsBuffer[idx];
    // malloc_or_die(m_req, 1);
  // hetm_pc_consume(free_nodes, (void**)&m_req);
  // printf("[%i] got %p\n", HeTM_thread_data->id, m_req);
  hetm_pc_produce(HeTM_offload_pc, m_req);
}

void HeTM_free_async_request(HeTM_async_req_s *req)
{
  // TODO: is not releasing anything --> assume enough space
  // hetm_pc_produce(free_nodes, req);
  // free(req);
}

#if HETM_LOG_TYPE != HETM_BMAP_LOG
static void run_checkTxCompressed(knlman_callback_params_s params)
{
  dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
  dim3 threads(params.threads.x, params.threads.y, params.threads.z);
  cudaStream_t stream = (cudaStream_t)params.stream;
  HeTM_cmp_s *data = (HeTM_cmp_s*)params.entryObj;
  HeTM_thread_s *threadData = (HeTM_thread_s*)data->clbkArgs;

  CUDA_EVENT_RECORD(threadData->cmpStartEvent, stream);
  threadData->didCallCmp = 1;
#if HETM_LOG_TYPE == HETM_VERS2_LOG
  // TODO: init 32kB of shared memory (is it needed?)
  HeTM_knl_checkTxCompressed<<<blocks, threads, 0, stream>>>(data->knlArgs);
#else
  HeTM_knl_checkTxCompressed<<<blocks, threads, 0, stream>>>(data->knlArgs);
#endif
  CUDA_EVENT_RECORD(threadData->cmpStopEvent, stream);
  CUDA_CHECK_ERROR(cudaStreamAddCallback(
    stream, cmpCallback, data->clbkArgs, 0
  ), "");
}

static void run_earlyCheckTxCompressed(knlman_callback_params_s params)
{
  dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
  dim3 threads(params.threads.x, params.threads.y, params.threads.z);
  cudaStream_t stream = (cudaStream_t)params.stream;
  HeTM_cmp_s *data = (HeTM_cmp_s*)params.entryObj;
  HeTM_thread_s *threadData = (HeTM_thread_s*)data->clbkArgs;

  CUDA_EVENT_RECORD(threadData->cmpStartEvent, stream);
  // threadData->didCallCmp = 1;
#if HETM_LOG_TYPE == HETM_VERS2_LOG
  // TODO:
  exit(EXIT_FAILURE);
#else
  HeTM_knl_earlyCheckTxCompressed<<<blocks, threads, 0, stream>>>(data->knlArgs);
#endif
  CUDA_EVENT_RECORD(threadData->cmpStopEvent, stream);
  // CUDA_CHECK_ERROR(cudaStreamAddCallback(
  //   stream, cmpCallback, data->clbkArgs, 0
  // ), "");
}

static void run_checkTxExplicit(knlman_callback_params_s params)
{
  dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
  dim3 threads(params.threads.x, params.threads.y, params.threads.z);
  cudaStream_t stream = (cudaStream_t)params.stream;
  HeTM_cmp_s *data = (HeTM_cmp_s*)params.entryObj;
  HeTM_thread_s *threadData = (HeTM_thread_s*)data->clbkArgs;

  cudaFuncSetCacheConfig(HeTM_knl_checkTxExplicit, cudaFuncCachePreferL1);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

  CUDA_EVENT_RECORD(threadData->cmpStartEvent, stream);
  HeTM_knl_checkTxExplicit<<<blocks, threads, 0, stream>>>(data->knlArgs);
  CUDA_EVENT_RECORD(threadData->cmpStopEvent, stream);
  CUDA_EVENT_ELAPSED_TIME(&threadData->timeCmp, threadData->cmpStartEvent,
    threadData->cmpStopEvent);
  threadData->timeCmpSum += threadData->timeCmp;

  CUDA_CHECK_ERROR(cudaStreamAddCallback(
      stream, cmpCallback, threadData, 0
    ), "");
}

static void CUDART_CB cmpCallback(cudaStream_t event, cudaError_t status, void *data)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)data;

  if(status != cudaSuccess) { // TODO: Handle error
    printf("CMP failed! >>> %s <<<\n", cudaGetErrorString(status));
    // TODO: exit application
  }

  TIMER_T now;
  TIMER_READ(now);
  double timeTaken = TIMER_DIFF_SECONDS(threadData->beforeCmpKernel, now);
  // printf(" --- send to %i wait cmp delay=%fms\n", threadData->id, timeTaken*1000);

  threadData->timeCmpKernels += timeTaken;
  threadData->isCmpDone = 1;
  __sync_synchronize(); // cmpCallback is called from a different thread
}
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */
