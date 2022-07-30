#ifndef HETM_H_GUARD_
#define HETM_H_GUARD_

#include "hetm-utils.h"
#include "hetm-producer-consumer.h"
#include "hetm-log.h"
#include "hetm-timer.h"

#include <cuda.h>
#include <cuda_profiler_api.h>

#define HETM_MAX_CMP_KERNEL_LAUNCHES 4
#define HETM_PC_BUFFER_SIZE          0x100
#define CONTINUE_COND (!HeTM_is_stop() || HeTM_get_GPU_status() != HETM_IS_EXIT)

#define HETM_CMP_DISABLED   0
#define HETM_CMP_COMPRESSED 1
#define HETM_CMP_EXPLICIT   2
#ifndef HETM_CMP_TYPE
#define HETM_CMP_TYPE HETM_CMP_DISABLED
#endif

typedef enum {
  HETM_BATCH_RUN  = 0,
  HETM_BATCH_DONE = 1,
  HETM_IS_EXIT    = 2,
  HETM_GPU_IDLE   = 3
} HETM_GPU_STATE;

typedef enum {
  HETM_CMP_OFF     = 0, // during normal execution
  HETM_CMP_STARTED = 1, // execution ended --> start comparing
  HETM_CPY_ASYNC   = 2, // copy while running TXs
  HETM_CMP_ASYNC   = 3, // CMP while running TXs
  HETM_DONE_ASYNC  = 4, // after cpy and cmp
  HETM_CMP_BLOCK   = 5  // block CPU while the compare completes
} HETM_CMP_STATE;

typedef enum {
  HETM_CPU_INV,
  HETM_GPU_INV
} HETM_INVALIDATE_POLICY;

// ------------- callbacks
// the GPU callback launches the kernel inside of it
typedef void(*HeTM_callback)(int, void*);
typedef void(*HeTM_request)(void*);
typedef int(*HeTM_map_tid_to_core)(int);
// -------------

typedef struct HeTM_thread_
{
  // configuration
  int id;
  HeTM_callback callback;
  void *args;
  pthread_t thread;
  int didCallCmp;

  // algorithm specific
  HETM_LOG_T *wSetLog;
  volatile int isCmpDone;
  volatile int isCmpVoid;
  int countCpy;
  int nbCpyDone;
  volatile int isCpyDone;

  cudaEvent_t cmpStartEvent, cmpStopEvent;
  float timeCmp;
  double timeCmpSum;

  cudaEvent_t cpyWSetStartEvent, cpyWSetStopEvent;
  cudaEvent_t cpyLogChunkStartEvent[STM_LOG_BUFFER_SIZE];
  cudaEvent_t cpyLogChunkStopEvent[STM_LOG_BUFFER_SIZE];
  unsigned long logChunkEventCounter, logChunkEventStore;
  float timeCpy;
  double timeCpySum;
  double timeMemCpySum;

  cudaEvent_t cpyDatasetStartEvent, cpyDatasetStopEvent;
  float timeCpyDataset;
  double timeCpyDatasetSum;

  size_t curWSetSize;
  HETM_CMP_STATE statusCMP;
  int nbCmpLaunches;
  long curNbTxs /* count all */, curNbTxsNonBlocking /* ADDR|VERS */;
  void *stream; // cudaStream_t

  volatile int isFirstChunk;
  size_t emptySpaceFirstChunk;

  TIMER_T backoffBegTimer;
  TIMER_T backoffEndTimer;
  TIMER_T blockingEndTimer;
  TIMER_T beforeCpyLogs;
  TIMER_T beforeCmpKernel;

  double timeLogs;
  double timeCmpKernels;
  double timeBackoff;
  double timeBlocked;

  int targetCopyNb;
  int curCopyNb;
  int doHardLogCpy;
  int isCopying;
  HETM_LOG_T truncated;
} HeTM_thread_s;

// sigleton
typedef struct HeTM_shared_
{
  // configuration
  volatile HETM_INVALIDATE_POLICY policy;
  int nbCPUThreads, nbGPUBlocks, nbGPUThreads;
  int isCPUEnabled, isGPUEnabled, nbThreads;

  // algorithm specific
  HETM_GPU_STATE statusGPU;
  int stopFlag, stopAsyncFlag;
  barrier_t GPUBarrier;
  int isInterconflict; // set to 1 when a CPU-GPU conflict is found

  // memory pool
  void *hostMemPool;
  void *devMemPool;
  void *devMemPoolBmap;
  void *hostBackupMemPool;
  void *devBackupMemPool;
  void *devBackupMemPoolBmap;
  void *devVersions;
  void *devMemPoolShadow;
  void *devChunks;
  void *devMemPoolBackup; // only for HETM_ADDR_LOG and HETM_BMAP_LOG
  void *devMemPoolBackupBmap;
  size_t sizeMemPool;

  // inter-conflict flag
  int *hostInterConflFlag, *devInterConflFlag;

  // wset in the CPU, rset in the GPU (bitmap or explict)
  void *wsetLog, *rsetLog;
  size_t wsetLogSize, rsetLogSize;

  // BMAP only, stores a cache for the wsetLog
  void *wsetCache;
  void *wsetCacheConfl;
  void *wsetCacheConfl2;
  void *wsetCacheConfl3; // CPU WS | GPU WS
  size_t wsetCacheSize, wsetCacheBits, nbChunks;

  // TODO: remove this is benchmark specific
  void *devCurandState; // type is curandState*
  size_t devCurandStateSize;

  HeTM_thread_s *threadsInfo;
  pthread_t asyncThread;

  cudaEvent_t batchStartEvent, batchStopEvent;

  long threadsWaitingSync;

  long batchCount;
  float timeBudget;

} HeTM_shared_s;

typedef struct HeTM_statistics_ {
  long nbBatches, nbBatchesSuccess, nbBatchesFail;
  long nbTxsGPU, nbCommittedTxsGPU, nbDroppedTxsGPU, nbAbortsGPU;
  long nbTxsCPU, nbCommittedTxsCPU, nbDroppedTxsCPU, nbAbortsCPU;
  long nbEarlyValAborts;
  size_t sizeCpyDataset, sizeCpyWSet /* ADDR|BMAP */, sizeCpyLogs /* ADDR|VERS */;
  size_t sizeCpyWSetCPUData;
  long nbBitmapConflicts;
  long nbCPUdataConflicts;
  long txsNonBlocking;
  double timeNonBlocking, timeBlocking;
  double timeCMP, timeAfterCMP;
  double timeGPU, timePRSTM;
  double timeAbortedBatches;
  double timeDtD;
  double timeCPU;
  double timeMemCpySum;
  double totalTimeCpyWSet, totalTimeCmp, totalTimeCpyDataset;
} HeTM_statistics_s;

typedef struct HeTM_init_ {
  HETM_INVALIDATE_POLICY policy;
  int nbCPUThreads, nbGPUBlocks, nbGPUThreads;
  float timeBudget;
  int isCPUEnabled, isGPUEnabled;
} HeTM_init_s;

typedef struct HeTM_async_req_ {
  void *args;
  HeTM_request fn;
} HeTM_async_req_s;

extern HeTM_shared_s HeTM_shared_data;
extern HeTM_statistics_s HeTM_stats_data;
extern hetm_pc_s *HeTM_offload_pc;
extern __thread HeTM_thread_s *HeTM_thread_data;
extern void *HeTM_memStream;
extern void *HeTM_memStream2;

#ifdef HETM_OLD_BMAP_IMPL
// Use a cache with 1 value written/not-written
const static int CACHE_GRANULE_SIZE = 2147483648; // 2GB
const static int CACHE_GRANULE_BITS = 31;
#else /* new one */
// const int CACHE_GRANULE_SIZE = 4096;
// const int CACHE_GRANULE_BITS = 12;
const static int CACHE_GRANULE_SIZE = DEFAULT_BITMAP_GRANULARITY;// 16384;
const static int CACHE_GRANULE_BITS = DEFAULT_BITMAP_GRANULARITY_BITS;// 14;
#endif /* HETM_OLD_BMAP_IMPL */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int HeTM_init(HeTM_init_s);
int HeTM_destroy();

//---------------------- Memory Pool
// Sets the memory pool size for HeTM, users can then
// allocate memory from this pool using HeTM_alloc(...)
int HeTM_mempool_init(size_t pool_size);
int HeTM_mempool_destroy();
int HeTM_mempool_cpy_to_cpu(size_t *copiedSize, long batchCount);
int HeTM_mempool_cpy_to_gpu(size_t *copiedSize, long batchCount);
int HeTM_mempool_cpy_to_gpu_backup(size_t *copiedSize, long batchCount);
int HeTM_mempool_cpy_to_gpu_main(size_t *copiedSize, long batchCount);
int HeTM_alloc(void **cpu_ptr, void **gpu_ptr, size_t);
int HeTM_free(void **cpu_ptr);
void* HeTM_map_addr_to_gpu(void *origin);
void* HeTM_map_addr_to_cpu(void *origin);

void HeTM_initCurandState();
void HeTM_destroyCurandState();

#if HETM_LOG_TYPE != HETM_BMAP_LOG
// returns 1 if out of space in the buffer (and does not copy)
void HeTM_wset_log_cpy_to_gpu(HeTM_thread_s*, chunked_log_node_s*, size_t*);
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */
//----------------------

//---------------------- Threading
// Call this before HeTM_start for a custom core mapping function
int HeTM_set_thread_mapping_fn(HeTM_map_tid_to_core);

void HeTM_cpu_thread(); // Do not call this
void HeTM_gpu_thread(); // Do not call this

// Registers two callbacks, first for the CPU, second for the GPU,
// and the global arguments (GPU thread will have the last id),
//  --- IMPORTANT:
// GPU callback must guarantee the kernel is launched before returning
int HeTM_start(HeTM_callback, HeTM_callback, void *args);
int HeTM_before_cpu_start(HeTM_callback); // uses these to collect statistics
int HeTM_after_cpu_finish(HeTM_callback);
int HeTM_before_gpu_start(HeTM_callback);
int HeTM_after_gpu_finish(HeTM_callback);

// choose the policy for the next batch
int HeTM_choose_policy(HeTM_callback);

// neither CPU nor GPU are running (executes after HeTM_after_batch)
int HeTM_before_batch(HeTM_callback);

// neither CPU nor GPU are running (executes before HeTM_after_batch)
int HeTM_after_batch(HeTM_callback);

int HeTM_before_kernel(HeTM_callback); // use it, e.g., to send input
int HeTM_after_kernel(HeTM_callback); // TODO: batch is not complete

// Registers a callback to be later executed (serially in a side thread)
void HeTM_async_request(HeTM_async_req_s req);
void HeTM_free_async_request(HeTM_async_req_s *req); // Do not call this

int HeTM_sync_barrier(); // Do not call this
int HeTM_flush_barrier(); // Do not call this

// Waits the threads. Note: HeTM_set_is_stop(1) must be called
// before so the threads know it is time to stop
int HeTM_join_CPU_threads();
//----------------------

//---------------------- getters/setters
int HeTM_get_inter_confl_flag(void *stream, int doSync);
int HeTM_reset_inter_confl_flag();
#define HeTM_set_is_stop(isStop)       (HeTM_shared_data.stopFlag = isStop)
#define HeTM_is_stop()                 (HeTM_shared_data.stopFlag)
#define HeTM_async_set_is_stop(isStop) (HeTM_shared_data.stopAsyncFlag = isStop)
#define HeTM_async_is_stop()           (HeTM_shared_data.stopAsyncFlag)
#define HeTM_set_is_interconflict(val) (HeTM_shared_data.isInterconflict = val)
#define HeTM_is_interconflict()        (HeTM_shared_data.isInterconflict)
#define HeTM_set_GPU_status(status)    (HeTM_shared_data.statusGPU = status)
#define HeTM_get_GPU_status()          (HeTM_shared_data.statusGPU)

// resets PR-STM lock table, flags, etc
int HeTM_reset_GPU_state(long batchCount);
//----------------------

#ifdef __cplusplus
}
#endif /* __cplusplus */

// ###############################
// ### Debuging ##################
// ###############################
#ifdef HETM_DEB
#define HETM_PRINT(...) printf("[%s]: ", __func__); printf(__VA_ARGS__); printf("\n");

// TODO: use flags to enable/disable each module prints
// #define HETM_DEB_THREADING(...) printf("[THR]"); HETM_PRINT(__VA_ARGS__)
#define HETM_DEB_THREADING(...) /* empty */
#define HETM_DEB_THRD_CPU(...)  printf("[CPU]"); HETM_PRINT(__VA_ARGS__)
#define HETM_DEB_THRD_GPU(...)  printf("[GPU]"); HETM_PRINT(__VA_ARGS__)

#else /* !HETM_DEB */
#define HETM_PRINT(...) /* empty */

// TODO: use flags to enable/disable each module prints
#define HETM_DEB_THREADING(...) /* empty */
#define HETM_DEB_THRD_CPU(...)  /* empty */
#define HETM_DEB_THRD_GPU(...)  /* empty */
#endif
// ###############################

#endif /* HETM_H_GUARD_ */
