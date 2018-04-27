#include "hetm.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <curand_kernel.h>
#include "memman.h"
#include "knlman.h"
#include "hetm-log.h"
#include "pr-stm-wrapper.cuh"
#include "hetm-cmp-kernels.cuh"

#include <map>

using namespace std;

static map<void*, size_t> alloced;
static map<void*, size_t> freed;
static size_t curSize;

#define DEFAULT_BITMAP_GRANULARITY_BITS (24) // 16MB is Ok?
#define DEFAULT_BITMAP_GRANULARITY (0x1<<24)
static size_t bitmapGran = DEFAULT_BITMAP_GRANULARITY;
// static size_t bitmapGranBits = DEFAULT_BITMAP_GRANULARITY_BITS;

static void CUDART_CB cpyCallback(cudaStream_t event, cudaError_t status, void *data);

// TODO: this function is too long --> chunk it!
int HeTM_mempool_init(size_t pool_size)
{
  size_t nbGranules = pool_size / PR_LOCK_GRANULARITY;
  size_t sizeRSetLog = 0;
  size_t sizeWSetBuffer = 0;
  int isError = 0;

#if (HETM_LOG_TYPE == HETM_BMAP_LOG || HETM_LOG_TYPE == HETM_ADDR_LOG) && !defined(HETM_DISABLE_CHUNKS)
  size_t granBmap;
#endif

  // Inits mempool TODO!!! --> GPU also chunked
  isError |= memman_alloc_dual("HeTM_mempool", pool_size, MEMMAN_NONE); // TODO: trade-offs with MEMMAN_UNIF
  memman_create_bitmap(memman_get_cpu(NULL), memman_get_gpu(NULL), bitmapGran);
  memman_bitmap_gpu(); // creates in GPU (TODO: change name)
#ifdef HETM_DISABLE_CHUNKS // TODO: ADD YET ANOTHER FLAG!!!
  memman_set_is_bmapped(0, 0);
#else /* HETM_CHUNKS enabled */
  memman_set_is_bmapped(0, 1);
#endif /* HETM_DISABLE_CHUNKS */
  memman_bmap_s *mainBMap = (memman_bmap_s*) memman_get_bmap(NULL);
  memman_alloc_gpu("HeTM_mempool_bmap", sizeof(memman_bmap_s), mainBMap, MEMMAN_NONE);

#if HETM_LOG_TYPE == HETM_VERS_LOG || HETM_LOG_TYPE == HETM_VERS2_LOG
  // does not require extra structs
#elif HETM_LOG_TYPE == HETM_ADDR_LOG
  memman_select("HeTM_mempool");
  HeTM_shared_data.hostMemPool = memman_get_cpu(NULL);
  // TODO: too many combinations
#ifdef HETM_DISABLE_CHUNKS
  isError |= memman_alloc_gpu("HeTM_mempool_backup", pool_size,
    HeTM_shared_data.hostMemPool, MEMMAN_NONE);
  stm_devMemPoolBackupBmap = NULL;
#else /* HETM CHUNKS ENABLED */
  isError |= memman_alloc_gpu("HeTM_mempool_backup", pool_size,
    HeTM_shared_data.hostMemPool, MEMMAN_NONE);
  memman_create_bitmap(memman_get_cpu(NULL), NULL, bitmapGran);
  memman_set_is_bmapped(1, 0);
  stm_devMemPoolBackupBmap = memman_get_bmap(&granBmap);
  printf("Got %p\n", stm_devMemPoolBackupBmap);
#endif /* HETM_DISABLE_CHUNKS */
  HeTM_shared_data.devMemPoolBackup = memman_get_gpu(NULL);

#elif HETM_LOG_TYPE == HETM_BMAP_LOG
  // TODO: same as previous
  memman_select("HeTM_mempool");
  HeTM_shared_data.hostMemPool = memman_get_cpu(NULL);

  // TODO: too many combinations
#ifdef HETM_DISABLE_CHUNKS
  isError |= memman_alloc_gpu("HeTM_mempool_backup", pool_size,
    HeTM_shared_data.hostMemPool, MEMMAN_NONE);
  HeTM_shared_data.devMemPoolBackup = memman_get_gpu(NULL);
  stm_devMemPoolBackupBmap = NULL;
  isError |= memman_alloc_dual("HeTM_cpu_wset", nbGranules, MEMMAN_NONE);
#else /* HETM CHUNKS ENABLED */
  isError |= memman_alloc_gpu("HeTM_mempool_backup", pool_size,
    HeTM_shared_data.hostMemPool, MEMMAN_NONE);
  memman_create_bitmap(memman_get_cpu(NULL), NULL, bitmapGran);
  memman_set_is_bmapped(1, 0);
  HeTM_shared_data.devMemPoolBackup = memman_get_gpu(NULL);
  stm_devMemPoolBackupBmap = memman_get_bmap(&granBmap);
  isError |= memman_alloc_dual("HeTM_cpu_wset", nbGranules, MEMMAN_NONE);
  memman_attach_bmap(stm_devMemPoolBackupBmap, granBmap / 4); /* 1B maps 4B */
  memman_set_is_bmapped(1, 0); // TODO: this is not working anymore...
#endif /* HETM_DISABLE_CHUNKS */

  memman_zero_cpu(NULL); // BMAP only
  memman_zero_gpu(NULL);
  HeTM_shared_data.wsetLogSize = nbGranules;
  HeTM_shared_data.wsetLog     = memman_get_gpu(NULL);
  stm_wsetCPU                  = memman_get_cpu(NULL);

#endif /* HETM_LOG_TYPE == HETM_BMAP_LOG */

  memman_select("HeTM_mempool");
  HeTM_shared_data.hostMemPool = memman_get_cpu(NULL);
  HeTM_shared_data.devMemPool  = memman_get_gpu(NULL);
  HeTM_shared_data.sizeMemPool = pool_size;
  stm_baseMemPool = HeTM_shared_data.hostMemPool;

  // Inits versions buffer
  // TODO: the versions array occupies more space than the dataset!!!

#if HETM_LOG_TYPE == HETM_VERS_LOG || HETM_LOG_TYPE == HETM_VERS2_LOG
  size_t sizeVersion = nbGranules*sizeof(long);
  isError |= memman_alloc_gpu("HeTM_versions", sizeVersion, NULL, MEMMAN_NONE);
  HeTM_shared_data.devVersions = memman_get_gpu(NULL);
  memman_zero_gpu(NULL);
#elif HETM_LOG_TYPE == HETM_ADDR_LOG
  // TODO: I'm using the HeTM_versions in HETM_ADDR_LOG as CPU-WSet
  size_t sizeVersion = nbGranules*sizeof(char); // WSet bmap
  isError |= memman_alloc_gpu("HeTM_versions", sizeVersion, NULL, MEMMAN_NONE);
  HeTM_shared_data.devVersions = memman_get_gpu(NULL);
  memman_zero_gpu(NULL);
#endif

  // Inits inter-conflict flag
  isError |= memman_alloc_dual("HeTM_interConflFlag", sizeof(int), MEMMAN_NONE);
  HeTM_shared_data.hostInterConflFlag = (int*)memman_get_cpu(NULL);
  HeTM_shared_data.devInterConflFlag  = (int*)memman_get_gpu(NULL);

#if HETM_CMP_TYPE == HETM_CMP_EXPLICIT
  sizeRSetLog = HeTM_shared_data.nbGPUThreads*HeTM_shared_data.nbGPUBlocks
    *HeTM_get_explicit_log_block_size()*sizeof(PR_GRANULE_T);
#elif HETM_CMP_TYPE == HETM_CMP_COMPRESSED
  // Bitmap with 1 byte per account, 1 means accessed (TODO: /8 + atomicOr)
  sizeRSetLog = pool_size / PR_LOCK_GRANULARITY;
#else
    // Error or disabled
#endif
  if (sizeRSetLog > 0) {
    // Inits GPU read-set log
    isError |= memman_alloc_gpu("HeTM_dev_rset", sizeRSetLog, NULL, 0);
    memman_zero_gpu(NULL);
    HeTM_shared_data.rsetLog = memman_get_gpu(NULL);
    HeTM_shared_data.rsetLogSize = sizeRSetLog;

    // inits CPU write-set buffer
#if HETM_LOG_TYPE != HETM_BMAP_LOG
    // TODO: using the bitmap in the cpu side
    sizeWSetBuffer = STM_LOG_BUFFER_SIZE*LOG_SIZE*sizeof(HeTM_CPULogEntry);
#if HETM_LOG_TYPE == HETM_VERS2_LOG
    sizeWSetBuffer *= LOG_ACTUAL_SIZE;
#endif
    sizeWSetBuffer *= HeTM_shared_data.nbCPUThreads; // 1 buffer per thread
    isError |= memman_alloc_gpu("HeTM_host_wset", sizeWSetBuffer, NULL, 0);
    memman_zero_gpu(NULL);
    HeTM_shared_data.wsetLog = memman_get_gpu(NULL);
    HeTM_shared_data.wsetLogSize = sizeWSetBuffer;
#endif
  } // else CMP is disabled

  HeTM_set_global_arg((HeTM_knl_global_s){
    .devMemPoolBasePtr  = HeTM_shared_data.devMemPool,
#if HETM_LOG_TYPE == HETM_ADDR_LOG || HETM_LOG_TYPE == HETM_BMAP_LOG
    .devMemPoolBackupBasePtr = HeTM_shared_data.devMemPoolBackup,
#endif
    .hostMemPoolBasePtr = HeTM_shared_data.hostMemPool,
    .versions           = HeTM_shared_data.devVersions,
    .isInterConfl       = HeTM_shared_data.devInterConflFlag,
    .explicitLogBlock   = HeTM_get_explicit_log_block_size(),
    .nbGranules         = nbGranules, // TODO: granules
    .devRSet            = HeTM_shared_data.rsetLog,
    .hostWSet           = HeTM_shared_data.wsetLog,
    .PRLockTable        = PR_lockTableDev,
    .randState          = HeTM_shared_data.devCurandState
  });

  curSize = 0;
  return isError;
}

int HeTM_mempool_destroy()
{
  memman_select("HeTM_mempool");
  memman_free_dual();
  if(!memman_select("HeTM_gpuLog"))
    memman_free_dual();
  memman_select("HeTM_versions");
  memman_free_gpu();
  memman_select("HeTM_dev_rset");
  memman_free_gpu();
  memman_select("HeTM_host_wset");
  memman_free_gpu();
  return 0;
}

void HeTM_initCurandState()
{
  int nbThreads = HeTM_shared_data.nbGPUThreads;
  int nbBlocks = HeTM_shared_data.nbGPUBlocks;
  size_t size = nbThreads * nbBlocks * sizeof(long); // TODO: from sizeof(curandState)
  memman_alloc_gpu("HeTM_curand_state", size, NULL, 0);
  HeTM_shared_data.devCurandState = memman_get_gpu(NULL);
  HeTM_setupCurand<<<nbBlocks, nbThreads>>>(HeTM_shared_data.devCurandState);
  cudaThreadSynchronize(); // TODO: blocks
}

void HeTM_destroyCurandState()
{
  memman_select("HeTM_curand_state");
  memman_free_gpu();
}

int HeTM_mempool_cpy_to_cpu(size_t *copiedSize)
{
  memman_select("HeTM_mempool");
  memman_cpy_to_cpu(HeTM_memStream, copiedSize);
  return 0; // TODO: error code
}

int HeTM_mempool_cpy_to_gpu(size_t *copiedSize)
{
  memman_select("HeTM_mempool");
  memman_cpy_to_gpu(HeTM_memStream, copiedSize);
  return 0; // TODO: error code
}

#if HETM_LOG_TYPE == HETM_VERS2_LOG
void HeTM_wset_log_cpy_to_gpu(
  HeTM_thread_s *threadData, chunked_log_node_s *node, size_t *size
) {
  char *buffer = node->chunk;
  cudaStream_t stream = (cudaStream_t)threadData->stream;
  size_t emptySpace;

  threadData->countCpy = 1; // TODO: start with big number
  threadData->isCpyDone = 0;
  threadData->nbCpyDone = 0;
  __sync_synchronize();
  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, stream);
  CUDA_CPY_TO_DEV_ASYNC(HeTM_shared_data.wsetLog, buffer, threadData->curWSetSize, stream);
  HeTM_stats_data.sizeCpyLogs += threadData->curWSetSize;
  emptySpace = MOD_CHUNKED_LOG_NODE_FREE_SPACE(node);
  HeTM_stats_data.sizeCpyWSet += emptySpace;
  if (threadData->isFirstChunk) {
    threadData->emptySpaceFirstChunk += emptySpace;
    threadData->isFirstChunk = 0;
  }
  CUDA_CHECK_ERROR(cudaStreamAddCallback(
    stream, cpyCallback, threadData, 0
  ), "");
  __sync_synchronize();
  if (threadData->nbCpyDone >= threadData->countCpy) {
    threadData->isCpyDone = 1;
  }
  __sync_synchronize();
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, stream);
}

#elif HETM_LOG_TYPE != HETM_BMAP_LOG

void HeTM_wset_log_cpy_to_gpu(
  HeTM_thread_s *threadData, chunked_log_node_s *node, size_t *size
) {
  void *res;
  HeTM_CPULogEntry *resAux;
  size_t sizeRes = 0, sizeToCpy, sizeBuffer;
  chunked_log_node_s *logAux;
  int tid = threadData->id;
  cudaStream_t stream = (cudaStream_t)threadData->stream;
  int count = 0;

  sizeBuffer = STM_LOG_BUFFER_SIZE * LOG_SIZE;

  res = HeTM_shared_data.wsetLog;
  resAux = (HeTM_CPULogEntry*)res;
  resAux += tid*sizeBuffer; // each thread has a bit of the buffer
  // TODO: this memset is needed
  // CUDA_CHECK_ERROR(cudaMemsetAsync(resAux, 0, sizeBuffer*sizeof(HeTM_CPULogEntry)), "");

  logAux = node;
  threadData->countCpy = 1000; // TODO: start with big number
  threadData->isCpyDone = 0;
  threadData->nbCpyDone = 0;
  CUDA_EVENT_RECORD(threadData->cpyWSetStartEvent, stream);
  size_t totCpy = 0;
  while (logAux != NULL) {
    sizeToCpy = logAux->p.pos; // size in bytes
    CUDA_CPY_TO_DEV_ASYNC(resAux, logAux->chunk, sizeToCpy, stream);
    totCpy += sizeToCpy;
    count++;
    CUDA_CHECK_ERROR(cudaStreamAddCallback(
        stream, cpyCallback, threadData, 0
      ), "");

    resAux += logAux->p.pos/sizeof(HeTM_CPULogEntry); // move ahead of the copied position
    sizeRes += logAux->p.pos/sizeof(HeTM_CPULogEntry);

    // TODO: now the log is <= STM_LOG_BUFFER_SIZE

    node = logAux;
    logAux = logAux->next;
    CHUNKED_LOG_FREE(node);
  }
  HeTM_stats_data.sizeCpyLogs += totCpy;
  // HeTM_stats_data.sizeCpyWSet += sizeBuffer - totCpy; // empty space

  threadData->countCpy = count; // TODO: does this work
  __sync_synchronize();
  if (threadData->nbCpyDone >= threadData->countCpy) {
    threadData->isCpyDone = 1;
    // printf(" --- send to %i wait cpy\n", threadData->id);
    // TODO: erase values
  }
  __sync_synchronize();

  if (size != NULL) *size = sizeRes; // returns the number of copied entries
  CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, stream);
}
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */

int HeTM_alloc(void **cpu_ptr, void **gpu_ptr, size_t size)
{
  size_t newSize = curSize + size;
  if (newSize > HeTM_shared_data.sizeMemPool) {
    // TODO: check the freed memory
    return -1;
  }

  // there is still space left
  char *curPtrHost = (char*)HeTM_shared_data.hostMemPool;
  char *curPtrDev  = (char*)HeTM_shared_data.devMemPool;
  curPtrHost += curSize;
  curPtrDev  += curSize;
  if (cpu_ptr) *cpu_ptr = (void*)curPtrHost;
  if (gpu_ptr) *gpu_ptr = (void*)curPtrDev;
  curSize = newSize;
  alloced.insert(make_pair(*cpu_ptr, size));

  return 0;
}

int HeTM_free(void **cpu_ptr)
{
  // TODO:
  auto it = alloced.find(*cpu_ptr);
  if (it == alloced.end()) {
    return -1; // not found
  }
  freed.insert(make_pair(*cpu_ptr, it->second));
  alloced.erase(it);
  return 0;
}

void* HeTM_map_addr_to_gpu(void *origin)
{
  uintptr_t o = (uintptr_t)origin;
  uintptr_t host = (uintptr_t)HeTM_shared_data.hostMemPool;
  uintptr_t dev  = (uintptr_t)HeTM_shared_data.devMemPool;
  return (void*)(o - host + dev);
}

void* HeTM_map_cpu_to_cpu(void *origin)
{
  uintptr_t o = (uintptr_t)origin;
  uintptr_t host = (uintptr_t)HeTM_shared_data.hostMemPool;
  uintptr_t dev  = (uintptr_t)HeTM_shared_data.devMemPool;
  return (void*)(o - dev + host);
}

int HeTM_reset_GPU_state()
{
  // CUDA_CHECK_ERROR(cudaMemset(PR_lockTableDev, 0, PR_LOCK_TABLE_SIZE*sizeof(int)), "");
  HeTM_reset_inter_confl_flag();
  memman_select("HeTM_dev_rset");
  memman_zero_gpu(NULL);
  return 0;
}

static void CUDART_CB cpyCallback(cudaStream_t event, cudaError_t status, void *data)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)data;

  if(status != cudaSuccess) { // TODO: Handle error
    printf("CMP failed! >>> %s <<<\n", cudaGetErrorString(status));
    // TODO: exit application
  }

  threadData->nbCpyDone++;
  if (threadData->nbCpyDone >= threadData->countCpy) {
    // printf(" --- send to %i wait cpy\n", threadData->id);
    threadData->isCpyDone = 1;
    // TODO: erase values?
  }
  __sync_synchronize(); // cmpCallback is called from a different thread
}
