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

#include "rdtsc.h"

#include <map>

using namespace std;

static map<void*, size_t> alloced;
static map<void*, size_t> freed;
static size_t curSize;

static size_t bitmapGran = DEFAULT_BITMAP_GRANULARITY;
// static size_t bitmapGranBits = DEFAULT_BITMAP_GRANULARITY_BITS;

#if HETM_LOG_TYPE != HETM_BMAP_LOG
static void CUDART_CB cpyCallback(cudaStream_t event, cudaError_t status, void *data);
#endif

static void init_mempool(size_t pool_size);
static void init_RSetWSet(size_t pool_size);
static void init_interConflFlag();

#if HETM_LOG_TYPE == HETM_VERS_LOG
  static void init_vers(size_t pool_size);
#elif  HETM_LOG_TYPE == HETM_BMAP_LOG
  static void init_bmap(size_t pool_size);
#endif

int HeTM_mempool_init(size_t pool_size)
{
  size_t nbGranules = pool_size / PR_LOCK_GRANULARITY;
  size_t nbChunks;

  init_mempool(pool_size);

#if HETM_LOG_TYPE == HETM_VERS_LOG
  init_vers(pool_size);
#elif  HETM_LOG_TYPE == HETM_BMAP_LOG
  init_bmap(pool_size);
#else /* HETM_LOG_TYPE not defined */
  printf(" >>> ERROR! HETM_LOG_TYPE not defined\n");
  exit(EXIT_FAILURE);
#endif

  init_interConflFlag();
  init_RSetWSet(pool_size);

  nbChunks = pool_size / DEFAULT_BITMAP_GRANULARITY;
  if (nbGranules % DEFAULT_BITMAP_GRANULARITY > 0) nbChunks++;
  HeTM_shared_data.nbChunks = nbChunks;

  // printf(" <<<<<<< HeTM_shared_data.devMemPoolBackupBmap = %p\n",  HeTM_shared_data.devMemPoolBackupBmap);
  memman_select("HeTM_mempool");
  memman_bmap_s *mainBMap = (memman_bmap_s*) memman_get_bmap(NULL);

  HeTM_set_global_arg((HeTM_knl_global_s){
    .devMemPoolBasePtr  = HeTM_shared_data.devMemPool,
    .devMemPoolBackupBasePtr = HeTM_shared_data.devMemPoolBackup,
    .devMemPoolBackupBmap = ((memman_bmap_s*)HeTM_shared_data.devMemPoolBackupBmap)->dev,
    .hostMemPoolBasePtr  = HeTM_shared_data.hostMemPool,
    .versions            = HeTM_shared_data.devVersions,
    .isInterConfl        = HeTM_shared_data.devInterConflFlag,
    .explicitLogBlock    = HeTM_get_explicit_log_block_size(),
    .nbGranules          = nbGranules, // TODO: granules
    .devRSet             = HeTM_shared_data.rsetLog,
    .hostWSet            = HeTM_shared_data.wsetLog,
    .hostWSetCache       = HeTM_shared_data.wsetCache,
    .hostWSetCacheConfl  = HeTM_shared_data.wsetCacheConfl,
    .hostWSetCacheConfl2 = HeTM_shared_data.wsetCacheConfl2,
    .hostWSetCacheConfl3 = HeTM_shared_data.wsetCacheConfl3,
    .hostWSetCacheSize   = HeTM_shared_data.wsetCacheSize,
    .hostWSetCacheBits   = HeTM_shared_data.wsetCacheBits,
    .hostWSetChunks      = nbChunks,
    .PRLockTable         = PR_lockTableDev,
    .randState           = HeTM_shared_data.devCurandState,
    .isGPUOnly           = (HeTM_shared_data.isCPUEnabled == 0),
    .GPUwsBmap           = mainBMap->dev
  });

  curSize = 0;
  return 0; // TODO: check for error
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
  cudaDeviceSynchronize(); // TODO: blocks
}

void HeTM_destroyCurandState()
{
  memman_select("HeTM_curand_state");
  memman_free_gpu();
}

int HeTM_mempool_cpy_to_cpu(size_t *copiedSize, long batchCount)
{
#ifndef USE_UNIF_MEM
  // memman_select("HeTM_mempool");
  // I'm assuming the gpu did the D->D copy
  memman_select("HeTM_mempool_backup");
  // if (HeTM_shared_data.isCPUEnabled == 0) {
  //   // there is some bug with the counter
    memman_cpy_to_cpu(HeTM_memStream, copiedSize, batchCount);
  // } else {
  //   memman_cpy_to_cpu(HeTM_memStream, copiedSize, batchCount);
  // }

#endif /* USE_UNIF_MEM */
  return 0; // TODO: error code
}

int HeTM_mempool_cpy_to_gpu_backup(size_t *copiedSize, long batchCount)
{
#ifndef USE_UNIF_MEM
  // bitmaps copied elsewhere (memman_cpy_to_cpu_buffer_bitmap)
  memman_select("HeTM_mempool_backup");
  void *devPtr = memman_get_gpu(NULL);
  memman_select("HeTM_mempool");
  memman_cpy_to_gpu_backup(devPtr, HeTM_memStream, copiedSize, batchCount);
#endif /* USE_UNIF_MEM */
  return 0; // TODO: error code
}

int HeTM_mempool_cpy_to_gpu_main(size_t *copiedSize, long batchCount)
{
#ifndef USE_UNIF_MEM
  // bitmaps copied elsewhere (memman_cpy_to_cpu_buffer_bitmap)
  memman_select("HeTM_mempool"); // I've just inverted these lines
  void *devPtr = memman_get_gpu(NULL);
  memman_select("HeTM_mempool_backup");
  memman_cpy_to_gpu_backup(devPtr, HeTM_memStream, copiedSize, batchCount);
#endif /* USE_UNIF_MEM */
  return 0; // TODO: error code
}

int HeTM_mempool_cpy_to_gpu(size_t *copiedSize, long batchCount)
{
#ifndef USE_UNIF_MEM
  memman_select("HeTM_mempool");
  memman_cpy_to_gpu(HeTM_memStream, copiedSize, batchCount);
#endif /* USE_UNIF_MEM */
  return 0; // TODO: error code
}

#if HETM_LOG_TYPE != HETM_BMAP_LOG

void HeTM_wset_log_cpy_to_gpu(
  HeTM_thread_s *threadData, chunked_log_node_s *node, size_t *size
) {
  void *res;
  HeTM_CPULogEntry *resAux;
  size_t sizeRes = 0, sizeToCpy, sizeBuffer, nbEntries;
  chunked_log_node_s *logAux;
  int tid = threadData->id;
  cudaStream_t stream = (cudaStream_t)threadData->stream;
  int count = 0;

  // printf("!%i! before set curCopyNb=%i\n", threadData->id, threadData->curCopyNb);

  sizeBuffer = STM_LOG_BUFFER_SIZE * LOG_SIZE;

  res = HeTM_shared_data.wsetLog;

  resAux = (HeTM_CPULogEntry*)res;
  resAux += tid*sizeBuffer; // each thread has a bit of the buffer

  // Thread may have copied previous chunks
  resAux += threadData->curCopyNb * LOG_SIZE;

  // TODO: the events slow down the experiment !!! ---- do not always take them
  if (threadData->logChunkEventCounter > 0 && HeTM_shared_data.batchCount == 1) {
    int nbCopies = threadData->logChunkEventCounter - threadData->logChunkEventStore;
    int counter = threadData->logChunkEventCounter;
    for (int i = 0; i < nbCopies; ++i) {
      int eventIdx = (i+threadData->logChunkEventStore) % STM_LOG_BUFFER_SIZE;
      float timeTaken;
      CUDA_EVENT_SYNCHRONIZE(threadData->cpyLogChunkStartEvent[eventIdx]);
      CUDA_EVENT_SYNCHRONIZE(threadData->cpyLogChunkStopEvent[eventIdx]);
      CUDA_EVENT_ELAPSED_TIME(&timeTaken, threadData->cpyLogChunkStartEvent[eventIdx],
        threadData->cpyLogChunkStopEvent[eventIdx]);
      // threadData->timeLogs is an average
      threadData->timeLogs += (1.0f / (counter + 1.0f)) * (timeTaken - threadData->timeLogs);
      threadData->timeCpy = timeTaken;
      threadData->timeCpySum += threadData->timeCpy;
      counter++;
    }
    threadData->logChunkEventStore = threadData->logChunkEventCounter;
  } else if (threadData->logChunkEventCounter > 0) {
    threadData->timeCpy = threadData->timeLogs; // uses the averaged value
    threadData->timeCpySum += threadData->timeCpy;
  }

  logAux = node;

  // unsigned long t0 = rdtsc(), t2, t3, t4;
  // unsigned long diffT2 = 0, diffCpy = 0;

  size_t totCpy = 0;
  // t2 = rdtsc();
  while (logAux != NULL) {
    sizeToCpy = logAux->p.pos; // size in bytes
    nbEntries = logAux->p.pos/sizeof(HeTM_CPULogEntry);

    if (sizeToCpy <= 0 || nbEntries > sizeBuffer || logAux->chunk == NULL) {
      node = logAux;
      logAux = logAux->next;
      continue;
    }

    // t3 = rdtsc();
    // diffT2 += t3-t2;
    if (HeTM_shared_data.batchCount == 1) {
      CUDA_EVENT_RECORD(threadData->cpyLogChunkStartEvent[
        threadData->logChunkEventCounter % STM_LOG_BUFFER_SIZE], stream);
    }
    if (CUDA_CPY_TO_DEV_ASYNC(resAux, logAux->chunk, sizeToCpy, stream) != cudaSuccess) {
      printf("ERROR copying Dev %p/%p <-- %p/%p Host (nbEntries=%zu) \n"
             "           BUFFER=%p/%p totalCpy=%zu maxSize=%zu\n",
        resAux, ((char*)resAux)+sizeToCpy, logAux->chunk, logAux->chunk+sizeToCpy, nbEntries, HeTM_shared_data.wsetLog,
        ((HeTM_CPULogEntry*)HeTM_shared_data.wsetLog) + HeTM_shared_data.wsetLogSize,
        totCpy, sizeBuffer*sizeof(HeTM_CPULogEntry));
    }
    if (HeTM_shared_data.batchCount == 1) {
      CUDA_EVENT_RECORD(threadData->cpyLogChunkStopEvent[
        threadData->logChunkEventCounter % STM_LOG_BUFFER_SIZE], stream);
    }
    threadData->logChunkEventCounter++; // uses the next event to collect time
    // t4 = rdtsc();
    // diffCpy += t4-t3;

    threadData->curCopyNb++;

    totCpy += sizeToCpy;
    count++;

    resAux += nbEntries; // move ahead of the copied position
    sizeRes += nbEntries;

    // TODO: now the log is <= STM_LOG_BUFFER_SIZE

    node = logAux;
    logAux = logAux->next;

    if (totCpy >= sizeBuffer*sizeof(HeTM_CPULogEntry)) {
      // no more space in the buffer try latter
      break;
    }
    // t2 = rdtsc();
  }

  // printf("set threadData->curCopyNb = %i\n", threadData->curCopyNb);

  if (totCpy > 0) {
    CUDA_CHECK_ERROR(cudaStreamAddCallback(
      stream, cpyCallback, threadData, 0
    ), "");
  } else {
    threadData->isCpyDone = 1;
  }
  threadData->isCopying = 0; // next one can come in

  // unsigned long t1 = rdtsc();
  //
  // printf("!%2i! after set curCopyNb=%2i did %2i passes in %lu kCycles, diffT2 = %lu kCycles, diffCpy = %lu kCycles\n",
  //   threadData->id, threadData->curCopyNb, count, (t1-t0) / 1000, diffT2 / 1000, diffCpy / 1000);

  HeTM_stats_data.sizeCpyLogs += totCpy;
  // HeTM_stats_data.sizeCpyWSet += sizeBuffer - totCpy; // empty space
  __sync_synchronize();

  if (size != NULL) *size = sizeRes; // returns the number of copied entries
  // CUDA_EVENT_RECORD(threadData->cpyWSetStopEvent, stream); // USE TIMER_T
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

int HeTM_reset_GPU_state(long batchCount)
{
  // CUDA_CHECK_ERROR(cudaMemset(PR_lockTableDev, 0, PR_LOCK_TABLE_SIZE*sizeof(int)), "");
  // HeTM_reset_inter_confl_flag(); // reset elsewhere
  if ((batchCount & 0xff) == 0xff) { // at some round --> reset
    memman_select("HeTM_dev_rset");
    memman_zero_gpu(NULL);
  }
  return 0;
}

#if HETM_LOG_TYPE != HETM_BMAP_LOG
static void CUDART_CB cpyCallback(cudaStream_t event, cudaError_t status, void *data)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)data;

  if(status != cudaSuccess) { // TODO: Handle error
    printf("CMP failed! >>> %s <<<\n", cudaGetErrorString(status));
    // TODO: exit application
  }

  // threadData->nbCpyDone++;
  // if (threadData->nbCpyDone >= threadData->countCpy) {
    threadData->isCpyDone = 1;
  //   // TODO: erase values?
  // }

  __sync_synchronize(); // cmpCallback is called from a different thread
}
#endif

static void init_mempool(size_t pool_size)
{
  size_t granBmap;
  memman_alloc_dual("HeTM_mempool", pool_size,
#ifdef USE_UNIF_MEM
    MEMMAN_UNIF
#else /* !USE_UNIF_MEM */
    MEMMAN_NONE
#endif /* USE_UNIF_MEM */
  );
  HeTM_shared_data.devMemPool  = memman_get_gpu(NULL);
  HeTM_shared_data.hostMemPool = memman_get_cpu(NULL);
  HeTM_shared_data.sizeMemPool = pool_size;
  stm_baseMemPool = HeTM_shared_data.hostMemPool;

#ifdef HETM_OVERLAP_CPY_BACK
  // TODO: duplicate HeTM_mempool here!
#endif /* HETM_OVERLAP_CPY_BACK */

  // bmap for the data-set chunking
  // TODO: bitmap is x4B larger (does not take in account ints of 4B)
  memman_create_bitmap(HeTM_shared_data.hostMemPool, HeTM_shared_data.devMemPool, bitmapGran);
  memman_bitmap_gpu(); // creates in GPU (TODO: change name)
  memman_bmap_s *mainBMap = (memman_bmap_s*) memman_get_bmap(&granBmap);
  HeTM_shared_data.devMemPoolBmap = mainBMap->dev;
  stm_devMemPoolBmap = mainBMap;

  memman_set_is_bmapped(1, 1);
  memman_alloc_gpu("HeTM_mempool_backup", pool_size,
    HeTM_shared_data.hostMemPool, MEMMAN_NONE);
  HeTM_shared_data.hostBackupMemPool = memman_get_cpu(NULL);
  HeTM_shared_data.devBackupMemPool = memman_get_gpu(NULL);
  memman_create_bitmap(HeTM_shared_data.hostBackupMemPool, HeTM_shared_data.devBackupMemPool, bitmapGran);
  memman_bitmap_gpu(); // creates in GPU (TODO: change name)
  memman_bmap_s *backupBMap = (memman_bmap_s*) memman_get_bmap(NULL);
  HeTM_shared_data.devBackupMemPoolBmap = backupBMap->dev;
  stm_devMemPoolBackupBmap = backupBMap;

  memman_attach_bmap(stm_devMemPoolBackupBmap, granBmap);
  memman_set_is_bmapped(1, 1);
  HeTM_shared_data.devMemPoolBackup = memman_get_gpu(NULL);
  HeTM_shared_data.devMemPoolBackupBmap = memman_get_bmap(NULL);

  // GPU-side write-set for chunk copy
  memman_alloc_gpu("HeTM_mempool_backup_bmap", sizeof(memman_bmap_s), backupBMap, MEMMAN_NONE);
  memman_alloc_gpu("HeTM_mempool_bmap", sizeof(memman_bmap_s), mainBMap, MEMMAN_NONE);
}

static void init_interConflFlag() {
  memman_alloc_dual("HeTM_interConflFlag", sizeof(int), MEMMAN_NONE);
  HeTM_shared_data.hostInterConflFlag = (int*)memman_get_cpu(NULL);
  HeTM_shared_data.devInterConflFlag  = (int*)memman_get_gpu(NULL);
}

static void init_RSetWSet(size_t pool_size) {
  size_t sizeRSetLog = 0;

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
    memman_alloc_gpu("HeTM_dev_rset", sizeRSetLog, NULL, 0);
    memman_zero_gpu(NULL);
    HeTM_shared_data.rsetLog = memman_get_gpu(NULL);
    HeTM_shared_data.rsetLogSize = sizeRSetLog;

    // inits CPU write-set buffer
#if HETM_LOG_TYPE != HETM_BMAP_LOG
    size_t sizeWSetBuffer = 0;
    sizeWSetBuffer = STM_LOG_BUFFER_SIZE*LOG_SIZE*sizeof(HeTM_CPULogEntry);
    sizeWSetBuffer *= HeTM_shared_data.nbCPUThreads; // 1 buffer per thread
    memman_alloc_gpu("HeTM_host_wset", sizeWSetBuffer, NULL, 0);
    memman_zero_gpu(NULL);
    HeTM_shared_data.wsetLog = memman_get_gpu(NULL);
    HeTM_shared_data.wsetLogSize = sizeWSetBuffer;
#endif
  } // else CMP is disabled
}

#if HETM_LOG_TYPE == HETM_BMAP_LOG
static void init_bmap(size_t pool_size)
{
  size_t nbGranules = pool_size / PR_LOCK_GRANULARITY;
  size_t granBmap;

  size_t cacheSize = nbGranules / CACHE_GRANULE_SIZE; // nbGranules / CACHE_GRANULE_SIZE;

  if (pool_size % CACHE_GRANULE_SIZE > 0) {
    cacheSize++;
  }

  memman_alloc_dual("HeTM_cpu_wset_cache", cacheSize, MEMMAN_NONE);

  // GPU set to 1 to say there was a conflict
  memman_alloc_dual("HeTM_cpu_wset_cache_confl", cacheSize, MEMMAN_NONE);
  memman_alloc_dual("HeTM_cpu_wset_cache_confl2", cacheSize, MEMMAN_NONE);
  memman_alloc_dual("HeTM_cpu_wset_cache_confl3", cacheSize, MEMMAN_NONE);

  memman_select("HeTM_mempool");
  memman_bmap_s *mainBMap = (memman_bmap_s*) memman_get_bmap(&granBmap);

  // memman_alloc_gpu("HeTM_mempool_backup", pool_size,
  //   HeTM_shared_data.hostMemPool, MEMMAN_NONE);
  // // memman_create_bitmap(memman_get_cpu(NULL), NULL, bitmapGran);
  // stm_devMemPoolBackupBmap = mainBMap;
  // memman_attach_bmap(stm_devMemPoolBackupBmap, granBmap);
  // memman_set_is_bmapped(1, 0);
  // HeTM_shared_data.devMemPoolBackup = memman_get_gpu(NULL);
  memman_alloc_dual("HeTM_cpu_wset", nbGranules, MEMMAN_NONE);
  memman_attach_bmap(stm_devMemPoolBackupBmap, granBmap / 4); /* 1B maps 4B */
  memman_set_is_bmapped(1, 0);

  memman_zero_cpu(NULL);
  memman_zero_gpu(NULL);
  HeTM_shared_data.wsetLogSize = nbGranules;
  HeTM_shared_data.wsetLog     = memman_get_gpu(NULL);
  stm_wsetCPU                  = memman_get_cpu(NULL);

  memman_select("HeTM_cpu_wset_cache");
  memman_zero_cpu(NULL);
  memman_zero_gpu(NULL);
  HeTM_shared_data.wsetCache     = memman_get_gpu(NULL);
  HeTM_shared_data.wsetCacheSize = CACHE_GRANULE_SIZE;
  HeTM_shared_data.wsetCacheBits = CACHE_GRANULE_BITS;
  stm_wsetCPUCache               = memman_get_cpu(NULL);
  stm_wsetCPUCacheBits           = CACHE_GRANULE_BITS;

  char *cache_x64;
  malloc_or_die(cache_x64, cacheSize * 64);
  stm_wsetCPUCache_x64 = cache_x64;

  memman_select("HeTM_cpu_wset_cache_confl2");
  memman_zero_cpu(NULL);
  memman_zero_gpu(NULL);
  HeTM_shared_data.wsetCacheConfl2 = memman_get_gpu(NULL);
  memman_select("HeTM_cpu_wset_cache_confl3");
  memman_zero_cpu(NULL);
  memman_zero_gpu(NULL);
  HeTM_shared_data.wsetCacheConfl3 = memman_get_gpu(NULL);
  memman_select("HeTM_cpu_wset_cache_confl");
  memman_zero_cpu(NULL);
  memman_zero_gpu(NULL);
  HeTM_shared_data.wsetCacheConfl = memman_get_gpu(NULL);

  memman_select("HeTM_mempool");
  HeTM_shared_data.hostMemPool = memman_get_cpu(NULL);
  // HeTM_shared_data.devMemPool  = memman_get_gpu(NULL);
  HeTM_shared_data.sizeMemPool = pool_size;
  stm_baseMemPool = HeTM_shared_data.hostMemPool;
}
#endif /* HETM_LOG_TYPE == HETM_BMAP_LOG */

#if HETM_LOG_TYPE == HETM_VERS_LOG
static void init_vers(size_t pool_size) {
  size_t nbGranules = pool_size / PR_LOCK_GRANULARITY;

  memman_select("HeTM_mempool");
  HeTM_shared_data.hostMemPool = memman_get_cpu(NULL);
  // HeTM_shared_data.devMemPool  = memman_get_gpu(NULL);
  HeTM_shared_data.sizeMemPool = pool_size;
  stm_baseMemPool = HeTM_shared_data.hostMemPool;

  // Inits versions buffer
  size_t sizeVersion = nbGranules*sizeof(long);
  memman_alloc_gpu("HeTM_versions", sizeVersion, NULL, MEMMAN_NONE);
  HeTM_shared_data.devVersions = memman_get_gpu(NULL);
  memman_zero_gpu(NULL);
}
#endif /* HETM_LOG_TYPE == HETM_VERS_LOG */
