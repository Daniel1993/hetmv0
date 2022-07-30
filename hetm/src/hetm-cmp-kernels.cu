#include "hetm-log.h"
#include "hetm-cmp-kernels.cuh"
#include "hetm.cuh"

// Accessible globally
__constant__ HeTM_knl_global_s HeTM_knl_global;

// ---------------------- EXPLICIT
static size_t explicitLogBlock = 0;
// -------------------------------

void HeTM_set_explicit_log_block_size(size_t size)
{
  explicitLogBlock = size;
}

size_t HeTM_get_explicit_log_block_size()
{
  return explicitLogBlock;
}

#if HETM_LOG_TYPE == HETM_BMAP_LOG

// HETM_BMAP_LOG requires a specific kernel

__global__ void HeTM_knl_checkTxBitmapCache(HeTM_knl_cmp_args_s args)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  size_t nbChunks = HeTM_knl_global.hostWSetChunks;

  if (id >= nbChunks) {
    return; // the thread has nothing to do
  }

  // unsigned char *rset = (unsigned char*)HeTM_knl_global.devRSet;
  unsigned char *rset = (unsigned char*)HeTM_knl_global.hostWSetCacheConfl;
  unsigned char *rwsetConfl = (unsigned char*)HeTM_knl_global.hostWSetCacheConfl2;
  unsigned char *unionWS = (unsigned char*)HeTM_knl_global.hostWSetCacheConfl3;
  unsigned char *GPUwset = (unsigned char*)HeTM_knl_global.GPUwsBmap;
  unsigned char *wset = (unsigned char*)HeTM_knl_global.hostWSetCache;

  // int cacheId = id >> wsetBits;
  // unsigned char isNewWrite = wset[cacheId] == args.batchCount;
  unsigned char isNewWrite = wset[id] == args.batchCount;

  // checks for a chunck of CPU dataset if it was read in GPU (4096 items)
  // memman_bmap_s *key_bmap = (memman_bmap_s*)HeTM_knl_global.devMemPoolBackupBmap;
  // char *bytes = key_bmap->dev;
  char *bytes = (char*)HeTM_knl_global.devMemPoolBackupBmap;

  if (isNewWrite) { // TODO: divergency
    bytes[id] = 1; /*args.batchCount*/
  }
  int isConfl = (rset[id] == args.batchCount) && isNewWrite;
  int isConflW = (GPUwset[id] && isNewWrite) ? 1 : 0;
  unionWS[id] = (GPUwset[id] || isNewWrite) ? 1 : 0;

  // the rset cache is also used as conflict detection
  rset[id] = isConfl ? args.batchCount : 0;
  rwsetConfl[id] = isConflW ? 1 : 0;

  // printf("id=%i rset[id]=%i rwsetConfl[id]=%i\n", id, (int)rset[id], (int)rwsetConfl[id]);

  if (isConfl) {
    *HeTM_knl_global.isInterConfl = 1;
    // printf("GPU conflict found\n");
    // ((unsigned char*)(HeTM_knl_global.hostWSetCacheConfl))[cacheId] = args.batchCount;
  }
}

__global__ void HeTM_knl_checkTxBitmap(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int idPlusOffset = id + offset;

  // TODO: these must be the same
  int sizeWSet = args.sizeWSet; /* Size of the host log */
  // int sizeRSet = args.sizeRSet; /* Size of the device log */

  if (idPlusOffset >= sizeWSet) return; // the thread has nothing to do

  unsigned char *rset = (unsigned char*)HeTM_knl_global.devRSet;
  unsigned char *wset = (unsigned char*)HeTM_knl_global.hostWSet;

  // TODO: use shared memory
  unsigned char isNewWrite = wset[idPlusOffset] == args.batchCount;
  unsigned char isConfl = (rset[idPlusOffset] == args.batchCount) && isNewWrite;

  if (isConfl) {
    // printf("[%i] found conflict wset[%i]=%i\n", id, idPlusOffset, (int)wset[idPlusOffset]);
    *HeTM_knl_global.isInterConfl = 1;
  }
  // if (isNewWrite) {
  //   a[id] = b[id];
  // }
}

__global__ void HeTM_knl_checkTxBitmap_Explicit(HeTM_knl_cmp_args_s args)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int sizeRSet = args.sizeRSet / sizeof(PR_GRANULE_T);
  unsigned char *wset = (unsigned char*)HeTM_knl_global.hostWSet; /* Host log */
  int *devLogR = (int*)HeTM_knl_global.devRSet;  /* GPU read log */
  int jobSize = sizeRSet / (blockDim.x*gridDim.x);
  size_t GPUIndexMaxVal = HeTM_knl_global.nbGranules;
  int i;
  jobSize++; // --> case mod is not zero
  for (i = 0; i < jobSize; ++i) {
    // TODO: review formula
    int idRSet = id*jobSize + i;
    if (idRSet >= sizeRSet || *HeTM_knl_global.isInterConfl) {
      return;
    }
    int GPUIndex = devLogR[idRSet] - 1;
    // TODO
    if (GPUIndex > -1 && GPUIndex < GPUIndexMaxVal && wset[GPUIndex] == 1) {
      *HeTM_knl_global.isInterConfl = 1;
    }
  }
}

__global__ void HeTM_knl_writeTxBitmap(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x + offset;

  // TODO: these must be the same
  int sizeWSet = args.sizeWSet; /* Size of the host log */
  // int sizeRSet = args.sizeRSet; /* Size of the device log */

  if (sizeWSet < id) return; // the thread has nothing to do

  // unsigned char *rset = (unsigned char*)HeTM_knl_global.devRSet;
  unsigned char *wset = (unsigned char*)HeTM_knl_global.hostWSet;
  PR_GRANULE_T *mempool = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
  PR_GRANULE_T *backup  = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBackupBasePtr;

  // TODO: use shared memory
  unsigned char isNewWrite = wset[id];

  long condToIgnore = !isNewWrite;
  condToIgnore = ((condToIgnore | (-condToIgnore)) >> 63);
  // long maskIgnore = -condToIgnore;
  long maskIgnore = condToIgnore; // TODO: for some obscure reason this is -1

  // applies -1 if address is invalid OR the index if valid
  mempool[id] = (maskIgnore & mempool[id]) | ((~maskIgnore) & backup[id]);
  // if (isNewWrite) {
  //   a[id] = b[id];
  // }
}

#else /* HETM_LOG_TYPE != HETM_BMAP_LOG */

__device__ void applyHostWritesOnDev(
  HeTM_CPULogEntry *fetch_stm, uintptr_t GPU_addr, int index
);

// ---------------------- COMPRESSED
__device__ void checkTx_versionLOG(
  int batchCount,
  int sizeWSet,
  int sizeRSet,
  int idCPUThread,
  int doApply
);
// ---------------------------------

/****************************************
*  HeTM_knl_checkTxCompressed()
*
*  Description: Compare device write-log with host log, when using compressed log
*
****************************************/
__global__ void HeTM_knl_checkTxCompressed(HeTM_knl_cmp_args_s args)
{
  int sizeWSet = args.sizeWSet; /* Size of the host log */
  int sizeRSet = args.sizeRSet; /* Size of the device log */
  int idCPUThr = args.idCPUThr; /* Size of the device log */
  int batchCount = args.batchCount;

  checkTx_versionLOG(batchCount, sizeWSet, sizeRSet, idCPUThr, 1);
}

// checks for conflicts but does not apply
__global__ void HeTM_knl_earlyCheckTxCompressed(HeTM_knl_cmp_args_s args)
{
  int sizeWSet = args.sizeWSet; /* Size of the host log */
  int sizeRSet = args.sizeRSet; /* Size of the device log */
  int idCPUThr = args.idCPUThr; /* Size of the device log */
  int batchCount = args.batchCount; // TODO: is 0

  checkTx_versionLOG(batchCount, sizeWSet, sizeRSet, idCPUThr, 0);
}

__global__ void HeTM_knl_checkTxExplicit(HeTM_knl_cmp_args_s args)
{
  HeTM_CPULogEntry *stm_log = (HeTM_CPULogEntry*)HeTM_knl_global.hostWSet; /* Host log */
  int sizeWSet = args.sizeWSet; /* Size of the host log */
  int sizeRSet = args.sizeRSet; /* Size of the host log */
  int *devLogR = (int*)HeTM_knl_global.devRSet;  /* GPU read log */

  // blockDim.y --> chunk of the CPU WSet (e.g., 32 entries),
  // i.e., gridDim.y == SizeWSet/blockDim.y

  // blockDim.x*blockDim.y == block of entries in the GPU rSet

  __shared__ int sharedRSet[CMP_EXPLICIT_THRS_PER_RSET];

  int threadsPerRSetBlock = blockDim.x*blockDim.y;
  int idRSet = threadIdx.x + threadIdx.y*blockDim.x;
  int rSetBlockIdx = blockIdx.x*threadsPerRSetBlock;

  // TODO: Either apply the CPU WSet here or create other kernel to do that

  if (idRSet + rSetBlockIdx >= sizeRSet) return;

  sharedRSet[idRSet] = devLogR[idRSet + rSetBlockIdx] - 1; /* 0 is for NULL */
  __syncthreads();

  int idWSet = threadIdx.y + blockIdx.y*blockDim.y;
  uintptr_t hostBasePtr = (uintptr_t)HeTM_knl_global.hostMemPoolBasePtr;
  int CPU_index = ((uintptr_t)stm_log[idWSet].pos - hostBasePtr) >> PR_LOCK_GRAN_BITS;

  if (idWSet >= sizeWSet) return;

  int i;
  for (i = 0; i < blockDim.y; ++i) {
    int idRSetWarp = (idRSet + CMP_EXPLICIT_THRS_PER_WSET*i) % threadsPerRSetBlock;
    int GPU_index = sharedRSet[idRSetWarp]; /* TODO: use __shfl */

    if (CPU_index == GPU_index) {
      *HeTM_knl_global.isInterConfl = 1;
      break;
    }
  }

  if (*HeTM_knl_global.isInterConfl == 0) {
    // TODO
    // applyHostWritesOnDev(fetch_stm, GPU_addr, index);
  }
}

// TODO: need comments
__device__ void checkTx_versionLOG(
  int batchCount,
  int sizeWSet,
  int sizeRSet,
  int idCPUThread,
  int doApply
) {
  int id;
  uintptr_t index = 0;
  uintptr_t GPU_addr = 0;
  HeTM_CPULogEntry *fetch_stm = NULL;
  int fetch_gpu = 0;
  unsigned char *logBM = (unsigned char*)HeTM_knl_global.devRSet;
  PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;

  HeTM_CPULogEntry *stm_log = (HeTM_CPULogEntry*)HeTM_knl_global.hostWSet;
  int nbEntriesLogBuffer = STM_LOG_BUFFER_SIZE * LOG_SIZE;
  stm_log += nbEntriesLogBuffer * idCPUThread;

  id = blockIdx.x*blockDim.x+threadIdx.x;

  if (id >= sizeWSet) return;

  fetch_stm = &stm_log[id];

  // Find the bitmap index for the CPU access
  // HeTM_hostMemPoolBasePtr is the base address of the bank accounts
  // fetch_stm->pos is the CPU address (then converted to an index)
  index = fetch_stm->pos;

  if (index == 0) return; // not defined

  index -= 1; // 0 is default value for empty

  GPU_addr = (uintptr_t)&(a[index]); // TODO: change the name "a" to accounts
  fetch_gpu = ByteM_GET_POS(index, logBM); //logBM[index];

  if (fetch_gpu == batchCount && *HeTM_knl_global.isInterConfl != 1) {
    // CPU and GPU conflict in some address
    // printf("found confl on index=%lu id=%i (last 4: %i %i %i %i)\n", index, id,
    //   stm_log[id-4].pos, stm_log[id-3].pos, stm_log[id-2].pos, stm_log[id-1].pos);
    // printf("Abort on index = %i doApply = %i\n", index, doApply);
    *HeTM_knl_global.isInterConfl = 1;
  }
  if (doApply) { applyHostWritesOnDev(fetch_stm, GPU_addr, index); }
}

__device__ void applyHostWritesOnDev(
  HeTM_CPULogEntry *fetch_stm, uintptr_t GPU_addr, int index
) {
#if HETM_LOG_TYPE == HETM_VERS_LOG
  // TODO: low resolution timer! Needs to deal with clock wraps
  uint32_t ts = fetch_stm->time_stamp;
  uint32_t ts_lock = (ts << 1) + 1;
  uint32_t ts_unlock = (ts << 1);
  uint32_t oldVers;
  uint32_t *vers = (uint32_t*)HeTM_knl_global.versions;
  PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
  PR_GRANULE_T *b = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBackupBasePtr;

  oldVers = vers[index];
  fetch_stm->pos = 0;
  while (oldVers < ts_unlock) { // we have fresher results
    if (atomicCAS(&(vers[index]), oldVers, ts_lock) == oldVers) {
      // locked! apply the results and unlock

      a[index] = fetch_stm->val;
      b[index] = fetch_stm->val;
      vers[index] = fetch_stm->time_stamp; // set GPU version

      char *bytes = (char*)HeTM_knl_global.devMemPoolBackupBmap;

      // index has granularity sizeof(int) --> transform it
      int chunkIdx = (index << 2) >> DEFAULT_BITMAP_GRANULARITY_BITS;
      bytes[chunkIdx] = 1;

      // unlock!
      atomicCAS(&(vers[index]), ts_lock, ts_unlock);
      break;
    }
    oldVers = vers[index];
  }

  // TODO: backup below
  //
  // while (fetch_stm) { // loops while can't grab the lock
  //   /* */
  //   //- TODO: DO NOT USE THE PR-STM LOCKS! --> we
  //   // probably can get away with the version table
  //   long *vers = (long*)HeTM_knl_global.versions;
  //   PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
  //   PR_GRANULE_T *b = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBackupBasePtr;
  //   int *mutex = HeTM_knl_global.PRLockTable;
  //   int *mux     = (int*)&(PR_GET_MTX(mutex, GPU_addr));
  //   int val      = *mux; // atomic read
  //   int isLocked = PR_CHECK_LOCK(val) || PR_CHECK_PRELOCK(val);
  //
  //   if (isLocked) continue;
  //
  //   int pr_version = PR_GET_VERSION(val);
  //   int pr_owner   = PR_GET_OWNER(val);
  //   int lockVal    = PR_LOCK_VAL(pr_version, pr_owner);
  //   // ---------
  //   // - this is what is consuming more time executing (and causing divergency)
  //   // ------------------------------------------------------------------------
  //   if (atomicCAS(mux, val, lockVal) == val) { // LOCK
  //     // if the comming write is more recent apply, if not ignore
  //     if (fetch_stm->time_stamp > vers[index]) {
  //
  //       // apply on main dataset and on snapshot --> CPU wins always
  //       a[index] = fetch_stm->val;
  //       b[index] = fetch_stm->val;
  //       vers[index] = fetch_stm->time_stamp; // set GPU version
  //
  //       // memman_bmap_s *key_bmap = (memman_bmap_s*)HeTM_knl_global.devMemPoolBackupBmap;
  //       // char *bytes = key_bmap->dev;
  //       char *bytes = (char*)HeTM_knl_global.devMemPoolBackupBmap;
  //
  //       // index has granularity sizeof(int) --> transform it
  //       int chunkIdx = (index << 2) >> DEFAULT_BITMAP_GRANULARITY_BITS;
  //       bytes[chunkIdx] = 1;
  //     }
  //     atomicCAS(mux, lockVal, 0); // UNLOCK
  //     break;
  //   } else if (fetch_stm->time_stamp <= vers[index]) {
  //     break;
  //   }
  //   // ------------------------------------------------------------------------
  //   // break;
  // }
#endif
}
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */

void HeTM_set_global_arg(HeTM_knl_global_s arg)
{
  CUDA_CHECK_ERROR(
    cudaMemcpyToSymbol(HeTM_knl_global, &arg, sizeof(HeTM_knl_global_s)),
    "");
  // printf("HeTM_knl_global.hostWSet = %p\n", HeTM_knl_global.hostWSet);
}
