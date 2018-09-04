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

  // TODO: these must be the same
  int sizeWSet = args.sizeWSet; /* Size of the entire log */
  // int sizeRSet = args.sizeRSet; /* Size of the device log */

  if (id >= sizeWSet) return; // the thread has nothing to do

  unsigned char *rset = (unsigned char*)HeTM_knl_global.devRSet;
  unsigned char *wset = (unsigned char*)HeTM_knl_global.hostWSetCache;
  size_t wsetBits = HeTM_knl_global.hostWSetCacheBits;
  // PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
  // PR_GRANULE_T *b = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBackupBasePtr;

  // TODO: use shared memory
  int cacheId = id >> wsetBits;
  unsigned char isNewWrite = wset[cacheId];
  unsigned char isConfl = rset[id] && isNewWrite;

  if (isConfl) {
    *HeTM_knl_global.isInterConfl = 1;
    ((unsigned char*)(HeTM_knl_global.hostWSetCacheConfl))[cacheId] = 1;
  }
  // if (isNewWrite) {
  //   a[id] = b[id];
  // }
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
  // PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
  // PR_GRANULE_T *b = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBackupBasePtr;

  // TODO: use shared memory
  unsigned char isNewWrite = wset[idPlusOffset];
  unsigned char isConfl = rset[idPlusOffset] && isNewWrite;

  if (isConfl) {
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
  PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
  PR_GRANULE_T *b = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBackupBasePtr;

  // TODO: use shared memory
  unsigned char isNewWrite = wset[id];

  long condToIgnore = !isNewWrite;
  condToIgnore = ((condToIgnore | (-condToIgnore)) >> 63);
  // long maskIgnore = -condToIgnore;
  long maskIgnore = condToIgnore; // TODO: for some obscure reason this is -1

  // applies -1 if address is invalid OR the index if valid
  a[id] = (maskIgnore & a[id]) | ((~maskIgnore) & b[id]);
  // if (isNewWrite) {
  //   a[id] = b[id];
  // }
}

#else /* HETM_LOG_TYPE != HETM_BMAP_LOG */

__device__ void applyHostWritesOnDev(
  HeTM_CPULogEntry *fetch_stm, uintptr_t GPU_addr, int index
);

// ---------------------- COMPRESSED
__device__ void checkTx_versionLOG(int sizeWSet, int sizeRSet);
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

  checkTx_versionLOG(sizeWSet, sizeRSet);
}

#if HETM_LOG_TYPE == HETM_ADDR_LOG
// TODO: launch this after comparison
__global__ void HeTM_knl_apply_cpu_data(int amount, size_t nbGranules)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
  PR_GRANULE_T *b = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBackupBasePtr;
  char *vers = (char*)HeTM_knl_global.versions;

  // TODO: could use shared_memory + unified_memory

  int i;
for (i = 0; i < amount; ++i) {
    int idx = id * amount + i;

    if (idx > nbGranules) return; // exceeded array space

    // check in bitmap if matches (if yes update GPU dataset)
    if (vers[idx] == 1) {
      a[idx] = b[idx]; // TODO: either missing vers[idx] OR error here
    }
  }
}
#endif

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
#if HETM_LOG_TYPE == HETM_VERS2_LOG
// entries_per_thread == LOG_SIZE*STM_LOG_BUFFER_SIZE
__device__ void checkTx_versionLOG(int sizeWSet, int entries_per_thread)
{
  int id;
  uintptr_t offset = 0;
  HeTM_CPULogEntry *fetch_stm = NULL;
  int i;

  // this ID maps some positions in the dataset
  id = blockIdx.x*blockDim.x+threadIdx.x;
  offset = id * entries_per_thread;

  if (id >= LOG_ACTUAL_SIZE) return; // Prime number

  // Rset here is how many entries the thread is responsible for
  // size_t nbGranules = HeTM_knl_global.nbGranules;
  unsigned char *logBM = (unsigned char*)HeTM_knl_global.devRSet;
  PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
  long *vers = (long*)HeTM_knl_global.versions;
  int isInterConfl = 0;

  HeTM_CPULogEntry *stm_log = (HeTM_CPULogEntry*)HeTM_knl_global.hostWSet;

  // TODO: init shared memory with default value (if default is not changed
  // then do not copy to global memory) (CRASHES!)
  const int maxCache = LOG_SIZE*LOG_THREADS_IN_BLOCK; // TODO: use shared 64*32 (128 threads each thread handles 32 tops)
  __shared__ long cacheVersion[maxCache]; // NEED ADDRESS!!!!
  __shared__ int cacheValue[maxCache];
  // __shared__ long cacheAddr[maxCache]; // if not enough space go for private mem

  // ---------------------------------------------------------------
  // read from Global Memory
  #pragma unroll
  for (i = 0; i < LOG_SIZE; ++i) { // entries_per_thread == 16
    cacheVersion[threadIdx.x + i*blockDim.x] = 0; // default version (apply always)
    // cacheAddr[threadIdx.x + i*blockDim.x] = index;
    // if (CPU_addr == (uintptr_t)-1 || CPU_addr == 0) {
    //   cacheAddr[threadIdx.x + i*blockDim.x] = -1;
    // }
  }
  __syncthreads();
  // ---------------------------------------------------------------

  #pragma unroll
  for (i = 0; i < LOG_SIZE; ++i) { // entries_per_thread == 16 (else crashes)
    fetch_stm = &(stm_log[id + i*LOG_ACTUAL_SIZE]);

    // -------------------------------------------------------------
    // Compute index
    // uintptr_t CPU_addr = (uintptr_t)fetch_stm->pos; // TODO: now this is an offset
    // uintptr_t DTST_offset = CPU_addr - (uintptr_t)HeTM_knl_global.hostMemPoolBasePtr;
    // uintptr_t GPU_addr = DTST_offset + (uintptr_t)a;
    // long index = DTST_offset >> PR_LOCK_GRAN_BITS;
    long index = fetch_stm->pos;
    // long condToIgnore = (CPU_addr == 0 || CPU_addr == (uintptr_t)-1);
    long condToIgnore = (index == 0); // 0 is default value for empty
    condToIgnore = ((condToIgnore | (-condToIgnore)) >> 63);
    // long maskIgnore = -condToIgnore;
    long maskIgnore = condToIgnore; // TODO: for some obscure reason this is -1

    // applies -1 if address is invalid OR the index if valid
    index = (maskIgnore & (long)-1L) | ((~maskIgnore) & (index-1));
    // -------------------------------------------------------------

    // if ((*HeTM_knl_global.isInterConfl) == 1) return;

    long condIsIndexMinus1 = (index == -1);
    long accessInByteMap = ((condIsIndexMinus1 | (-condIsIndexMinus1)) >> 63);
    // fixedIndex for invalid entries --> 0 TS and val
    long fixedIndex = ((~accessInByteMap) & index); // 0 or index
    // if (index == -1) break;

    int fetch_gpu = ByteM_GET_POS(fixedIndex, logBM);

    isInterConfl = (fetch_gpu && !condIsIndexMinus1) || isInterConfl;

    // ---------------------------------------------------------------
    // if (ts > cacheTs) {
    //   cacheVersion[threadIdx.x + i*blockDim.x] = fetch_stm->time_stamp;
    //   cacheValue[threadIdx.x + i*blockDim.x] = fetch_stm->val;
    // }
    // ---------------------------------------------------------------
    long cacheTs  = vers[fixedIndex];
    long cacheVal = a[fixedIndex];

    long condToApply = (fetch_stm->time_stamp > cacheTs); // 0 or 1
    condToApply = (condToApply | (-condToApply)) >> 63;
    // 0 (all bits 0, means keep prev) or -1 (all bits 1, means keep next)
    // long maskApply = -condToApply;
    long maskApply = condToApply; // TODO: for some obscure reason this is -1

    // invalid entries, last value is applied
    cacheVersion[threadIdx.x + i*blockDim.x] = ((~maskApply) & cacheTs) | (maskApply & fetch_stm->time_stamp);
    cacheValue[threadIdx.x + i*blockDim.x] = ((~maskApply) & cacheVal) | (maskApply & fetch_stm->val);
    // ---------------------------------------------------------------
  }

  // ---------------------------------------------------------------
  __syncthreads();
  #pragma unroll
  for (i = 0; i < LOG_SIZE; ++i) { // entries_per_thread == 16
    fetch_stm = &(stm_log[id + i*LOG_ACTUAL_SIZE]);

    // -------------------------------------------------------------
    // Compute index
    uintptr_t CPU_addr = (uintptr_t)fetch_stm->pos;
    uintptr_t DTST_offset = CPU_addr - (uintptr_t)HeTM_knl_global.hostMemPoolBasePtr;
    // uintptr_t GPU_addr = DTST_offset + (uintptr_t)a;
    long index = DTST_offset >> PR_LOCK_GRAN_BITS;
    long condToIgnore = (CPU_addr == 0 || CPU_addr == (uintptr_t)-1);
    condToIgnore = ((condToIgnore | (-condToIgnore)) >> 63);
    // long maskIgnore = -condToIgnore;
    long maskIgnore = condToIgnore; // TODO: for some obscure reason this is -1

    // applies -1 if address is invalid OR the index if valid
    index = (maskIgnore & (long)-1L) | ((~maskIgnore) & index);

    // long index = cacheAddr[threadIdx.x + i*blockDim.x];
    // if (index == -1) break;

    long condIsIndexMinus1 = (index == -1);
    long accessInByteMap = ((condIsIndexMinus1 | (-condIsIndexMinus1)) >> 63);
    long fixedIndex = ((~accessInByteMap) & index);

    // TODO: concurrent kernels!!!
    // if(fixedIndex < 0 || fixedIndex > HeTM_knl_global.nbGranules) {
    //   printf("invalid: index=%li, pos=%p\n", fixedIndex, (void*)CPU_addr);
    // }
    vers[fixedIndex] = cacheVersion[threadIdx.x + i*blockDim.x];
    a[fixedIndex] = cacheValue[threadIdx.x + i*blockDim.x];

    // long assumed, old;
    // int oldVal;
    // oldVal = a[index]; // TODO: this one is tricky --> is it atomic?
    // old = vers[index];
    // do {
    //   if (cacheVersion[threadIdx.x + i*blockDim.x] < old) break;
    //   assumed = old;
    //   old = atomicCAS((unsigned long long int*)&(vers[index]),
    //     (unsigned long long int)assumed,
    //     (unsigned long long int)cacheVersion[threadIdx.x + i*blockDim.x]);
    // } while (assumed != old);
    // if (cacheVersion[threadIdx.x + i*blockDim.x] < old) continue;
    // atomicCAS((int*)&a[index], oldVal, (int)cacheValue[threadIdx.x + i*blockDim.x]);
  }

  if (isInterConfl) {
    *HeTM_knl_global.isInterConfl = 1;
  }
  // ---------------------------------------------------------------
}

#else /* HETM_LOG_TYPE != HETM_VERS2_LOG */
__device__ void checkTx_versionLOG(int sizeWSet, int sizeRSet)
{
  int id;
  uintptr_t offset = 0, index = 0;
  uintptr_t GPU_addr = 0, CPU_addr = 0;
  HeTM_CPULogEntry *fetch_stm = NULL;
  int fetch_gpu = 0;
  unsigned char *logBM = (unsigned char*)HeTM_knl_global.devRSet;
  PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;

  HeTM_CPULogEntry *stm_log = (HeTM_CPULogEntry*)HeTM_knl_global.hostWSet;

  id = blockIdx.x*blockDim.x+threadIdx.x;

  if ((*HeTM_knl_global.isInterConfl) == 0 && id < sizeWSet) { //Check for currently running comparisons
    fetch_stm = &stm_log[id];

    // Find the bitmap index for the CPU access
    // HeTM_hostMemPoolBasePtr is the base address of the bank accounts
    // fetch_stm->pos is the CPU address (then converted to an index)
    // CPU_addr = (uintptr_t)fetch_stm->pos;
    index = fetch_stm->pos;

    if (index == 0) return; // not defined

    index -= 1; // 0 is default value for empty

    // offset   = CPU_addr - (uintptr_t)HeTM_knl_global.hostMemPoolBasePtr;
    // GPU_addr = offset + (uintptr_t)a; // TODO: change the name "a" to accounts
    GPU_addr = (uintptr_t)&(a[index]); // TODO: change the name "a" to accounts
    // index    = offset >> PR_LOCK_GRAN_BITS;
    fetch_gpu = ByteM_GET_POS(index, logBM); //logBM[index];

    if (fetch_gpu != 0) {
      // CPU and GPU conflict in some address
      printf("found confl on index=%i\n", index);
      *HeTM_knl_global.isInterConfl = 1;
    }

    applyHostWritesOnDev(fetch_stm, GPU_addr, index);
  }
}

__device__ void applyHostWritesOnDev(
  HeTM_CPULogEntry *fetch_stm, uintptr_t GPU_addr, int index
) {
#if HETM_LOG_TYPE == HETM_VERS_LOG
  while (fetch_stm && !(*HeTM_knl_global.isInterConfl)) {
    /* */
    //- TODO: refactor!
    long *vers = (long*)HeTM_knl_global.versions;
    PR_GRANULE_T *a = (PR_GRANULE_T*)HeTM_knl_global.devMemPoolBasePtr;
    int *mutex = HeTM_knl_global.PRLockTable;
    int *mux     = (int*)&(PR_GET_MTX(mutex, GPU_addr));
    int val      = *mux; // atomic read
    int isLocked = PR_CHECK_LOCK(val) || PR_CHECK_PRELOCK(val);

    if (isLocked) continue;

    int pr_version = PR_GET_VERSION(val);
    int pr_owner   = PR_GET_OWNER(val);
    int lockVal    = PR_LOCK_VAL(pr_version, pr_owner);
    if (atomicCAS(mux, val, lockVal) == val) { // LOCK
      // if the comming write is more recent apply, if not ignore
      if (fetch_stm->time_stamp > vers[index]) {
        // int read = a[index];
        // if (atomicCAS(&a[index], read, fetch_stm->val) != read) {
        //   printf(" >>>> Error! concurrent access!\n");
        // }
        a[index] = fetch_stm->val; // TODO: apply is changing bank invariant!
        vers[index] = fetch_stm->time_stamp; // set GPU version
      }
      atomicCAS(mux, lockVal, 0); // UNLOCK
      break;
    } else if (fetch_stm->time_stamp <= vers[index]) {
      break;
    }
  }
#elif HETM_LOG_TYPE == HETM_ADDR_LOG
  char *vers = (char*)HeTM_knl_global.versions;
  // TODO: doesn't work! there is some bug here!
  // int mod = index & 0b11;
  // int div = index >> 2;
  // int res = 1 << mod;
  // atomicOr((int*)&(vers[div<<2]), res);
  vers[index] = 1; // in a final kernel merge with CPU dataset
#endif
}
#endif /* HETM_LOG_TYPE == HETM_VERS2_LOG */

#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */

void HeTM_set_global_arg(HeTM_knl_global_s arg)
{
  CUDA_CHECK_ERROR(
    cudaMemcpyToSymbol(HeTM_knl_global, &arg, sizeof(HeTM_knl_global_s)),
    "");
  // printf("HeTM_knl_global.hostWSet = %p\n", HeTM_knl_global.hostWSet);
}
