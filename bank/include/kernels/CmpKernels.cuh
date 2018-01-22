#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmp_kernels.cuh"
#include "pr-stm-wrapper.cuh"

#define LOG_ARGS_DECL \
  int *dev_flag,         /*Flag in global memory*/ \
  int size_stm,          /*Size of the host log*/ \
  int size_log,          /*Size of the device log*/ \
  int *mutex,            /*Lock array*/ \
  int *devLogR,          /*GPU read log */ \
  account_t *a,          /*Values array*/ \
  account_t *b,          /*Host values array*/ \
  HeTM_CPULogEntry *stm_log,   /*Host log*/ \
  long *vers             /*Version*/

#define LOG_ARGS_CALL \
  dev_flag, size_stm, size_log, mutex, devLogR, a, b, stm_log, vers

// ---------------------- COMPRESSED
// TODO: TOO MANY ARGUMENTS!!!
__device__ void applyHostWritesOnDev(
  HeTM_CPULogEntry *fetch_stm, int * dev_flag, int * mutex, uintptr_t GPU_addr,
  long * vers, account_t *a, int index
);
__device__ void checkTx_versionLOG(LOG_ARGS_DECL);
// ---------------------------------


// ---------------------- EXPLICIT
// -------------------------------


/****************************************
 *  HeTM_knl_checkTxCompressed()
 *
 *  Description: Compare device write-log with host log, when using compressed log
 *
 ****************************************/
__global__ void HeTM_knl_checkTxCompressed(HeTM_knl_checkTxCompressed_s args)
{
  // TODO: refactor
  int *dev_flag       = args.dev_flag; /* Flag in global memory */
  HeTM_CPULogEntry *stm_log = args.stm_log;  /* Host log */
  int size_stm        = args.size_stm; /* Size of the host log */
  int size_log        = args.size_log; /* Size of the device log */
  int *mutex          = args.mutex;    /* Lock array */
  int *devLogR        = args.devLogR;  /* GPU read log */
  account_t *a        = args.a;        /* Values array */
  account_t *b        = args.b;        /* Host values array */
  long *vers          = args.vers;     /* Version */

  checkTx_versionLOG(LOG_ARGS_CALL);
}

__global__ void HeTM_knl_checkTxExplicit(HeTM_knl_checkTxExplicit_s args)
{
  int *dev_flag       = args.dev_flag; /* Flag in global memory */
  HeTM_CPULogEntry *stm_log = (HeTM_CPULogEntry*)args.stm_log; /* Host log */
  int size_stm        = args.size_stm; /* Size of the host log */
  int size_logR       = args.size_logR; /* Size of the host log */
  int *devLogR        = args.devLogR;  /* GPU read log */
  // int *mutex          = args.mutex;
  // account_t *a        = args.a;        /* Values array */
  // account_t *b        = args.b;        /* Values array */
  // long *vers          = args.vers;     /* Version */
  // -----

  // blockDim.y --> chunk of the CPU WSet (e.g., 32 entries),
  // i.e., gridDim.y == SizeWSet/blockDim.y

  // blockDim.x*blockDim.y == block of entries in the GPU rSet

  __shared__ int sharedRSet[CMP_EXPLICIT_THRS_PER_RSET];

  int threadsPerRSetBlock = blockDim.x*blockDim.y;
  int idRSet = threadIdx.x + threadIdx.y*blockDim.x;
  int rSetBlockIdx = blockIdx.x*threadsPerRSetBlock;

  // TODO: Either apply the CPU WSet here or create other kernel to do that

  if (idRSet + rSetBlockIdx >= size_logR) return;

  sharedRSet[idRSet] = devLogR[idRSet + rSetBlockIdx] - 1; /* 0 is for NULL */
  __syncthreads();

  int idWSet = threadIdx.y + blockIdx.y*blockDim.y;
  int CPU_index = ((uintptr_t)stm_log[idWSet].pos - (uintptr_t)dev_basePoint) >> PR_LOCK_GRAN_BITS;

  if (idWSet >= size_stm) return;

  for (int i = 0; i < blockDim.y; ++i) {
    int idRSetWarp = (idRSet + CMP_EXPLICIT_THRS_PER_WSET*i) % threadsPerRSetBlock;
    int GPU_index = sharedRSet[idRSetWarp]; /* TODO: use __shfl */

    if (CPU_index == GPU_index) {
      *dev_flag = 1;
      break;
    }
  }

  if (*dev_flag == 0) {
    // TODO: apply to global memory (expensive here)
    // applyHostWritesOnDev(fetch_stm, dev_flag, mutex, GPU_addr, vers, a, index);
  }
}

// TODO: TOO MANY ARGUMENTS!!!
__device__ void applyHostWritesOnDev(
  HeTM_CPULogEntry *fetch_stm, int *dev_flag, int *mutex, uintptr_t GPU_addr,
  long *vers, account_t *a, int index
) {
  int flag = 1;
  while (flag && fetch_stm && !(*dev_flag)) {
    int *mux     = (int*)&(PR_GET_MTX(mutex, GPU_addr));
    int val      = *mux; // atomic read
    int isLocked = PR_CHECK_LOCK(val) || PR_CHECK_PRELOCK(val);

    if (isLocked) continue;

    int pr_version = PR_GET_VERSION(val);
    int pr_owner   = PR_GET_OWNER(val);
    int lockVal    = PR_LOCK_VAL(pr_version, pr_owner);

    /* */
    if (atomicCAS(mux, val, lockVal) == val) { // LOCK
      // if the comming write is more recent apply, if not ignore
      if (fetch_stm->time_stamp > vers[index]) {
        a[index] = fetch_stm->val;
        vers[index] = fetch_stm->time_stamp; // set GPU version
      }
      atomicCAS(mux, lockVal, 0); // UNLOCK
      flag = 0; // break;
    } else {
      // in-GPU account is fresher
      if (fetch_stm->time_stamp <= vers[index]) {
        flag = 0; // break;
      }
    }
    /* */

  }
}

// TODO: need comments
__device__ void checkTx_versionLOG(LOG_ARGS_DECL)
{
  int id;
  uintptr_t offset = 0, index = 0;
  uintptr_t GPU_addr = 0, CPU_addr = 0;
  HeTM_CPULogEntry *fetch_stm = NULL;
  int fetch_gpu = 0;
  unsigned short *logBM = (unsigned short*)devLogR;

  id = blockIdx.x*blockDim.x+threadIdx.x;

  if ((*dev_flag) == 0 && id < size_stm) { //Check for currently running comparisons
    fetch_stm = &stm_log[id];

    // Find the bitmap index for the CPU access
    // dev_basePoint is the base address of the bank accounts
    // fetch_stm->pos is the CPU address (then converted to an index)
    CPU_addr = (uintptr_t)fetch_stm->pos;
    offset   = CPU_addr - (uintptr_t)dev_basePoint;
    GPU_addr = offset + (uintptr_t)a; // TODO: change the name "a" to accounts
    index    = offset >> PR_LOCK_GRAN_BITS;
    fetch_gpu = ByteM_GET_POS(index, logBM); //logBM[index];

    if (fetch_gpu != 0) {
      // CPU and GPU conflict in some address
      (*dev_flag) = 1;
    }

    applyHostWritesOnDev(fetch_stm, dev_flag, mutex, GPU_addr, vers, a, index);
  }
}
