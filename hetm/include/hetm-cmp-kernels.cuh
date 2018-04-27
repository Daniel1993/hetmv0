#ifndef HETM_CMP_KERNELS_H_GUARD_
#define HETM_CMP_KERNELS_H_GUARD_

#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "cmp_kernels.cuh"
#include "pr-stm-wrapper.cuh"
#include "bitmap.h"
#include "hetm-log.h"

#define CMP_EXPLICIT_THRS_PER_RSET 1024 /* divide per CMP_EXPLICIT_THRS_PER_WSET */
#define CMP_EXPLICIT_THRS_PER_WSET 32 /* nb. of entries to compare */

/****************************************
 *	HeTM_knl_checkTxCompressed()
 *
 *	Description: Compare device write-log with host log, when using compressed log
 *
 ****************************************/
typedef struct HeTM_knl_cmp_args_ {
	int sizeWSet; /* Size of the host log */
	int sizeRSet; /* Size of the device log */
} HeTM_knl_cmp_args_s;

typedef struct HeTM_cmp_ {
	HeTM_knl_cmp_args_s knlArgs;
	HeTM_thread_s *clbkArgs;
} HeTM_cmp_s;

typedef struct HeTM_knl_global_ {
  void *devMemPoolBasePtr;
#if HETM_LOG_TYPE == HETM_ADDR_LOG || HETM_LOG_TYPE == HETM_BMAP_LOG
  void *devMemPoolBackupBasePtr;
#endif
  void *hostMemPoolBasePtr;
  void *versions;
  int *isInterConfl;
  size_t explicitLogBlock;
	size_t nbGranules;
  void *devRSet;
  void *hostWSet;
	int *PRLockTable;
	void *randState;
} HeTM_knl_global_s;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void HeTM_set_global_arg(HeTM_knl_global_s);

// set the size of the explicit log (benchmark dependent), it is equal to
// the number of writes of a thread (all transactions) in a GPU kernel
void HeTM_set_explicit_log_block_size(size_t);
size_t HeTM_get_explicit_log_block_size();

#ifdef __cplusplus
}
#endif /* __cplusplus */


#if HETM_LOG_TYPE == HETM_BMAP_LOG
// HETM_BMAP_LOG requires a specific kernel
__global__ void HeTM_knl_checkTxBitmap(HeTM_knl_cmp_args_s args);
#endif /* HETM_LOG_TYPE == HETM_BMAP_LOG */

#if HETM_LOG_TYPE != HETM_BMAP_LOG
// problems compiling the log
__global__ void HeTM_knl_checkTxCompressed(HeTM_knl_cmp_args_s args);
__global__ void HeTM_knl_checkTxExplicit(HeTM_knl_cmp_args_s args);
#endif /* HETM_LOG_TYPE != HETM_BMAP_LOG */

#if HETM_LOG_TYPE == HETM_ADDR_LOG
__global__ void HeTM_knl_apply_cpu_data(int amount, size_t nbGranules);
#endif /* HETM_LOG_TYPE == HETM_ADDR_LOG */

#endif /* HETM_CMP_KERNELS_H_GUARD_ */
