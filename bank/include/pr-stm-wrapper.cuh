#ifndef PR_STM_WRAPPER_H_GUARD_
#define PR_STM_WRAPPER_H_GUARD_

#include "hetm-types.h"

#define PR_GRANULE_T        account_t
#define PR_LOCK_GRANULARITY 4 /* size in bytes */
#define PR_LOCK_GRAN_BITS   2 /* log2 of the size in bytes */

#define PR_LOCK_TABLE_SIZE  0x800000

//
#ifdef BENCH_MEMCD
#define PR_MAX_RWSET_SIZE   NUMBER_WAYS
#else /* !BENCH_MEMCD */
#define PR_MAX_RWSET_SIZE   BANK_NB_TRANSFERS
#endif /* BENCH_MEMCD */

#include "pr-stm.cuh"

#define PR_rand(n) \
	PR_i_rand(args, n) \
//

__global__ void setupKernel(void *args);
__device__ unsigned PR_i_rand(pr_tx_args_dev_host_s args, unsigned n);

#endif /* PR_STM_WRAPPER_H_GUARD_ */
