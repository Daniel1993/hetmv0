/* Include this file in your pr-stm-implementation file,
 * then use the MACROs accordingly!
 * Do not forget to include pr-stm-internal.cuh after this
 */
#ifndef PR_STM_WRAPPER_H_GUARD_
#define PR_STM_WRAPPER_H_GUARD_

#define PR_GRANULE_T        int // TODO
#define PR_LOCK_GRANULARITY 4 /* size in bytes */
#define PR_LOCK_GRAN_BITS   2 /* log2 of the size in bytes */

#define PR_LOCK_TABLE_SIZE  0x800000

// TODO: this is benchmark dependent
// #ifdef BENCH_MEMCD
// #define PR_MAX_RWSET_SIZE   NUMBER_WAYS
// #else /* !BENCH_MEMCD */
// #define PR_MAX_RWSET_SIZE   BANK_NB_TRANSFERS
// #endif /* BENCH_MEMCD */

#include "pr-stm.cuh"

#include "bitmap.h"
#include "memman.h"
#include "hetm.cuh"

#ifdef PR_ARGS_S_EXT
#undef PR_ARGS_S_EXT
#endif
#define PR_ARGS_S_EXT \
	typedef struct { \
		void *dev_rset; \
		void *dev_rset_cache; \
		long *state; /* TODO: this is benchmark specific */ \
		void *devMemPoolBasePtr, *hostMemPoolBasePtr; \
		void *devChunks; \
		memman_bmap_s *bmap; \
		memman_bmap_s *bmapBackup; \
		long batchCount; \
		int isGPUOnly; \
		HeTM_GPU_log_explicit_s /* Explicit log only */ \
	} HeTM_GPU_log_s \
//

#if HETM_CMP_TYPE == HETM_CMP_EXPLICIT
/* TODO: need logPos (do every thread commit the same number of transactions?) */
#define HeTM_GPU_log_explicit_s \
	unsigned explicitLogBlock; \
	unsigned *explicitLogOffThr; \
//
#define HeTM_GPU_log_explicit_prepare \
	size_t explicitLogCounter = PR_threadNum*PR_blockNum; \
	memman_alloc_gpu("HeTM_explicit_log_OffThrs", explicitLogCounter*sizeof(unsigned), NULL, MEMMAN_THRLC); \
	memman_select("HeTM_explicit_log_OffThrs"); \
	GPU_log->explicitLogOffThr = (unsigned*)memman_get_gpu(NULL); \
	GPU_log->explicitLogBlock = HeTM_get_explicit_log_block_size(); \
	memman_zero_gpu(NULL); \
//
// TODO: EXPLICIT_LOG_BLOCK depends on the benchmark (on bank is the number of TXs per thread)
#define HeTM_GPU_log_explicit_before_reads \
	int tid_ = blockIdx.x*blockDim.x + threadIdx.x; \
	long blockSize = tid_ * GPU_log->explicitLogBlock; \
	long logOffset = GPU_log->explicitLogOffThr[tid_]; \
	/*__shared__ int sharedLog[];*/ /* TODO: this needs to be dynamically alloc'ed */ \
//
#define HeTM_GPU_log_explicit_after_reads \
	GPU_log->explicitLogOffThr[tid_] += PR_MAX_RWSET_SIZE; \
//
#define HeTM_GPU_log_explicit_teardown \
	/*memman_select("HeTM_explicit_log_OffThrs"); \
	memman_free_dual();*/ \
//
#elif HETM_CMP_TYPE == HETM_CMP_COMPRESSED
#define HeTM_GPU_log_explicit_s                  /* empty */
#define HeTM_GPU_log_explicit_prepare            /* empty */
#define HeTM_GPU_log_explicit_before_reads       /* empty */
#define HeTM_GPU_log_explicit_after_reads        /* empty */
#define HeTM_GPU_log_explicit_teardown           /* empty */
#else
// error or disabled
#define HeTM_GPU_log_explicit_s                  /* empty */
#define HeTM_GPU_log_explicit_prepare            /* empty */
#define HeTM_GPU_log_explicit_before_reads       /* empty */
#define HeTM_GPU_log_explicit_after_reads        /* empty */
#define HeTM_GPU_log_explicit_teardown           /* empty */
#endif

#ifdef PR_DEV_BUFF_S_EXT
#undef PR_DEV_BUFF_S_EXT
#endif
#define PR_DEV_BUFF_S_EXT \
	typedef struct { \
		HeTM_GPU_log_s gpuLog; \
		long *state; \
	} HeTM_GPU_dbuf_log_s \

#ifdef PR_BEFORE_RUN_EXT
#undef PR_BEFORE_RUN_EXT
#endif

// TODO: each time we start a kernel HeTM_gpuLog must be copied!!!
#define PR_BEFORE_RUN_EXT(args) ({ \
	HeTM_GPU_log_s *GPU_log; \
	/* Fail on multiple allocs (memman_select needed) */ \
	memman_alloc_dual("HeTM_gpuLog", sizeof(HeTM_GPU_log_s), MEMMAN_THRLC); \
	memman_select("HeTM_gpuLog"); \
	GPU_log = (HeTM_GPU_log_s*)memman_get_cpu(NULL); \
	/* TODO: explicit log only */ \
	HeTM_GPU_log_explicit_prepare /* this selects other memory */ \
	/* ---------------------- */ \
	GPU_log->dev_rset = HeTM_shared_data.rsetLog; \
	GPU_log->dev_rset_cache = HeTM_shared_data.wsetCacheConfl; \
	GPU_log->devMemPoolBasePtr  = HeTM_shared_data.devMemPool; \
	GPU_log->hostMemPoolBasePtr = HeTM_shared_data.hostMemPool; \
	GPU_log->state = (long*)HeTM_shared_data.devCurandState; /* TODO: application specific */ \
	GPU_log->devChunks = HeTM_shared_data.devChunks; \
	GPU_log->batchCount = HeTM_shared_data.batchCount; \
	GPU_log->isGPUOnly = (HeTM_shared_data.isCPUEnabled == 0); \
	/* ---------------------- */ \
	memman_select("HeTM_mempool_bmap"); \
	memman_cpy_to_gpu(HeTM_memStream2, NULL, *hetm_batchCount); \
	memman_bmap_s *bmap = (memman_bmap_s*)memman_get_cpu(NULL); \
	GPU_log->bmap = (memman_bmap_s*)memman_get_gpu(NULL); \
	if ((HeTM_shared_data.batchCount & 0xff) == 1) { \
		CUDA_CHECK_ERROR(cudaMemsetAsync(bmap->dev, 0, bmap->div, (cudaStream_t)HeTM_memStream2), ""); \
	} \
	/* ---------------------- */ \
	memman_select("HeTM_mempool_backup_bmap"); \
	memman_cpy_to_gpu(HeTM_memStream2, NULL, *hetm_batchCount); \
	memman_bmap_s *bmapBackup = (memman_bmap_s*)memman_get_cpu(NULL); \
	GPU_log->bmapBackup = (memman_bmap_s*)memman_get_gpu(NULL); \
	if ((HeTM_shared_data.batchCount & 0xff) == 1) { \
		CUDA_CHECK_ERROR(cudaMemsetAsync(bmapBackup->dev, 0, bmapBackup->div, (cudaStream_t)HeTM_memStream2), ""); \
  } \
	/* ---------------------- */ \
	args->host.pr_args_ext = (void*)GPU_log; \
	memman_select("HeTM_gpuLog"); \
	args->dev.pr_args_ext = memman_get_gpu(NULL); \
	memman_cpy_to_gpu(HeTM_memStream2, NULL, *hetm_batchCount); \
	cudaStreamSynchronize((cudaStream_t)HeTM_memStream2); \
}) \

#ifdef PR_AFTER_RUN_EXT
#undef PR_AFTER_RUN_EXT
#endif

// TODO: avoid this HeTM_gpuLog thing
#define PR_AFTER_RUN_EXT(args) ({ \
	HeTM_GPU_log_explicit_teardown; \
	/*memman_select("HeTM_gpuLog"); \
	memman_free_dual();*/ /* TODO: clean this memory */ \
}) \

// Logs the read-set after acquiring the locks
// TODO: check write/write conflicts

#if HETM_CMP_TYPE == HETM_CMP_EXPLICIT
/* TODO: need logPos (do every thread commit the same number of transactions?) */
#define SET_ON_LOG(addr) \
	int *explicitLog = (int*)GPU_log->dev_rset; \
	unsigned logPos = blockSize + logOffset; \
	uintptr_t rsetAddr = (uintptr_t)(addr); \
	uintptr_t devBAddr = (uintptr_t)GPU_log->devMemPoolBasePtr; \
	uintptr_t pos = (rsetAddr - devBAddr) >> PR_LOCK_GRAN_BITS; /* stores the index instead of the address */ \
	explicitLog[logPos + i] = pos + 1 /* 0 is NULL */ \
//
/*if (GPU_log->explicitLogOffThr[tid_]==98) printf("[%i] explicitLogOffset=%i, explicitLogOffThr=%i, i=%i\n", (int)tid_,\
(int)explicitLogOffset, (int)GPU_log->explicitLogOffThr[tid_], i);*/ \
#elif HETM_CMP_TYPE == HETM_CMP_COMPRESSED

#if HETM_LOG_TYPE == HETM_BMAP_LOG
#define SET_ON_BMAP \
	uintptr_t _pos_cache = _pos >> DEFAULT_BITMAP_GRANULARITY_BITS; \
	void *_RSetBitmap = GPU_log->dev_rset; \
	void *_RSetBitmap_cache = GPU_log->dev_rset_cache; \
	ByteM_SET_POS(_pos_cache, _RSetBitmap_cache, GPU_log->batchCount); \
	ByteM_SET_POS(_pos, _RSetBitmap, GPU_log->batchCount) \
//
#else /* VERS */
#define SET_ON_BMAP \
	void *_RSetBitmap = GPU_log->dev_rset; \
	ByteM_SET_POS(_pos, _RSetBitmap, GPU_log->batchCount) \
//
#endif

#ifndef HETM_REDUCED_RS
#define HETM_REDUCED_RS 0
#endif /* HETM_REDUCED_RS */

#define SET_ON_LOG(addr) \
	uintptr_t _rsetAddr = (uintptr_t)(addr); \
	uintptr_t _devBAddr = (uintptr_t)GPU_log->devMemPoolBasePtr; \
	uintptr_t _pos = (_rsetAddr - _devBAddr) >> (PR_LOCK_GRAN_BITS+HETM_REDUCED_RS); \
	SET_ON_BMAP \
//
#else
// error or disabled
#define SET_ON_LOG(addr) /* empty */
#endif

#ifdef PR_AFTER_VAL_LOCKS_EXT
#undef PR_AFTER_VAL_LOCKS_EXT
#endif
// also addes the write to the backup (case that we need to invalidate the CPU)
#if HETM_CMP_TYPE == HETM_CMP_DISABLED
#define PR_AFTER_VAL_LOCKS_EXT(args) /* empty */
#else /* HETM_CMP_TYPE != HETM_CMP_DISABLED */

// TODO: instrumentation in the GPU is reduced via "isGPUOnly" flag, which
// also afects the copies (copy all)

#ifdef HETM_DISABLE_RS
#define PR_AFTER_VAL_LOCKS_GATHER_READ_SET(i) /* empty */
#else /* !HETM_DISABLE_RS */
#define PR_AFTER_VAL_LOCKS_GATHER_READ_SET(i) \
	for (i = 0; i < args->rset.size; i++) { \
		SET_ON_LOG(args->rset.addrs[i]); \
	} \
//
#endif /* HETM_DISABLE_RS */

#ifdef HETM_DISABLE_WS
#define PR_AFTER_VAL_LOCKS_GATHER_WRITE_SET(i) /* empty */
#else /* !HETM_DISABLE_WS	 */
#define PR_AFTER_VAL_LOCKS_GATHER_WRITE_SET(i) \
	for (i = 0; i < args->wset.size; i++) { \
		/* this is avoided through a memcpy D->D after batch */ \
		memman_access_addr_dev(GPU_log->bmap, args->wset.addrs[i], GPU_log->batchCount); \
		SET_ON_LOG(args->rset.addrs[i]); \
	} \
//
#endif /* HETM_DISABLE_WS */

#define PR_AFTER_VAL_LOCKS_EXT(args) ({ \
  int i; \
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args->pr_args_ext; \
	/* TODO: explicit log only */ \
	/*if (!GPU_log->isGPUOnly) {*/ \
		HeTM_GPU_log_explicit_before_reads \
		/* ---------------------- */ \
		/* add read to devLogR */ \
		PR_AFTER_VAL_LOCKS_GATHER_READ_SET(i); \
		PR_AFTER_VAL_LOCKS_GATHER_WRITE_SET(i); \
		/* TODO: explicit logOnly */ \
		HeTM_GPU_log_explicit_after_reads /* offset of the next transaction */ \
		/* ---------------------- */ \
	/*}*/ \
}) \
//
#endif /* HETM_CMP_TYPE == HETM_CMP_DISABLED */

#ifdef PR_AFTER_WRITEBACK_EXT
#undef PR_AFTER_WRITEBACK_EXT
#endif

// Logs the GPU write-set after acquiring the locks
#define PR_AFTER_WRITEBACK_EXT(args, i, addr, val) ({ \
	/* TODO: not implemented */ \
	/* HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args->pr_args_ext;*/ \
}) \

#define PR_i_rand(args, n) ({ \
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext; \
	int id = PR_THREAD_IDX; \
	unsigned x; \
	long *state = GPU_log->state; \
	x = RAND_R_FNC(state[id]); \
	(unsigned) (x % n); \
}) \
//
#define PR_rand(n) \
	PR_i_rand(args, n) \
//

__global__ void HeTM_setupCurand(void *args);

#endif /* PR_STM_WRAPPER_H_GUARD_ */
