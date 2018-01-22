#ifndef BANK_H
#define BANK_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef TM_STATISTICS3
#define TM_STATISTICS3
#endif

#include <sched.h>

#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "hetm-types.h"
#include "hetm.cuh"
#include "cuda_wrapper.h"
#include "shared.h"
#include "timer.h"
#include "memman.h"
#include "bitmap.h"

#define RO                              1
#define RW                              0

#if defined(TM_GCC)
# include "../../abi/gcc/tm_macros.h"
#elif defined(TM_DTMC)
# include "../../abi/dtmc/tm_macros.h"
#elif defined(TM_INTEL)
# include "../../abi/intel/tm_macros.h"
#elif defined(TM_ABI)
# include "../../abi/tm_macros.h"
#endif /* defined(TM_ABI) */

#if defined(TM_GCC) || defined(TM_DTMC) || defined(TM_INTEL) || defined(TM_ABI)
# define TM_COMPILER
/* Add some attributes to library function */
TM_PURE
void exit(int status);
TM_PURE
void perror(const char *s);
#else /* Compile with explicit calls to tinySTM */

#include "stm.h"
#include "mod_ab.h"
#include "log.h"
#include "cuda_defines.h"

/*
 * Useful macros to work with transactions. Note that, to use nested
 * transactions, one should check the environment returned by
 * stm_get_env() and only call sigsetjmp() if it is not null.
 */

#ifndef USE_TSX_IMPL
// goes for TinySTM
#define TM_START(tid, ro)     { stm_tx_attr_t _a = {{.id = tid, .read_only = ro}}; sigjmp_buf *_e = stm_start(_a); if (_e != NULL) sigsetjmp(*_e, 0)
#define TM_LOAD(addr)         stm_load((stm_word_t *)addr)
#define TM_STORE(addr, value) stm_store((stm_word_t *)addr, (stm_word_t)value)
#define TM_COMMIT             stm_commit(); }

#define TM_GET_LOG(p)         stm_get_stats("HeTM_CPULog", &p);
#define TM_LOG(val)           stm_log_add(0,val)
#define TM_LOG2(pos,val)      stm_log_add(pos,val)
#define TM_FREE(point)        stm_logel_free(point)

#define TM_INIT(nb_threads)   stm_init(); mod_ab_init(0, NULL)
#define TM_EXIT               stm_exit()
#define TM_INIT_THREAD(p,s)   stm_init_thread(); stm_log_init_bm(p,s)
#define TM_EXIT_THREAD        stm_exit_thread()
#else /* USE_TSX_IMPL */
// redefine with TSX
#include "tsx_impl.h"
#define TM_START(tid, ro) 		HTM_SGL_begin();
#define TM_COMMIT 		        HTM_SGL_commit();
#define TM_LOAD(addr)         HTM_SGL_read(addr)
#define TM_STORE(addr, value) HTM_SGL_write(addr, value)

#define TM_GET_LOG(p)         HeTM_get_log(&p)
#define TM_LOG(val)           /*stm_log_add(0,val)*/
#define TM_LOG2(pos,val)      /*stm_log_add(pos,val)*/
#define TM_FREE(point)        stm_logel_free(point)

#define TM_INIT(nb_threads)		HTM_init(nb_threads); stm_init(); mod_ab_init(0, NULL)
#define TM_EXIT               HTM_exit(); stm_exit()
#define TM_INIT_THREAD(p,s)   HTM_SGL_init_thr(); stm_init_thread(); stm_log_init_bm(p, s)
#define TM_EXIT_THREAD        HTM_SGL_exit_thr(); stm_exit_thread()

#endif /* USE_TSX_IMPL */

#endif /* Compile with explicit calls to tinySTM */

#define DEFAULT_DURATION       5000
#define DEFAULT_NB_ACCOUNTS    100000
#define DEFAULT_NB_THREADS     4
#define DEFAULT_READ_ALL       0
#define DEFAULT_SEED           0
#define DEFAULT_WRITE_ALL      0
#define DEFAULT_READ_THREADS   0
#define DEFAULT_WRITE_THREADS  0
#define DEFAULT_DISJOINT       0
#define DEFAULT_OUTPUT_FILE    Bank_LOG
#define DEFAULT_ITERATIONS     1
#define DEFAULT_NB_TRFS        1
#define DEFAULT_NB_GPU_THRS    DEFAULT_threadNum // cuda_defines.h
#define DEFAULT_NB_GPU_BLKS    DEFAULT_blockNum

#ifndef MAX_THREADS
#define MAX_THREADS            128
#endif

#define MAX_ITER               100

#ifndef CPU_PART
#define	CPU_PART               0.6 // percentage of accessible granule pool for the CPU (from the end)
#endif
#ifndef GPU_PART
#define GPU_PART               0.6 // percentage of accessible granule pool for the GPU (from the start)
#endif
#ifndef P_INTERSECT
#define P_INTERSECT            0.5 // percentage of accessible granule pool for the GPU (from the start)
#endif

#define CMP_COMPRESSED 1
#define CMP_EXPLICIT   2
#ifndef CMP_TYPE
#define CMP_TYPE CMP_COMPRESSED
#endif

// #ifndef GPUEn
// #pragma message "GPUEn not defined"
// #define GPUEn                  1 // Use GPU acceleration
// #endif
// #ifndef CPUEn
// #pragma message "CPUEn not defined"
// #define CPUEn                  1 // Use CPU processing
// #endif

#define XSTR(s)                STR(s)
#define STR(s)                 #s

#ifndef TAG
#define TAG                    1
#endif

#define NODEBUG                0 // Set to 1 to kill prints for obtained results
#define RUN_ONCE               0 // Set to 1 to run only one cuda cycle

//Uncomment to use a data structure instead of an array.
// #define USE_ARRAY // ACCOUNTS in struct is nuked!!!

//Set to 1 to enable compressed transfer
#define BM_TRANSF               1

//Uncomment to allow CPU invalidation
//#define CPU_INV

//Uncomment to enable cpu/gpu balancing
//#define SYNC_BALANCING
#ifndef SYNC_BALANCING_VALS
#define	SYNC_BALANCING_VALS     1
#endif /* SYNC_BALANCING_VALS */
#ifndef SYNC_BALANCING_VALF
#define SYNC_BALANCING_VALF     1
#endif /* SYNC_BALANCING_VALF */

//Uncoment to uses single threaded comparison
// #define USE_STREAM
// TODO: now is not working for more than 1 stream
// Idea to solve: each time a log needs comparison, a
//                GPU buffer is created on demand
#define STREAM_COUNT            1 // 32 /* TODO: allocates STREAM_COUNT * nb_threads logs in GPU */

#define LOCK_VAL                4

#define RESET_MUTEX(val, mutex) \
	val = 0; \
	__sync_synchronize(); \
//

//#define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT printf
#else /* DEBUG */
#define DEBUG_PRINT(...)
#endif /* DEBUG */

// TODO: use this or rand_r(&seed) instead of erand48
#define RAND_R_FNC(seed) ({ \
	register unsigned long next = seed; \
	register unsigned long result; \
	next *= 1103515245; \
	next += 12345; \
	result = (unsigned long) (next / 65536) % 2048; \
	next *= 1103515245; \
	next += 12345; \
	result <<= 10; \
	result ^= (unsigned long) (next / 65536) % 1024; \
	next *= 1103515245; \
	next += 12345; \
	result <<= 10; \
	result ^= (unsigned long) (next / 65536) % 1024; \
	seed = next; \
	result; \
})

/* ################################################################### *
 * THREAD SRTUCTURE
 * ################################################################### */

typedef struct thread_data thread_data_t;

struct thread_data {

	// TODO: this is shared between benchmarks --> each bench gets its struct
	bank_t *bank;
	memcd_t *memcd;

	char *filename;
	unsigned long reads, writes, updates, aborts;
	int GPUthreadNum;
	int GPUblockNum;
  cuda_t * cd; // GPUEn==1
  unsigned long nb_transfer;
  unsigned long nb_transfer_gpu_only;
  unsigned long nb_read_all;
  unsigned long nb_write_all;
  char *fn;

/* #ifndef TM_COMPILER */
// statistics
  char *cm;
  unsigned long nb_aborts;
  unsigned long nb_aborts_1;
  unsigned long nb_aborts_2;
  unsigned long nb_aborts_locked_read;
  unsigned long nb_aborts_locked_write;
  unsigned long nb_aborts_validate_read;
  unsigned long nb_aborts_validate_write;
  unsigned long nb_aborts_validate_commit;
  unsigned long nb_aborts_invalid_memory;
  unsigned long nb_aborts_killed;
  unsigned long locked_reads_ok;
  unsigned long locked_reads_failed;
  unsigned long max_retries;
/* #endif !TM_COMPILER */

  unsigned int seed;
  int id;
  double duration;
  double duration2;
  int nb_accounts;
  int nb_threads;
  int nb_threadsCPU;
  int read_all;
  int read_threads;
  int write_all;
  int write_threads;
  int iter;
  int trfs;
  int trans;

	TIMER_T start;
	TIMER_T end;
	TIMER_T final; // final is a C++ keyword (exception handling)
  struct timespec timeout;
  double throughput;
	double tot_throughput[MAX_ITER];
  double tot_duration[MAX_ITER];
	double tot_duration2[MAX_ITER];
  unsigned long tot_commits[MAX_ITER];
	unsigned long tot_aborts[MAX_ITER];
	HeTM_CPULogNode_t *HeTM_CPULog;
	long global_commits;
/* #if GPUEn == 1 */
  cuda_t *cuda_st;
  double throughput_gpu;
  double throughput_gpu_only;
	double tot_throughput_gpu[MAX_ITER];
	double tot_throughput_gpu_only[MAX_ITER];
  unsigned long tot_comp[MAX_ITER];
	unsigned long tot_tx[MAX_ITER];
	unsigned long tot_cuda[MAX_ITER];
  unsigned long tot_loop[MAX_ITER];
	unsigned long tot_loops[MAX_ITER];
	unsigned long tot_trf2cpu[MAX_ITER];
	unsigned long tot_trf2gpu[MAX_ITER];
	unsigned long tot_trfcmp[MAX_ITER];
  unsigned long tot_commits_gpu[MAX_ITER];
	unsigned long tot_aborts_gpu[MAX_ITER];
/* #endif GPUEn == 1 */

	// TODO: memcached GPU (in the future create two benchmarks without duplicating every thing)
	int set_percent;
	int shared_percent;
	int num_ways;

	struct thread_data *dthreads;

  char padding[64];

} __attribute__((packed));

// GLOBALS
extern long long int global_fix;
extern thread_data_t parsedData; // input data is here

// functions

void bank_parseArgs(int argc, char **argv, thread_data_t *data);
void bank_check_params(thread_data_t *data);
void bank_printStats(thread_data_t *data);
void bank_statsFile(thread_data_t *data);
void bank_between_iter(thread_data_t *data, int iter);

/* ################################################################### *
 * Sorting
 * ################################################################### */

int compare_int (const void *a, const void *b);
int compare_double (const void *a, const void *b);

// -------------

int bank_sum(bank_t *bank);

/* ################################################################### *
 * Granule pool bounds
 * ################################################################### */

//   Memory access layout
// +------------------+--------------+
// |     GPU_PART     |  NOT ACCESS  |
// +------------------+--------------+
// +------------+--------------------+
// | NOT ACCESS |      CPU_PART      |
// +------------+--------------------+

#define GPU_BOT_IDX(size) \
 	(0) \
//
#define GPU_TOP_IDX(size) \
 	((int)(GPU_PART * (double)size)) \
//
#define CPU_BOT_IDX(size) \
 	((int)((1.0-CPU_PART)*(double)size)) \
//
#define CPU_TOP_IDX(size) \
 	(size-1) \
//
#define INTERSECT_BOT_IDX(size) CPU_BOT_IDX(size)
#define INTERSECT_TOP_IDX(size) GPU_TOP_IDX(size)

#define NO_INTERSECT_GPU_BOT_IDX(size) GPU_BOT_IDX(size)
#define NO_INTERSECT_GPU_TOP_IDX(size) CPU_BOT_IDX(size)
#define NO_INTERSECT_CPU_BOT_IDX(size) GPU_TOP_IDX(size)
#define NO_INTERSECT_CPU_TOP_IDX(size) CPU_TOP_IDX(size)

#define IS_INTERSECT_HIT(r) \
	((double)((int)(r)%100000)/100000.0 < P_INTERSECT)
//
#define RANDOM_ACCESS(r, bot, top) \
	(((int)(r) % (int)((top) - (bot))) + (int)(bot)) \
//
#define CPU_ACCESS(r, size) \
	RANDOM_ACCESS(r, NO_INTERSECT_CPU_BOT_IDX(size), NO_INTERSECT_CPU_TOP_IDX(size)) \
//
#define GPU_ACCESS(r, size) \
	RANDOM_ACCESS(r, NO_INTERSECT_GPU_BOT_IDX(size), NO_INTERSECT_GPU_TOP_IDX(size)) \
//
#define INTERSECT_ACCESS(r, size) \
	RANDOM_ACCESS(r, INTERSECT_BOT_IDX(size), INTERSECT_TOP_IDX(size)) \
//
#ifdef HETM_DEB
#define DEBUG_KERNEL() ({ \
  int countOnIntersection = 0; \
  size_t sizeOnIntersection = 0; \
	FILE *fp = fopen("bitmap.txt", "w"); \
	memman_select("Stats_OnIntersect"); \
  int *onIntersect = (int*)memman_get_cpu(&sizeOnIntersection); \
  for (i = 0; i < sizeOnIntersection / sizeof(int); ++i) { \
    countOnIntersection += onIntersect[i]; \
  } \
  printf("GPU hit the intersection %i times\n", countOnIntersection); \
  size_t bitmapSize; \
	memman_select("HeTM_dev_rset"); \
  void *bitmap = memman_get_cpu(&bitmapSize); \
  bitmap_print(bitmap, bitmapSize, fp); \
	fclose(fp); \
}) \
//
#else
#define DEBUG_KERNEL() /* empty */
#endif

#endif /* BANK_H */
