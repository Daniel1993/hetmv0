#ifndef BANK_H
#define BANK_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
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
#include "hetm-timer.h"
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
#include "hetm-log.h"
#include "cuda_defines.h"
#include "stm-wrapper.h"

#endif /* Compile with explicit calls to tinySTM */

#define DEFAULT_DURATION       5000
#define DEFAULT_NB_ACCOUNTS    100000
#define DEFAULT_NB_THREADS     4
#define DEFAULT_READ_ALL       0
#define DEFAULT_SEED           0
#define DEFAULT_HPROB          90
#define DEFAULT_HMULT          0.05
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
  double stddev;
  int hprob;
  double hmult;
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
	TIMER_T last;
  struct timespec timeout;
  double throughput;
	double tot_throughput[MAX_ITER];
  double tot_duration[MAX_ITER];
	double tot_duration2[MAX_ITER];
  unsigned long tot_commits[MAX_ITER];
	unsigned long tot_aborts[MAX_ITER];
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

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

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

#ifdef __cplusplus
}
#endif /* __cplusplus */

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

// create a workload with hotspots:
// --> major hotspot within a chunk (max 16MB)
// --> minor accesses in some other chunks
// --> zero accesses in remaining chunks


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
#define INTERSECT_ACCESS_GPU(r, size) \
	RANDOM_ACCESS(r, GPU_BOT_IDX(size), INTERSECT_TOP_IDX(size)) \
//
#define INTERSECT_ACCESS_CPU(r, size) \
	RANDOM_ACCESS(r, INTERSECT_BOT_IDX(size), CPU_TOP_IDX(size)) \
//

// if not "Hot" is Medium
#define IS_ACCESS_H(r, threshold) ({ \
	((r % 100) < threshold); \
})
//

#ifndef MAX
#define MAX(a,b) ({ \
	__typeof__ (a) _a = (a); \
  __typeof__ (b) _b = (b); \
  _a > _b ? _a : _b; \
})
#endif /* MAX */

#define MIN(a,b) ({ \
	__typeof__ (a) _a = (a); \
  __typeof__ (b) _b = (b); \
  _a < _b ? _a : _b; \
})

// HMult in ]0, 1[
#define CPU_ACCESS_H(r, HMult, size) ({ \
	RANDOM_ACCESS(r, MAX((1.0-(HMult))*(size), 0), size); \
})

// MMult in ]0, 1[ (also include HMult area) --> use, e.g., 2 times HMult
#define CPU_ACCESS_M(r, MMult, size) \
	CPU_ACCESS_H(r, MMult, size)

// HMult in ]0, 1[
#define GPU_ACCESS_H(r, HMult, size) ({ \
	RANDOM_ACCESS(r, 0, MIN((HMult)*(size), size)); \
})

// MMult in ]0, 1[ (also include HMult area) --> use, e.g., 2 times HMult
#define GPU_ACCESS_M(r, MMult, size) \
	GPU_ACCESS_H(r, MMult, size)

#endif /* BANK_H */
