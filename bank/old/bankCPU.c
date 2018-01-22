#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* _GNU_SOURCE */

#include "shared.h"
#include "thread.h"
#include "timer.h"
#include "cuda_wrapper.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#if defined(TM_GCC)
# include "../../abi/gcc/tm_macros.h"
#elif defined(TM_DTMC)
# include "../../abi/dtmc/tm_macros.h"
#elif defined(TM_INTEL)
# include "../../abi/intel/tm_macros.h"
#elif defined(TM_ABI)
# include "../../abi/tm_macros.h"
#endif /* defined(TM_ABI) */

// TODOs: 1) add timeout, after which transactions stop
//        2) for some reason sigle threaded is fast --> init in thread?


//#define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT printf
#else
#define DEBUG_PRINT(...)
#endif

#if defined(TM_GCC) || defined(TM_DTMC) || defined(TM_INTEL) || defined(TM_ABI)
# define TM_COMPILER
/* Add some attributes to library function */
TM_PURE
void exit(int status);
TM_PURE
void perror(const char *s);
#else

#include "stm.h"
#include "mod_ab.h"
#include "log.h"
#include "cuda_defines.h"

#ifndef USE_TSX_IMPL

static unsigned int tiny_aborts, tiny_commits;

// goes for TinySTM
#define TM_START(tid, ro)        { stm_tx_attr_t _a = {{.id = tid, .read_only = ro}}; sigjmp_buf *_e = stm_start(_a); if (_e != NULL) sigsetjmp(*_e, 0)
#define TM_LOAD(addr)            stm_load((stm_word_t *)addr)
#define TM_STORE(addr, value)    stm_store((stm_word_t *)addr, (stm_word_t)value)
#define TM_COMMIT                stm_commit(); }
#define TM_LOG(val)              //stm_log_add(0,val)
#define TM_LOG2(pos,val)         //stm_log_add(pos,val)
#define TM_FREE(point)           //stm_logel_free(point)
#define TM_GET_LOG(p)            stm_get_stats("HeTM_CPULog", &p);
#define TM_GET_NB_COMMITS(var)   var = tiny_aborts
#define TM_GET_NB_FALLBACK(var)  var = 0
#define TM_GET_NB_ABORTS(var)    var = tiny_commits
#define TM_GET_NB_CONFL(var)     var = 0
#define TM_GET_NB_CAPAC(var)     var = 0

#define TM_INIT(nb_threads)	     stm_init(); mod_ab_init(0, NULL)
#define TM_EXIT                  stm_exit()
#define TM_INIT_THREAD(p,s)      stm_init_thread(); //stm_log_init_bm(p,s)
#define TM_EXIT_THREAD        ({ \
	unsigned int aborts, commits; \
	stm_get_stats("nb_commits", &aborts); \
	stm_get_stats("nb_aborts", &commits); \
	__sync_add_and_fetch(&tiny_aborts, aborts); \
	__sync_add_and_fetch(&tiny_commits, commits); \
	 stm_exit_thread(); \
})
#else /* USE_TSX_IMPL */
// redefine with TSX
#include "tsx_impl.h"
#define TM_START(tid, ro) 		  HTM_SGL_begin();
#define TM_COMMIT 		          HTM_SGL_commit();
#define TM_LOAD(addr)           HTM_SGL_read(addr)
#define TM_STORE(addr, value)   HTM_SGL_write(addr, value)

#define TM_GET_LOG(p)           HeTM_get_log(&p)
#define TM_LOG(val)             // stm_log_add(0,val)
#define TM_LOG2(pos,val)        // stm_log_add(pos,val)
#define TM_FREE(point)          // stm_logel_free(point)
#define TM_GET_NB_COMMITS(var)  var = TM_get_error(SUCCESS)
#define TM_GET_NB_FALLBACK(var) var = TM_get_error(FALLBACK)
#define TM_GET_NB_ABORTS(var)   var = TM_get_error(ABORT)
#define TM_GET_NB_CONFL(var)    var = TM_get_error(CONFLICT)
#define TM_GET_NB_CAPAC(var)    var = TM_get_error(CAPACITY)

#define TM_INIT(nb_threads)		  HTM_init(nb_threads); stm_init(); mod_ab_init(0, NULL)
#define TM_EXIT                 HTM_exit(); stm_exit()
#define TM_INIT_THREAD(p,s)     HTM_SGL_init_thr(); stm_init(); stm_log_initBM(HeTM_log, p, s)
#define TM_EXIT_THREAD          printf("\n   EXIT TSX\n\n"); \
	HTM_SGL_exit_thr(); // stm_exit_thread()

#ifdef HTM_NO_WR_INST
#undef HTM_SGL_after_write
#define HTM_SGL_after_write(addr, val) /* empty, no log instrumentation */
#endif

#endif
#endif

#ifndef FILE_NAME
#define FILE_NAME "bank_stats"
#endif

static char *file_name;

// ########################## defines

#define XSTR(s)                         STR(s)
#define STR(s)                          #s

#ifndef ALIGNED
#define ALIGNED __attribute__((aligned(64)))
#endif
#define GRANULE_T intptr_t

#define NOP_X10 asm volatile("nop\n\t\nnop\n\t\nnop\n\t\nnop\n\t\nnop" \
"\n\t\nnop\n\t\nnop\n\t\nnop\n\t\nnop\n\t\nnop" ::: "memory")

#define SPIN_TIME(nb) ({ \
	volatile int i; \
	for (i = 0; i < nb * 10; ++i) NOP_X10; \
	0; \
})

# define no_argument        0
# define required_argument  1
# define optional_argument  2

// TODO: cache-align vs non-cache-align
// TODO: bank accounts with different granularities
//  * CACHE_LINE_SIZE / sizeof(GRANULE_T)
// #undef CACHE_ALIGN_POOL // do not use it here
// #ifdef CACHE_ALIGN_POOL
// #define GET_ACCOUNT(pool, i) (pool[i * CACHE_LINE_SIZE / sizeof(GRANULE_T)])
// #else
// #define GET_ACCOUNT(pool, i) (pool[i])
// #endif

#define GET_ACCOUNT(pool, i) &(pool[i])

#define THREAD_SPACING 1
#define THREAD_OFFSET 0

#define READ_ONLY_NB_ACCOUNTS 200

/*
#define RAND_R_FNC_aux(seed) \
({ \
unsigned seed_addr = seed; \
unsigned res = rand_r(&seed_addr); \
seed = seed_addr; \
res; \
})
// */
//*
// #define RAND_R_FNC_aux(seed) rand_r(&seed)
#define RAND_R_FNC_aux(seed) ({ \
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
// */
// ########################## constants

// static const int  ---> not a constant in C
// static const int  ---> it is a constant in C++

#define DEFAULT_NB_ACCOUNTS     128
#define DEFAULT_NB_THREADS      1
#define DEFAULT_NB_NTX_THREADS  0
#define DEFAULT_BANK_BUDGET     5000
#define DEFAULT_TX_SIZE         1
#define DEFAULT_TRANSFER_LIM    3
#define DEFAULT_NB_TRANSFERS    5000
#define DEFAULT_SEQUENTIAL      0
#define DEFAULT_UPDATE_RATE     100
#define DEFAULT_NO_CONFL        0
#define DEFAULT_TX_TIME         0
#define DEFAULT_OUTSIDE_TIME    0
#define DEFAULT_READ_SIZE       200

// ########################## variables

static int nb_accounts    = DEFAULT_NB_ACCOUNTS,
nb_threads     = DEFAULT_NB_THREADS,
nb_ntx_threads = DEFAULT_NB_NTX_THREADS,
bank_budget    = DEFAULT_BANK_BUDGET,
tx_size        = DEFAULT_TX_SIZE,
transfer_lim   = DEFAULT_TRANSFER_LIM,
seq            = DEFAULT_SEQUENTIAL,
update_rate    = DEFAULT_UPDATE_RATE,
no_conflicts   = DEFAULT_NO_CONFL,
outside_time   = DEFAULT_OUTSIDE_TIME,
read_size      = DEFAULT_READ_SIZE,
tx_time        = DEFAULT_TX_TIME;

int nb_transfers = DEFAULT_NB_TRANSFERS;

static int isEnd = 0; // change to 1 to end the simulation

static ALIGNED GRANULE_T **pool;
static __thread ALIGNED GRANULE_T *loc_pool;
static int nt_per_thread;
static TIMER_T c_ts1, c_ts2;
static pthread_barrier_t barrier;
static ALIGNED __thread int nb_write_txs_pt = 0;

static int nb_exit_threads = 0;
static unsigned long long count_time_tx;
static unsigned long long count_time_outside;
static double time_taken;
long nb_of_done_transactions = 0;

// ########################## functions
static int bank_total(GRANULE_T *pool);
static void initialize_pool(GRANULE_T *pool);
static int bank_total_tx(GRANULE_T *pool);
static void random_transfer(void *arg);
static void* timeout_fn(void *arg);
static void stats_to_gnuplot_file(char *filename);
static void* timeout_fn(void *arg);
// ########################## main

int main(int argc, char **argv)
{
	int c, i;

	struct option long_options[] = {
		// These options don't set a flag
		{"help",                      no_argument,       NULL, 'h'},
		{"sequential",                no_argument,       NULL, 'q'},
		{"no-confl",                  no_argument,       NULL, 'c'},
		{"duration",                  required_argument, NULL, 'd'},
		{"num-accounts",              required_argument, NULL, 'a'},
		{"num-threads",               required_argument, NULL, 'n'},
		{"file",                      required_argument, NULL, 'f'},
		{"update-rate",               required_argument, NULL, 'u'},
		{"size-tx",                   required_argument, NULL, 's'},
		{"budget",                    required_argument, NULL, 'b'},
		{"wait",                      required_argument, NULL, 'w'},
		{"transfer-lim",              required_argument, NULL, 't'},
		{"outside",                   required_argument, NULL, 'o'},
		{"read-size",                 required_argument, NULL, 'r'},
		{NULL, 0, NULL, 0}
	};

	while(1) {
		i = 0;
		c = getopt_long(argc, argv, "hqcd:a:n:u:f:s:b:t:w:o:m:r:",
		long_options, &i);

		if(c == -1)
		break;

		if(c == 0 && long_options[i].flag == 0)
		c = long_options[i].val;

		switch(c) {
			case 0:
			/* Flag is automatically set */
			break;
			case 'h':
			printf("bank -- STM stress test "
			"\n"
			"Usage:\n"
			"  bank [options...]\n"
			"\n"
			"Options:\n"
			"  -h, --help\n"
			"        Print this message\n"
			"  -q, --sequential\n"
			"        Transfer from contiguos accounts\n"
			"  -f, --file\n"
			"        File to output data\n"
			"  -c, --no-confl\n"
			"        Use different accounts per thread\n"
			"  -d, --duration <int>\n"
			"        Number of transfers (default=" XSTR(DEFAULT_NB_TRANSFERS) ")\n"
			"  -a, --nb-accounts <int>\n"
			"        Number of accounts (default=" XSTR(DEFAULT_NB_ACCOUNTS) ")\n"
			"  -n, --num-threads <int>\n"
			"        Number of threads (default=" XSTR(DEFAULT_NB_THREADS) ")\n"
			"  -m, --num-ntx-threads <int>\n"
			"        Number of threads not running transactions (default=" XSTR(DEFAULT_NB_THREADS) ")\n"
			"  -u, --update-rate <int>\n"
			"        Percentage of updates (default=" XSTR(DEFAULT_UPDATE_RATE) ")\n"
			"  -s, --size-tx <int>\n"
			"        Number transfers in transaction (default=" XSTR(DEFAULT_TX_SIZE) ")\n"
			"  -b, --budget <int>\n"
			"        Bank budget (default=" XSTR(DEFAULT_BANK_BUDGET) ")\n"
			"  -t, --transfer-lim <int>\n"
			"        Transfer limit (default=" XSTR(DEFAULT_TRANSFER_LIM) ")\n"
			"  -w, --wait <int>\n"
			"        Transaction time (default=" XSTR(DEFAULT_TX_TIME) ")\n"
			"  -o, --outside <int>\n"
			"        Outside transaction time (default=" XSTR(DEFAULT_OUTSIDE_TIME) ")\n"
			"  -r, --read-size <int>\n"
			"        Outside transaction time (default=" XSTR(DEFAULT_OUTSIDE_TIME) ")\n"
		);
		exit(EXIT_SUCCESS);
		case 'q':
		seq = 1;
		break;
		case 'c':
		no_conflicts = 1;
		break;
		case 'd':
		nb_transfers = atoi(optarg);
		break;
		case 'b':
		bank_budget = atoi(optarg);
		break;
		case 'a':
		nb_accounts = atoi(optarg);
		break;
		case 'n':
		nb_threads = atoi(optarg);
		break;
		case 'f':
		file_name = optarg;
		break;
		case 'm':
		nb_ntx_threads = atoi(optarg);
		break;
		case 's':
		tx_size = atoi(optarg);
		break;
		case 't':
		transfer_lim = atoi(optarg);
		break;
		case 'u':
		update_rate = atoi(optarg);
		break;
		case 'w':
		tx_time = atoi(optarg);
		break;
		case 'o':
		outside_time = atoi(optarg);
		break;
		case 'r':
		read_size = atoi(optarg);
		break;
		case '?':
		printf("Use -h or --help for help\n");
		exit(EXIT_SUCCESS);
		default:
		exit(EXIT_FAILURE);
	}
}

#ifdef CACHE_ALIGN_POOL
printf(" ########## Bank is cache align! ########## \n");
#endif

nt_per_thread = nb_transfers / nb_threads;

pthread_barrier_init(&barrier, NULL, nb_threads);

// thread example
printf(" Start program ========== \n");
printf("   NO_CONFLICTS: %i\n", no_conflicts);
printf("     SEQUENTIAL: %i\n", seq);
printf("     NB_THREADS: %i\n", nb_threads);
printf("   NB_TRANSFERS: %i\n", nb_transfers);
printf("    NB_ACCOUNTS: %i\n", nb_accounts);
printf("    BANK_BUDGET: %i\n", bank_budget);
printf("        TX_SIZE: %i\n", tx_size);
printf("        TX_TIME: %i\n", tx_time);
printf("   OUTSIDE_TIME: %i\n", outside_time);
printf("    UPDATE_RATE: %i\n", update_rate);
printf(" TRANSFER_LIMIT: %i\n", transfer_lim);
printf(" -------------------- \n");
printf(" TXs PER THREAD: %i\n", nt_per_thread);
printf(" ======================== \n");

// TODO: these TM_MALLOCs should be probably normal mallocs
if (no_conflicts) {
	pool = (GRANULE_T**) malloc(nb_threads * sizeof (GRANULE_T*));

	for (i = 0; i < nb_threads; ++i) {
		pool[i] = (GRANULE_T*) malloc(nb_accounts * sizeof(GRANULE_T));
		initialize_pool(pool[i]);
	}
}
else {
	pool = (GRANULE_T**) malloc(sizeof (GRANULE_T*));
	*pool = (GRANULE_T*) malloc(nb_accounts * sizeof(GRANULE_T));
}

srand(clock()); // TODO: seed

// TODO: in NVM context test if there is money in the bank
int total = bank_total(pool[0]);
if (total != bank_budget) {
	printf("Wrong bank amount: %i\n", total);
	initialize_pool(pool[0]);
}
total = bank_total(pool[0]);
printf("Bank amount: %i\n", total);

TM_INIT(nb_threads);
thread_startup(nb_threads);

pthread_t timerThr;
pthread_create(&timerThr, NULL, timeout_fn, NULL);

TIMER_READ(c_ts1);
thread_start(random_transfer, NULL); // TODO: threading
TIMER_READ(c_ts2);

time_taken = TIMER_DIFF_SECONDS(c_ts1, c_ts2);
printf("\nTime = %0.6lf\n", time_taken);

stats_to_gnuplot_file(file_name);

TM_EXIT;
thread_shutdown();

return EXIT_SUCCESS;
}

// ########################## function implementation

static void random_transfer(void *arg)
{
	int i;
	int tx;
	int nb_accounts_loc  = nb_accounts,
	update_rate_loc  = update_rate,
	tx_size_loc      = tx_size,
	transfer_lim_loc = transfer_lim,
	tx_time_loc      = tx_time;
	int tid;
	int seq_loc = seq;

	intptr_t total;

	TM_INIT_THREAD(*pool, nb_accounts*sizeof(GRANULE_T)); // TODO: remove this from the timer

	// TM_free(TM_alloc(64)); // TEST

	tid = thread_getId();

	unsigned seed = 123456 ^ (tid << 3);

	if (tid >= nb_threads) {
		return; // error!
	}

	if (no_conflicts) { // TODO: kill this one
		loc_pool = pool[tid];
	}
	else {
		loc_pool = *pool;
	}

	while (!isEnd) {
	// for (tx = 0; tx < nt_per_thread; ++tx) {}
	// for (tx = 0; (tx < nt_per_thread || nb_exit_threads < nb_threads); ++tx) {}
	// while (__sync_add_and_fetch(&nb_of_done_transactions, 1) < nb_transfers) {}

		int sender_amount;
		int recipient_amount;
		int sender_new_amount;
		int recipient_new_amount;

		int transfer_amount;
		int recipient;
		int sender;

		if ((RAND_R_FNC_aux(seed) % 100) < update_rate_loc) {
			TM_START(tid, 0);

			if (seq_loc) {
				sender = (RAND_R_FNC_aux(seed) % nb_accounts_loc);
			}

			// -------------------------------------------
			// do some computation that is not erased by the compiler
			for (i = 0; i < tx_size_loc ; ++i) {

				if (!seq_loc) {
					recipient = (RAND_R_FNC_aux(seed) % nb_accounts_loc);
					sender = (RAND_R_FNC_aux(seed) % nb_accounts_loc);
				} else {
					recipient = (sender + 10) % nb_accounts_loc;
					sender = (recipient + 10) % nb_accounts_loc;
				}

				if (sender == recipient) {
					--i;
					continue;
				}

				// Read
				sender_amount = TM_LOAD(GET_ACCOUNT(loc_pool, sender));
				recipient_amount = TM_LOAD(GET_ACCOUNT(loc_pool, recipient));

				// Process
				transfer_amount = (RAND_R_FNC_aux(seed) % transfer_lim_loc) + 1; //

				sender_new_amount = sender_amount - transfer_amount;
				recipient_new_amount = recipient_amount + transfer_amount;

				if (sender_new_amount < 0) {
					// the sender does not have the money! write back the same
					sender_new_amount = sender_amount;
					recipient_new_amount = recipient_amount;
				}

				// Write
				//if ((RAND_R_FNC_aux(seed) % 100) < update_rate_loc) {
				TM_STORE(GET_ACCOUNT(loc_pool, sender), sender_new_amount);
				TM_STORE(GET_ACCOUNT(loc_pool, recipient), recipient_new_amount);
				//}
			}
			// -------------------------------------------

			TM_COMMIT;
		} else {
			bank_total_tx(loc_pool);
		}

		if (tx == nt_per_thread) {
			__sync_add_and_fetch(&nb_exit_threads, 1); // TODO
			__sync_synchronize();
		}
	}

	// printf("thread %i exit!\n", tid);
	TM_EXIT_THREAD;
}

static void initialize_pool(GRANULE_T *pool)
{
	int i;

	int inc_step = 10;

	for (i = 0; i < nb_accounts; ++i) {
		*GET_ACCOUNT(pool, i) = 0; // Replace with memset
	}

	for (i = inc_step; i < bank_budget; i += inc_step) {
		int account = rand() % nb_accounts;
		int new_val = *GET_ACCOUNT(pool, account) + inc_step;

		*GET_ACCOUNT(pool, account) = new_val;
	}

	i -= inc_step;

	for (; i < bank_budget; ++i) {
		int account = rand() % nb_accounts;
		int new_val = *GET_ACCOUNT(pool, account) + 1;

		*GET_ACCOUNT(pool, account) = new_val;
	}

	// check the total in the checkpoint
	// printf("Total in checkpoint: %i \n", bank_total(pool));
}

static int bank_total(GRANULE_T *pool)
{
	int i, res = 0;

	for (i = 0; i < nb_accounts; ++i) {
		res += *GET_ACCOUNT(pool, i);
	}

	return res;
}

static int bank_total_tx(GRANULE_T *pool)
{
	int i, read_size_loc = read_size;
	int nb_accounts_loc = nb_accounts;
	ALIGNED GRANULE_T *pool_loc = pool;
	int res = 0, rand;
	static __thread unsigned aux_seed = 1234;

	TM_START(0, 0);
	for (i = 0; i < read_size_loc; ++i) {
		int j;
		rand = (RAND_R_FNC_aux(aux_seed) % nb_accounts);
		// if seq -->
		j = (rand + i * 10) % nb_accounts_loc;
		res += TM_LOAD(GET_ACCOUNT(pool_loc, j));
	}
	TM_COMMIT;

	return res;
}

static void* timeout_fn(void *arg)
{
	struct timespec timeout;
	double duration = nb_transfers;
	timeout.tv_sec = duration / 1000;
  timeout.tv_nsec = ((long)duration % 1000) * 1000000;

	nanosleep(&timeout, NULL);

	isEnd = 1;
	__sync_synchronize();
	return NULL;
}

static void stats_to_gnuplot_file(char *filename) {
	FILE *gp_fp = fopen(filename, "a");

	if (ftell(gp_fp) < 8) {
		fprintf(gp_fp, "#"
			"NB_ACCOUNTS\t"       // [01] NB_ACCOUNTS
			"THREADS\t"           // [02] THREADS
			"TIME\t"              // [03] TIME
			"COMMITS\t"           // [04] COMMITS
			"FALLBACK\t"          // [05] FALLBACK
			"ABORTS\t"            // [06] ABORTS
			"PROB_ABORT\t"        // [07] PROB_ABORT
			"THROUGHPUT\t"        // [08] THROUGHPUT
			"TX_SIZE\t"           // [09] TX_SIZE
			"CONFLICT\t"          // [10] CONFLICT
			"CAPACITY\n"          // [11] CAPACITY
		);
	}

	unsigned int commits, fallback, aborts, conflict, capacity;
	double pa, X;

	TM_GET_NB_FALLBACK(fallback);
	TM_GET_NB_ABORTS(aborts);
	TM_GET_NB_CONFL(conflict);
	TM_GET_NB_CAPAC(capacity);
	TM_GET_NB_COMMITS(commits);

	pa = (double)aborts / (double)(aborts + commits + fallback);
	X = (double)(commits + fallback) / (double)time_taken;

	fprintf(gp_fp, "%i\t", nb_accounts);      // [01] NB_ACCOUNTS
	fprintf(gp_fp, "%i\t", nb_threads);       // [02] THREADS
	fprintf(gp_fp, "%f\t", time_taken);       // [03] TIME
	fprintf(gp_fp, "%u\t", commits);          // [04] COMMITS
	fprintf(gp_fp, "%u\t", fallback);         // [05] FALLBACK
	fprintf(gp_fp, "%u\t", aborts);           // [06] ABORTS
	fprintf(gp_fp, "%f\t", pa);               // [07] PROB_ABORT
	fprintf(gp_fp, "%f\t", X);                // [08] THROUGHPUT
	fprintf(gp_fp, "%i\t", tx_size);          // [09] TX_SIZE
	fprintf(gp_fp, "%i\t", conflict);         // [10] CONFLICT
	fprintf(gp_fp, "%i\n", capacity);         // [11] CAPACITY
	fclose(gp_fp);

	printf("printed stats\n");
}
