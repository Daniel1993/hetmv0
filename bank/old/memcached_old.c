#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_wrapper.h"
#include <string.h>
#include <unistd.h>

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
#else /* Compile with explicit calls to tinySTM */

#include "stm.h"
#include "mod_ab.h"
#include "log.h"
#include "cuda_defines.h"
#include "utils.h"
#include "shared.h"
#include "thread.h"
#include "bank.h"
#include "bank_aux.h"

/* ################################################################### *
 * GLOBALS
 * ################################################################### */

static volatile int stop;
static volatile int cuda_stop;
static volatile int cpu_enable_flag;
static volatile int global_ts;
static volatile HeTM_CPULogNode_t *HeTM_CPULog[32];						//FIXME
static int cmp_flag;

/* ################################################################### *
 * DATA STUCTURE
 * ################################################################### */

typedef long account_t;

typedef struct bank {
  account_t * accounts;
  int * version;
  long size;
  int ways;
  int * version_bm;
  int version_bm_size;
} bank_t;

/* ################################################################### *
 * THREAD STUCTURE
 * ################################################################### */

typedef struct thread_data {
  bank_t *bank;
  barrier_t *barrier;
  barrier_t *cuda_barrier;
  cuda_t * cd;
  unsigned long nb_transfer;
  unsigned long nb_read_all;
  unsigned long nb_write_all;
#ifndef TM_COMPILER
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
#endif /* ! TM_COMPILER */
  unsigned int seed;
  int id;
  int read_all;
  int read_threads;
  int write_all;
  int write_threads;
  int disjoint;
  int nb_threads, nb_threadsCPU;
  int set_percent;
  char padding[64];
} thread_data_t;

typedef enum {READ, STREAM, FINAL, END} cmp_status;

/* ################################################################### *
 * SEARCH/INSERTION FUNCITONS
 * ################################################################### */

typedef struct packet {
	int vers;
	long key;
} packet_t;

packet_t search_entry(bank_t *b, unsigned int *positions)
{
  int z = 0;
  int n, base, goal;
  int val, vers;
  packet_t response;

  /* Allow overdrafts */
  TM_START(z, RW);

  base = positions[0]*b->ways;
  goal = (positions[0] & 0xffff ) % b->ways;

  for (n = 0; n < goal; n++) {
	  val = TM_LOAD(&b->accounts[base+n]);
	  vers = TM_LOAD(&b->version[base+n]);

	  response.key = val;
	  response.vers = vers;
  }
  n -= 1;
  TM_STORE(&b->version[base+n],global_ts);

  TM_COMMIT;

  int pos = (base + n) >> BM_HASH;
  /*if(pos>=b->version_bm)
	printf("BUG: pos=%d ->base=%d n=%d\n",pos, positions[0], b->version_bm);*/
  b->version_bm[pos] = 1;

  return response;
}

packet_t new_entry(bank_t * b, unsigned int * positions)
{
  int z = 0;
  int n, base, goal;
  int val, vers, min_pos, min;
  packet_t response;

  /* Allow overdrafts */
  TM_START(z, RW);
  min = -1; min_pos=0;

  base = positions[0] * b->ways;
  goal = positions[0] & 0xffff;

  for (n = 0; n < b->ways; n++) {

	  val = TM_LOAD(&b->accounts[base+n]);
	  vers = TM_LOAD(&b->version[base+n]);

	  //Check if it is free or if it is the same value
	  if(val == goal || vers == 0) {
		min_pos = n;
		break;
	  } else {
		//Record timestamp value if is the oldest
		  if(( min == -1) || (vers < min) ) {
			min = vers;
			min_pos = n;
		  }
	  }
  }

  TM_STORE(&b->accounts[base+min_pos],goal);
  TM_STORE(&b->version[base+min_pos],global_ts);

  response.key = goal;
  response.vers = base+min_pos;

  TM_COMMIT;

  return response;
}

/* ################################################################### *
 * TRANSACTION THREADS
 * ################################################################### */

static void *test(void *data)
{
  int nb;
  unsigned int accounts_vec[bankNum], i;
  thread_data_t *d = (thread_data_t *)data;
  int nb_accounts = d->bank->size;
  unsigned short seed[3];
  unsigned int * queue_pointer = NULL;
  int queue_size = 0, queue_current = 0;
  void * p = NULL;
  int total_el = 0;
  HeTM_CPULogNode_t * logAux = NULL;
  cmp_status state = FINAL;

  bindThread(d->id);

  int n = 0, count = 0;
  int * flag_check = (int*) CudaMallocWrapper(sizeof(int));
  stream_t *s = jobWithCuda_initStream(d->cd, d->id, STREAM_COUNT, flag_check);
  HeTM_CPULogEntry * logVec[STREAM_COUNT];
  for(count=0; count<STREAM_COUNT; count++) {
    logVec[count] = (HeTM_CPULogEntry*) CudaMallocWrapper(LOG_SIZE*sizeof(HeTM_CPULogEntry));
  }
  state = READ;

  /* Initialize seed (use rand48 as rand is poor) */
  seed[0] = (unsigned short)rand_r(&d->seed);
  seed[1] = (unsigned short)rand_r(&d->seed);
  seed[2] = (unsigned short)rand_r(&d->seed);

  /*Populate queue for first run*/
  queue_size = 4*bankNum + d->id*bankNum;
  queue_pointer = readQueue(d->cd->q, queue_size, 0);
  queue_pointer[0] = (unsigned int) (erand48(seed) * 500);

  /* Create transaction */
  TM_INIT_THREAD(d->bank->accounts, d->bank->size*d->bank->ways);
  /* Wait on barrier */
  barrier_cross(d->barrier);

  while (stop == 0 || cuda_stop!=2) {

    TEST_GPU_IN_LOOP;

  	if(cpu_enable_flag == 1) {

  		/* Fetch values from the queue */
  		for(i=0; i<1; i++) {
  			if (queue_current >= queue_size) {
  				//queue emptied, fetch new values
  				queue_pointer = readQueue(d->cd->q, queue_size, 0);
  				queue_current = 0;
  			}
  			accounts_vec[i] = queue_pointer[queue_current];
  			queue_current++;
  		}

  		nb = (int)(erand48(seed) * 1000);
  		if (nb >= d->set_percent) {
  			/* Read (GET) */
  			search_entry(d->bank, accounts_vec);
  			d->nb_read_all++;
  		} else {
  			/* Write (SET) */
  			new_entry(d->bank, accounts_vec);
  			d->nb_write_all++;
  		}
  	}
  }

  DEBUG_PRINT("Thread %d exiting!\n", d->id);
#ifndef TM_COMPILER
  stm_get_stats("nb_aborts", &d->nb_aborts);
  stm_get_stats("nb_aborts_1", &d->nb_aborts_1);
  stm_get_stats("nb_aborts_2", &d->nb_aborts_2);
  stm_get_stats("nb_aborts_locked_read", &d->nb_aborts_locked_read);
  stm_get_stats("nb_aborts_locked_write", &d->nb_aborts_locked_write);
  stm_get_stats("nb_aborts_validate_read", &d->nb_aborts_validate_read);
  stm_get_stats("nb_aborts_validate_write", &d->nb_aborts_validate_write);
  stm_get_stats("nb_aborts_validate_commit", &d->nb_aborts_validate_commit);
  stm_get_stats("nb_aborts_invalid_memory", &d->nb_aborts_invalid_memory);
  stm_get_stats("nb_aborts_killed", &d->nb_aborts_killed);
  stm_get_stats("locked_reads_ok", &d->locked_reads_ok);
  stm_get_stats("locked_reads_failed", &d->locked_reads_failed);
  stm_get_stats("max_retries", &d->max_retries);
#endif /* ! TM_COMPILER */

  /* Free transaction */
  queue_pointer = NULL;
  BANK_TEARDOWN_TX();

  TM_EXIT_THREAD;

  return NULL;
}

/* ################################################################### *
 * CUDA THREAD
 * ################################################################### */

static void *test_cuda(void *data) {
  thread_data_t *d = (thread_data_t *)data;
  int i, n, loop=0;
  int flag, flag_check = 0;
  int counter_aborts = 0, counter_commits = 0, counter_sync = 0;		//Counters
  int duration= 0, tx_duration = 0;										//Timers
  int trf_2cpu= 0, trf_2gpu = 0;										//Timers
  float	duration_cmp = 0, trans_cmp = 0;
  struct timeval tx_start, tx_end, lock_start, lock_end, cmp_start, cmp_end;				//Timer structure
  HeTM_CPULogNode_t *aux;
  cuda_t *cd = d->cd;
  long * a = d->bank->accounts;
  int * v = d->bank->version;
  long * helper = NULL;
  int sync_fail = 0, sync_sucess = 0;

  int * valid_vec = NULL;
#if BM_TRANSF==1
  valid_vec = (int *)malloc( d->bank->size * sizeof(int));
#endif

  gettimeofday(&tx_start, NULL);

  DEBUG_PRINT("Starting GPU execution.\n");
  flag = jobWithCuda_run(cd,a,global_ts,ASINC_CONF);					//Start execution
  barrier_cross(d->barrier);											//Signal CPU threads to start

  if (flag == 0) {
	//Cuda error detection
	printf("CUDA run failed.\n");
	gettimeofday(&tx_end, NULL);
	cuda_stop = 2;
  } else {
#if GPUEn==0
  flag=0; cuda_stop = 2;
#else
	DEBUG_PRINT("Starting transaction kernel!\n");
#endif
  }

  //Cicle till execution concludes
  while (flag) {									//MAIN LOOP
    loop++; helper = NULL;
	jobWithCuda_wait();								//Wait for GPU execution to conclude


	gettimeofday(&tx_end, NULL);
	DEBUG_PRINT("Transaction kernel concluded\n");
	tx_duration += (tx_end.tv_sec * 1000 + tx_end.tv_usec / 1000) - (tx_start.tv_sec * 1000 + tx_start.tv_usec / 1000);

#ifndef NO_LOCK
	//Lock threads
	cuda_stop = 1;
	barrier_cross(d->cuda_barrier);
#endif

	DEBUG_PRINT("All threads locked\n");								/* LOCKED */
	gettimeofday(&lock_start, NULL);
	flag_check = 0;

	//Copy host data to device
	if ( !jobWithCuda_dupd(*cd, a) )
		flag_check=-1;
	gettimeofday(&cmp_start, NULL);
	trf_2gpu += (cmp_start.tv_sec * 1000 + cmp_start.tv_usec / 1000) - (lock_start.tv_sec * 1000 + lock_start.tv_usec / 1000);

#ifdef USE_STREAM
    struct timeval timerStream;				//Timer
	DEBUG_PRINT("Starting comparisson kernel!\n");

#ifdef NO_LOCK
	cuda_stop = 1;
#else
	barrier_cross(d->cuda_barrier);
#endif
#if BM_TRANSF==1

#endif
	barrier_cross(d->cuda_barrier);

	flag_check = cmp_flag;

	gettimeofday(&timerStream, NULL);
	duration_cmp +=  (timerStream.tv_sec * 1000 + timerStream.tv_usec / 1000) - (cmp_start.tv_sec * 1000 + cmp_start.tv_usec / 1000);
#else

	HeTM_CPULogNode_t *vec_log[32];
	for(i=0; i< d->nb_threads-1; i++) 									//Fetch tx logs
		vec_log[i] = HeTM_CPULog[i];

	//Main Comparison Loop
	n = 0;
	DEBUG_PRINT("Starting comparisson kernel!\n");
	while(n != d->nb_threads-1 && flag_check == 0) { 	//Reached the end of every list;
		n = 0;
		for(i=0; i< d->nb_threads-1 && flag_check==0; i++) {
			if ( vec_log[i] == NULL ) {
				DEBUG_PRINT("Reached end of %d.\n",i);
				n++;
				continue;
			}

			DEBUG_PRINT("Lauching comparison of size: %d\n", vec_log[i]->curPos);

			flag_check = jobWithCuda_check(*cd, vec_log[i]->array, vec_log[i]->curPos, &duration_cmp, &trans_cmp);
			vec_log[i] = vec_log[i]->next;

			//DEBUG_PRINT("Comprisson sucessfull.\n");
		}
		//DEBUG_PRINT("Complete comparisson loop.\n");
	}
#endif

	//Check for errors
	if ( flag_check == -1)	{
		printf("Comparison crashed. CUDA thread exiting\n");
		flag = 0;
	}

	//Transfer back results to CPU
	DEBUG_PRINT("Final comparison result: %d\n", flag_check);
	gettimeofday(&cmp_start, NULL);
	if(flag_check == 0) {				//Successful execution
		sync_fail = 0; sync_sucess++; 			//Sequence Counter
		helper = 0;

		if ( jobWithCuda_hupd(cd, a, v, valid_vec ) == 1 ) {
			counter_sync++;			//Success Counter
		} else {
			printf("BUG\n");
			flag = 0;
		}
#ifdef SYNC_BALANCING
#if CPUEn == 1
		if(sync_sucess == SYNC_BALANCING_VALS && cpu_enable_flag == 0)
			cpu_enable_flag = 1;
#endif
#endif
#if RUN_ONCE == 1
		flag = 0;
#endif
	} else {							//Unccessful execution
		sync_fail++; sync_sucess = 0;		//Sequence Counter
		helper = jobWithCuda_swap(*cd);		//Update device data for re-rerun
#ifdef SYNC_BALANCING
#if CPUEn == 1
		if(sync_fail == SYNC_BALANCING_VALF)
			cpu_enable_flag = 0;
#endif
#endif
	}
	gettimeofday(&cmp_end, NULL);

	//Check if we are re-running
#if RUN_ONCE == 1
	if(flag && stop == 0) {
#else
	if(stop == 0 && flag_check!=-1) {
#endif
		cuda_stop = 0;
	} else {
		//Times up
		cuda_stop = 2;
		flag = 0;
	}

#ifdef USE_STREAM
	cmp_flag = 0;
#endif
#if CPUEn==1
	if(flag_check==1 || sync_sucess % (VERSION_UPD_RATE + 1) == VERSION_UPD_RATE) {
		jobWithCuda_updVersion(cd,v,d->bank->version_bm, d->bank->version_bm_size);
		memset(d->bank->version_bm, 0,  d->bank->version_bm_size*sizeof(int));
	}
#endif
	gettimeofday(&lock_end, NULL);
	global_ts++;			//Global_timestamp
	barrier_cross(d->cuda_barrier);

	DEBUG_PRINT("All threads unlocked\n");
																		//printf("loop2: %d\n",loop);

	//Rerun transaction kernel
	if(cuda_stop != 2) {

		if( flag_check == 0 )
			jobWithCuda_getStats(*cd,&counter_aborts,&counter_commits);						//Update stats

		if ( jobWithCuda_run(cd, helper, global_ts, flag_check ))
			DEBUG_PRINT("Running new transaction kernel!\n");

		gettimeofday(&tx_start, NULL);
	}
																		//printf("loop4: %d\n",loop);

	//Clean up the memory
	do {
		n = 0;
		for(i=0; i< d->nb_threads-1; i++) {
			if ( HeTM_CPULog[i] == NULL) {
				n++;
				continue;
			}
			aux = HeTM_CPULog[i];
			HeTM_CPULog[i] = HeTM_CPULog[i]->next;
			TM_FREE(aux);
			//printf("Free from thread %d\n", i);
		}
	} while(n != d->nb_threads-1); 	//Reached the end of every list;

	duration += (lock_end.tv_sec * 1000 + lock_end.tv_usec / 1000) - (lock_start.tv_sec * 1000 + lock_start.tv_usec / 1000);
	trf_2cpu += (cmp_end.tv_sec * 1000 + cmp_end.tv_usec / 1000) - (cmp_start.tv_sec * 1000 + cmp_start.tv_usec / 1000);
  }


  printf("CUDA thread terminated after %d(%d successful) run(s). \nTotal cuda execution time: %d ms.\n", loop, counter_sync, duration);
  if(flag_check==0)
	jobWithCuda_getStats(*cd,&counter_aborts,&counter_commits);

  d->nb_transfer = counter_commits;
  d->nb_aborts = counter_aborts;
  d->nb_aborts_1 = duration;
  d->nb_aborts_2 = duration_cmp;
  d->nb_aborts_locked_read = loop;
  d->nb_aborts_locked_write = counter_sync;
  d->nb_aborts_validate_read = trf_2cpu;
  d->nb_aborts_validate_write = trf_2gpu;
  d->nb_aborts_validate_commit = trans_cmp;
  d->nb_aborts_invalid_memory = 0;
  d->nb_aborts_killed = 0;
  d->locked_reads_ok = 0;
  d->locked_reads_failed = 0;
  d->max_retries = tx_duration;

  DEBUG_PRINT("CUDA thread exiting.\n");

  return NULL;
}


/* Signal Catcher */
static void catcher(int sig)
{
  static int nb = 0;
  printf("CAUGHT SIGNAL %d\n", sig);
  if (++nb >= 3)
    exit(1);
}

/* ################################################################### *
 *
 * MAIN
 *
 * ################################################################### */

int main(int argc, char **argv)
{
  struct option long_options[] = {
    // These options don't set a flag
    {"help",                      no_argument,       NULL, 'h'},
    {"accounts",                  required_argument, NULL, 'a'},
    {"contention-manager",        required_argument, NULL, 'c'},
    {"duration",                  required_argument, NULL, 'd'},
    {"num-threads",               required_argument, NULL, 'n'},
    {"read-all-rate",             required_argument, NULL, 'r'},
    {"read-threads",              required_argument, NULL, 'R'},
    {"seed",                      required_argument, NULL, 's'},
    {"write-all-rate",            required_argument, NULL, 'w'},
    {"write-threads",             required_argument, NULL, 'W'},
	{"num-iterations",            required_argument, NULL, 'i'},
	{"shared-rate",            	  required_argument, NULL, 'S'},
	{"new-rate",            	  required_argument, NULL, 'N'},
	{"number-ways",            	  required_argument, NULL, 'l'},
	{"gpu-threads",           	  required_argument, NULL, 'T'},
	{"gpu-blocks",           	  required_argument, NULL, 'B'},
	{"output-file",               required_argument, NULL, 'f'},
    {"disjoint",                  no_argument,       NULL, 'j'},
    {NULL, 0, NULL, 0}
  };

  bank_t *bank;
  barrier_t cuda_barrier;
  int i, c, j, ret = -1;
  unsigned long reads, writes, updates;
#ifndef TM_COMPILER
  char *s;
  unsigned long aborts, aborts_1, aborts_2,
    aborts_locked_read, aborts_locked_write,
    aborts_validate_read, aborts_validate_write, aborts_validate_commit,
    aborts_invalid_memory, aborts_killed,
    locked_reads_ok, locked_reads_failed, max_retries;
  //stm_ab_stats_t ab_stats;
  char *cm = NULL;
#endif /* ! TM_COMPILER */
  thread_data_t *data;
  pthread_t *threads;
  pthread_attr_t attr;
  barrier_t barrier;
  int num_ways = NUMBER_WAYS;
  int shared_percent = QUEUE_SHARED_VALUE;
  int set_percent = WRITE_PERCENT;
  int gt = DEFAULT_threadNum;
  int gb = DEFAULT_blockNum;
  struct timeval start, end, final;
  struct timespec timeout;
  int duration = DEFAULT_DURATION;
  int duration2 = DEFAULT_DURATION;
  int nb_accounts = DEFAULT_NB_ACCOUNTS;
  int nb_threads = DEFAULT_NB_THREADS;
  int nb_threadsCPU = DEFAULT_NB_THREADS;
  int read_all = DEFAULT_READ_ALL;
  int read_threads = DEFAULT_READ_THREADS;
  int seed = DEFAULT_SEED;
  int write_all = DEFAULT_WRITE_ALL;
  int write_threads = DEFAULT_WRITE_THREADS;
  int disjoint = DEFAULT_DISJOINT;
  int iter = DEFAULT_ITERATIONS;
  double throughput, tot_throughput[MAX_ITER];
  unsigned int tot_duration[MAX_ITER], tot_duration2[MAX_ITER];
  unsigned long tot_commits[MAX_ITER], tot_aborts[MAX_ITER];
  cuda_t * cuda_st;
  int * bitmap;
  double throughput_gpu, tot_throughput_gpu[MAX_ITER];
  unsigned int tot_comp[MAX_ITER], tot_tx[MAX_ITER], tot_cuda[MAX_ITER];
  unsigned int tot_loop[MAX_ITER], tot_loops[MAX_ITER], tot_trf2cpu[MAX_ITER], tot_trf2gpu[MAX_ITER], tot_trfcmp[MAX_ITER];;
  unsigned long tot_commits_gpu[MAX_ITER], tot_aborts_gpu[MAX_ITER];

  sigset_t block_set;
  char * fn = NULL;
  char filename[128];


  while(1) {
    i = 0;
    c = getopt_long(argc, argv, "ha:c:d:n:r:R:s:w:W:i:S:N:l:T:B:f:j", long_options, &i);

    if(c == -1)
      break;

    if(c == 0 && long_options[i].flag == 0)
      c = long_options[i].val;

    switch(c) {
     case 0:
       /* Flag is automatically set */
       break;
     case 'h':
       printf("bank -- STM stress test\n"
              "\n"
              "Usage:\n"
              "  bank [options...]\n"
              "\n"
              "Options:\n"
              "  -h, --help\n"
              "        Print this message\n"
              "  -a, --accounts <int>\n"
              "        Number of accounts in the bank (default=" XSTR(DEFAULT_NB_ACCOUNTS) ")\n"
#ifndef TM_COMPILER
              "  -c, --contention-manager <string>\n"
              "        Contention manager for resolving conflicts (default=suicide)\n"
#endif /* ! TM_COMPILER */
              "  -d, --duration <int>\n"
              "        Test duration in milliseconds (0=infinite, default=" XSTR(DEFAULT_DURATION) ")\n"
              "  -n, --num-threads <int>\n"
              "        Number of threads (default=" XSTR(DEFAULT_NB_THREADS) ")\n"
              "  -r, --read-all-rate <int>\n"
              "        Percentage of search transactions (default=" XSTR(DEFAULT_READ_ALL) ")\n"
              "  -R, --read-threads <int>\n"
              "        Number of threads issuing only search transactions (default=" XSTR(DEFAULT_READ_THREADS) ")\n"
              "  -s, --seed <int>\n"
              "        RNG seed (0=time-based, default=" XSTR(DEFAULT_SEED) ")\n"
              "  -w, --write-all-rate <int>\n"
              "        Percentage of write-all transactions (default=" XSTR(DEFAULT_WRITE_ALL) ")\n"
              "  -W, --write-threads <int>\n"
              "        Number of threads issuing only write-all transactions (default=" XSTR(DEFAULT_WRITE_THREADS) ")\n"
			  "  -i, --num-iterations <int>\n"
              "        Number of iterations to execute (default=" XSTR(DEFAULT_ITERATIONS) ")\n"
			  "  -S, --share-rate <int>\n"
              "        Percentage of transactions that are targeted at the shared zone (default=" XSTR(QUEUE_SHARED_VALUE) ")\n"
			  "  -N, --new-rate <int>\n"
              "        Percentage of set requests (default=" XSTR(DEFAULT_SET_RATE) ")\n"
			  "  -l, --num-ways <int>\n"
              "        Number of 'ways' for each hashtable entry entry (default=" XSTR(NUMBER_WAYS) ")\n"
			  "  -T, --gpu-threads <int>\n"
              "        Number of GPU threads (default=" XSTR(DEFAULT_threadNum) ")\n"
			  "  -B, --gpu-blocks <int>\n"
              "        Number of GPU blocks (default=" XSTR(DEFAULT_blockNum) ")\n"
			  "  -f, --file-name <string>\n"
              "        Output file name (default=" XSTR(DEFAULT_OUTPUT_FILE) ")\n"
         );
       exit(0);
     case 'a':
       nb_accounts = atoi(optarg);
       break;
#ifndef TM_COMPILER
     case 'c':
       cm = optarg;
       break;
#endif /* ! TM_COMPILER */
     case 'd':
       duration = atoi(optarg);
       break;
     case 'n':
       nb_threads = atoi(optarg);
       break;
     case 'r':
       read_all = atoi(optarg);
       break;
     case 'R':
       read_threads = atoi(optarg);
       break;
     case 's':
       seed = atoi(optarg);
       break;
     case 'w':
       write_all = atoi(optarg);
       break;
     case 'W':
       write_threads = atoi(optarg);
       break;
     case 'j':
       disjoint = 1;
       break;
	 case 'i':
       iter = atoi(optarg);
       break;
	 case 'S':
       shared_percent = atoi(optarg);
       break;
	 case 'N':
       set_percent = atoi(optarg);
       break;
	 case 'l':
       num_ways = atoi(optarg);
       break;
	 case 'T':
       gt = atoi(optarg);
	   gt = (gt >> 5) * 32;
       break;
	 case 'B':
       gb = atoi(optarg);
       break;
     case '?':
       printf("Use -h or --help for help\n");
       exit(0);
	 case 'f':
		fn = optarg;
		break;
     default:
       exit(1);
    }
  }

  jobWithCuda_exit(NULL);		//Reset Cuda Device

  if (fn!=NULL) {
	strncpy(filename,fn,100);
	filename[100]='\0';
  } else {
	strcpy(filename,XSTR(DEFAULT_OUTPUT_FILE));
	filename[4]='\0';
  }
  strcat(filename,".csv\0");

  assert(duration >= 0);
  assert(nb_accounts >= 2);
  assert(nb_threads > 0);
  assert(read_all >= 0 && write_all >= 0 && read_all + write_all <= 100);
  assert(read_threads + write_threads <= nb_threads);
  assert(iter <= MAX_ITER);
  assert(shared_percent <= 100);
  assert(set_percent <= 1000);
  assert(num_ways <= NUMBER_WAYS);

  printf("Nb accounts    : %d\n", nb_accounts);
#ifndef TM_COMPILER
  printf("CM             : %s\n", (cm == NULL ? "DEFAULT" : cm));
#endif /* ! TM_COMPILER */
  printf("Duration       : %d\n", duration);
  printf("Iterations     : %d\n", iter);
  printf("Nb threads     : %d\n", nb_threads);
  //printf("Read-all rate  : %d\n", read_all);
  //printf("Read threads   : %d\n", read_threads);
  //printf("Seed           : %d\n", seed);
  printf("Write rate     : %d\n", set_percent);
  printf("Shared rate    : %d\n", shared_percent);
  printf("Number of ways : %d\n", num_ways);
  printf("Type sizes     : int=%d/long=%d/ptr=%d/word=%d\n",
         (int)sizeof(int),
         (int)sizeof(long),
         (int)sizeof(void *),
         (int)sizeof(size_t));
  printf("Output file    : %s\n", filename);
  DEBUG_PRINT("Debug	       : Enabled\n");

#ifndef TM_COMPILER
  if (stm_get_parameter("compile_flags", &s))
	printf("STM flags      : %s\n", s);
#endif

  timeout.tv_sec = duration / 1000;
  timeout.tv_nsec = (duration % 1000) * 1000000;
  nb_threadsCPU = nb_threads;

  nb_threads++;
  barrier_init(&cuda_barrier, nb_threads );

  if ((data = (thread_data_t *)malloc(nb_threads * sizeof(thread_data_t))) == NULL) {
    perror("malloc");
    exit(1);
  }
  if ((threads = (pthread_t *)malloc(nb_threads * sizeof(pthread_t))) == NULL) {
    perror("malloc");
    exit(1);
  }

  if (seed == 0)
    srand((int)time(NULL));
  else
    srand(seed);

  bank = (bank_t *)malloc(sizeof(bank_t));
  bank->accounts = (account_t *)malloc(nb_accounts * num_ways * sizeof(account_t));
  bank->version = (int *)malloc(nb_accounts * num_ways * sizeof(int));
  bank->size = nb_accounts;
  bank->ways = num_ways;
  memset(bank->accounts, 0, nb_accounts* num_ways *sizeof(long));
  memset(bank->version, 0, nb_accounts* num_ways *sizeof(int));

  long entries = nb_accounts * num_ways;
  bank->version_bm_size = 1 + (entries >> BM_HASH);
  bank->version_bm = bitmap = (int*) malloc( bank->version_bm_size * sizeof(int));
  memset(bitmap, 0,  bank->version_bm_size*sizeof(int));

  for(j = 0; j<iter; j++) {				//Loop for testing purposes

	  //Init GPU
	  DEBUG_PRINT("Initializing GPU.\n");

	  cuda_st = jobWithCuda_init(bank->accounts, bank->size, bank->ways, 0, 0, gt, gb, set_percent, shared_percent);
	  //DEBUG_PRINT("Base: %lu %lu \n", bank->accounts, &bank->accounts);
	  if (cuda_st == NULL) {
		printf("CUDA init failed.\n");
		exit(-1);
	  }

	  //Clear flags
	  cuda_stop = stop = 0;
	  cpu_enable_flag = 1;
	  global_ts = 1;
#if CPUEn==0
	  cpu_enable_flag = 0;
#endif

	  /* Init STM */
	  printf("Initializing STM\n");
	  TM_INIT;

	#ifndef TM_COMPILER
	  if (cm != NULL) {
		if (stm_set_parameter("cm_policy", cm) == 0)
		  printf("WARNING: cannot set contention manager \"%s\"\n", cm);
	  }
	#endif /* ! TM_COMPILER */

	  /* Access set from all threads */
	  barrier_init(&barrier, nb_threads + 1);
	  pthread_attr_init(&attr);
	  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	  for (i = 0; i < nb_threads; i++) {
		printf("Creating thread %d\n", i);
		data[i].id = i;
		data[i].read_all = read_all;
		data[i].read_threads = read_threads;
		data[i].write_all = write_all;
		data[i].write_threads = write_threads;
		data[i].disjoint = disjoint;
		data[i].nb_threads = nb_threads;
		data[i].nb_transfer = 0;
		data[i].nb_read_all = 0;
		data[i].nb_write_all = 0;
	#ifndef TM_COMPILER
		data[i].nb_aborts = 0;
		data[i].nb_aborts_1 = 0;
		data[i].nb_aborts_2 = 0;
		data[i].nb_aborts_locked_read = 0;
		data[i].nb_aborts_locked_write = 0;
		data[i].nb_aborts_validate_read = 0;
		data[i].nb_aborts_validate_write = 0;
		data[i].nb_aborts_validate_commit = 0;
		data[i].nb_aborts_invalid_memory = 0;
		data[i].nb_aborts_killed = 0;
		data[i].locked_reads_ok = 0;
		data[i].locked_reads_failed = 0;
		data[i].max_retries = 0;
	#endif /* ! TM_COMPILER */
		data[i].seed = rand();
		data[i].bank = bank;
		data[i].barrier = &barrier;
		data[i].cuda_barrier = &cuda_barrier;
		data[i].cd = cuda_st;
		data[i].set_percent = set_percent;
		if(i == nb_threadsCPU){
			if (pthread_create(&threads[i], &attr, test_cuda, (void *)(&data[i])) != 0) {
			  fprintf(stderr, "Error creating thread\n");
			  exit(1);
			}
		}else{
			HeTM_CPULog[i] = NULL;
			if (pthread_create(&threads[i], &attr, test, (void *)(&data[i])) != 0) {
			  fprintf(stderr, "Error creating thread\n");
			  exit(1);
			}
		}

	  }
	  pthread_attr_destroy(&attr);

	  /* Catch some signals */
	  if (signal(SIGHUP, catcher) == SIG_ERR ||
		  signal(SIGINT, catcher) == SIG_ERR ||
		  signal(SIGTERM, catcher) == SIG_ERR) {
		perror("signal");
		exit(1);
	  }

	  /* Start threads */
	  barrier_cross(&barrier);

	  printf("STARTING...(Run %d)\n", j);

	  gettimeofday(&start, NULL);


	  if (duration > 0) {
		nanosleep(&timeout, NULL);
	  } else {
		sigemptyset(&block_set);
		sigsuspend(&block_set);
	  }
	  stop = 1;
#if GPUEn==0
	  cuda_stop=2;
#endif

	  gettimeofday(&end, NULL);
	  printf("STOPPING...\n");

	  /* Wait for thread completion */
	  for (i = 0; i < nb_threads; i++) {
		if (pthread_join(threads[i], NULL) != 0) {
		  fprintf(stderr, "Error waiting for thread completion\n");
		  exit(1);
		}
	  }

	  gettimeofday(&final, NULL);

	  duration = (end.tv_sec * 1000 + end.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);
	  duration2 = (final.tv_sec * 1000 + final.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);

	#ifndef TM_COMPILER
	  aborts = 0;
	  aborts_1 = 0;
	  aborts_2 = 0;
	  aborts_locked_read = 0;
	  aborts_locked_write = 0;
	  aborts_validate_read = 0;
	  aborts_validate_write = 0;
	  aborts_validate_commit = 0;
	  aborts_invalid_memory = 0;
	  aborts_killed = 0;
	  locked_reads_ok = 0;
	  locked_reads_failed = 0;
	  max_retries = 0;
	#endif /* ! TM_COMPILER */
	  reads = 0;
	  writes = 0;
	  updates = 0;

	  ret = 0;
	#if NODEBUG == 0
	  for (i = 0; i < nb_threadsCPU; i++) {
		printf("Thread %d\n", i);
		printf("  #transfer   : %lu\n", data[i].nb_transfer);
		printf("  #read-all   : %lu\n", data[i].nb_read_all);
		printf("  #write-all  : %lu\n", data[i].nb_write_all);
		printf("  #aborts     : %lu\n", data[i].nb_aborts);
	#ifndef TM_COMPILER
	#ifdef DEBUG
		printf("    #lock-r   : %lu\n", data[i].nb_aborts_locked_read);
		printf("    #lock-w   : %lu\n", data[i].nb_aborts_locked_write);
		printf("    #val-r    : %lu\n", data[i].nb_aborts_validate_read);
		printf("    #val-w    : %lu\n", data[i].nb_aborts_validate_write);
		printf("    #val-c    : %lu\n", data[i].nb_aborts_validate_commit);
		printf("    #inv-mem  : %lu\n", data[i].nb_aborts_invalid_memory);
		printf("    #killed   : %lu\n", data[i].nb_aborts_killed);
		printf("  #aborts>=1  : %lu\n", data[i].nb_aborts_1);
		printf("  #aborts>=2  : %lu\n", data[i].nb_aborts_2);
		printf("  #lr-ok      : %lu\n", data[i].locked_reads_ok);
		printf("  #lr-failed  : %lu\n", data[i].locked_reads_failed);
		printf("  Max retries : %lu\n", data[i].max_retries);
	#endif
		aborts += data[i].nb_aborts;
		aborts_1 += data[i].nb_aborts_1;
		aborts_2 += data[i].nb_aborts_2;
		aborts_locked_read += data[i].nb_aborts_locked_read;
		aborts_locked_write += data[i].nb_aborts_locked_write;
		aborts_validate_read += data[i].nb_aborts_validate_read;
		aborts_validate_write += data[i].nb_aborts_validate_write;
		aborts_validate_commit += data[i].nb_aborts_validate_commit;
		aborts_invalid_memory += data[i].nb_aborts_invalid_memory;
		aborts_killed += data[i].nb_aborts_killed;
		locked_reads_ok += data[i].locked_reads_ok;
		locked_reads_failed += data[i].locked_reads_failed;
		if (max_retries < data[i].max_retries)
		  max_retries = data[i].max_retries;
	#endif /* ! TM_COMPILER */
		updates += data[i].nb_transfer;
		reads += data[i].nb_read_all;
		writes += data[i].nb_write_all;
	  }
	  /* Sanity check */
	  ret = 0;
	  printf("Duration      : %d (ms)\n", duration);
	  printf("#txs          : %lu (%f / s)\n", reads + writes + updates, (reads + writes + updates) * 1000.0 / duration);
	  printf("#read txs     : %lu (%f / s)\n", reads, reads * 1000.0 / duration);
	  printf("#write txs    : %lu (%f / s)\n", writes, writes * 1000.0 / duration);
	  printf("#update txs   : %lu (%f / s)\n", updates, updates * 1000.0 / duration);
	  printf("#aborts       : %lu (%f / s)\n", aborts, aborts * 1000.0 / duration);
	#ifndef TM_COMPILER
	#ifdef DEBUG
	  printf("  #lock-r     : %lu (%f / s)\n", aborts_locked_read, aborts_locked_read * 1000.0 / duration);
	  printf("  #lock-w     : %lu (%f / s)\n", aborts_locked_write, aborts_locked_write * 1000.0 / duration);
	  printf("  #val-r      : %lu (%f / s)\n", aborts_validate_read, aborts_validate_read * 1000.0 / duration);
	  printf("  #val-w      : %lu (%f / s)\n", aborts_validate_write, aborts_validate_write * 1000.0 / duration);
	  printf("  #val-c      : %lu (%f / s)\n", aborts_validate_commit, aborts_validate_commit * 1000.0 / duration);
	  printf("  #inv-mem    : %lu (%f / s)\n", aborts_invalid_memory, aborts_invalid_memory * 1000.0 / duration);
	  printf("  #killed     : %lu (%f / s)\n", aborts_killed, aborts_killed * 1000.0 / duration);
	  printf("#aborts>=1    : %lu (%f / s)\n", aborts_1, aborts_1 * 1000.0 / duration);
	  printf("#aborts>=2    : %lu (%f / s)\n", aborts_2, aborts_2 * 1000.0 / duration);
	  printf("#lr-ok        : %lu (%f / s)\n", locked_reads_ok, locked_reads_ok * 1000.0 / duration);
	  printf("#lr-failed    : %lu (%f / s)\n", locked_reads_failed, locked_reads_failed * 1000.0 / duration);
	  printf("Max retries   : %lu\n", max_retries);
	#endif

	  /*for (i = 0; stm_get_ab_stats(i, &ab_stats) != 0; i++) {
		printf("Atomic block  : %d\n", i);
		printf("  #samples    : %lu\n", ab_stats.samples);
		printf("  Mean        : %f\n", ab_stats.mean);
		printf("  Variance    : %f\n", ab_stats.variance);
		printf("  Min         : %f\n", ab_stats.min);
		printf("  Max         : %f\n", ab_stats.max);
		printf("  50th perc.  : %f\n", ab_stats.percentile_50);
		printf("  90th perc.  : %f\n", ab_stats.percentile_90);
		printf("  95th perc.  : %f\n", ab_stats.percentile_95);
	  }*/
	#endif /* ! TM_COMPILER */
	  printf("Duration      : %d (ms)\n", duration);
	  printf("Combined OPs  :  (%f / s)\n", (reads + writes + updates + data[nb_threadsCPU].nb_transfer) * 1000.0 / duration2);
	  printf("Real duration : %d (ms)\n", duration2);
	#endif

	/* Save info between iterations*/
	throughput = (reads + writes + updates) * 1000.0 / duration2;
	throughput_gpu = (reads + writes + updates + data[nb_threadsCPU].nb_transfer) * 1000.0 / duration2;
	tot_commits_gpu[j] = data[nb_threadsCPU].nb_transfer;
	tot_throughput_gpu[j] = throughput_gpu;
	tot_aborts_gpu[j] =  data[nb_threadsCPU].nb_aborts;
	tot_comp[j] = data[nb_threadsCPU].nb_aborts_2;
	tot_tx[j] = data[nb_threadsCPU].max_retries;
	tot_cuda[j] = data[nb_threadsCPU].nb_aborts_1;
	tot_loop[j] = data[nb_threadsCPU].nb_aborts_locked_read;
	tot_loops[j] = data[nb_threadsCPU].nb_aborts_locked_write;
	tot_trf2cpu[j] = data[nb_threadsCPU].nb_aborts_validate_read;
	tot_trf2gpu[j] = data[nb_threadsCPU].nb_aborts_validate_write;
	tot_trfcmp[j] = data[nb_threadsCPU].nb_aborts_validate_commit;
	tot_commits[j] = reads + writes + updates;
	tot_duration[j] = duration;
	tot_duration2[j] = duration2;
	tot_throughput[j] = throughput;
	tot_aborts[j] = aborts;

	/* Cleanup STM */
	TM_EXIT;

	//Clean up the bank
	for (i = 0; i < bank->size*bank->ways; i++) {
		bank->accounts[i] = 0;
		bank->version[i] = 0;
	}
	memset(bitmap, 0,  bank->version_bm_size*sizeof(int));


	/*Cleanup GPU*/
	if ( j+1 == iter)
		printf("Finished all run(s)!\n");
	else
		jobWithCuda_exit(cuda_st);
  }

  DEBUG_PRINT("Sorting info.\n");

  /*Sort Arrays*/
  qsort(tot_duration,iter,sizeof(int),compare_int);
  qsort(tot_duration2,iter,sizeof(int),compare_int);
  qsort(tot_commits,iter,sizeof(int),compare_int);
  qsort(tot_aborts,iter,sizeof(int),compare_int);
  qsort(tot_commits_gpu,iter,sizeof(int),compare_int);
  qsort(tot_throughput,iter,sizeof(double),compare_double);
  qsort(tot_throughput_gpu,iter,sizeof(double),compare_double);
  qsort(tot_aborts_gpu,iter,sizeof(int),compare_int);
  qsort(tot_cuda,iter,sizeof(int),compare_int);
  qsort(tot_comp,iter,sizeof(int),compare_int);
  qsort(tot_tx,iter,sizeof(int),compare_int);
  qsort(tot_loop,iter,sizeof(int),compare_int);
  qsort(tot_loops,iter,sizeof(int),compare_int);
  qsort(tot_trf2cpu,iter,sizeof(int),compare_int);
  qsort(tot_trf2gpu,iter,sizeof(int),compare_int);
  qsort(tot_trfcmp,iter,sizeof(int),compare_int);

  /* Save to file */
  DEBUG_PRINT("Saving to file.\n");
  FILE *f = fopen(filename, "a");
  if (f == NULL)
	{
    printf("Error opening file!\n");
    exit(1);
  }
  if(ftell(f)==0) {	//New File, print headers
	fprintf(f,"sep=;\r\n");
	fprintf(f,"Nb Accounts(1);Exp Duration(2);Real Duration(3);CPU Commits(4);GPU Commits(5);CPU Aborts(6);GPU Aborts(7);CPU Throughput(8);GPU Throughput(9);");
	fprintf(f,"Total Lock Time(10);Total Comp Time(11);Transf CPU(12);Transf GPU(13);Transf Cmp(14);Tx Time(15);Nb GPU runs(16);Nb success(17);");
	fprintf(f,"Share Percent(18);Write Percent(19);Num_ways(20);Label(21);\r\n");
  }

  fprintf(f,"%d;%d;%d;%lu;",nb_accounts, tot_duration[iter/2], tot_duration2[iter/2], tot_commits[iter/2]);
  fprintf(f,"%lu;%lu;%lu;%f;%f;",tot_commits_gpu[iter/2], tot_aborts[iter/2], tot_aborts_gpu[iter/2], tot_throughput[iter/2], tot_throughput_gpu[iter/2]);
  fprintf(f,"%d;%d;%d;%d;%d;%d;", tot_cuda[iter/2], tot_comp[iter/2], tot_trf2cpu[iter/2], tot_trf2gpu[iter/2], tot_trfcmp[iter/2], tot_tx[iter/2]);
  fprintf(f,"%d;%d;%d;%d;%d;%d;%d;LABEL\r\n",  tot_loop[iter/2], tot_loops[iter/2], shared_percent, set_percent, num_ways,gb,gt);

  fclose(f);
  DEBUG_PRINT("Saved to file!\n");


  /* Delete bank and accounts */
  free(bank->accounts);
  free(bank->version);
  //free(bitmap);
  free(bank);
  jobWithCuda_exit(cuda_st);


  free(threads);
  free(data);

  return 0;
}
