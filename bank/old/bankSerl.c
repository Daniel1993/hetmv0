/*
 * File:
 *   bank.c
 * Author(s):
 *   Pascal Felber <pascal.felber@unine.ch>
 *   Patrick Marlier <patrick.marlier@unine.ch>
 * Description:
 *   Bank stress test.
 *
 * Copyright (c) 2007-2014.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, version 2
 * of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This program has a dual license and can also be distributed
 * under the terms of the MIT license.
 */

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

#include "shared.h"

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
#include "hetm-log.h"
#include "cuda_defines.h"

/*
 * Useful macros to work with transactions. Note that, to use nested
 * transactions, one should check the environment returned by
 * stm_get_env() and only call sigsetjmp() if it is not null.
 */

#ifndef USE_TSX_IMPL
// goes for TinySTM
#define TM_START(tid, ro)               /*{ stm_tx_attr_t _a = {{.id = tid, .read_only = ro}}; sigjmp_buf *_e = stm_start(_a); if (_e != NULL) sigsetjmp(*_e, 0)*/
#define TM_LOAD(addr)                   *addr; /*stm_load((stm_word_t *)addr)*/
#define TM_STORE(addr, value)           *addr = (__typeof__(*addr))value; /*stm_store((stm_word_t *)addr, (stm_word_t)value)*/
#define TM_COMMIT                       /*stm_commit(); }*/
#define TM_LOG(val)          		    //stm_log_add(0,val)
#define TM_LOG2(pos,val)          		//stm_log_add(pos,val)
#define TM_FREE(point)          		//stm_logel_free(point)
#define TM_GET_LOG(p)         				stm_get_stats("HeTM_CPULog", &p);

#define TM_INIT(nb_threads)		          /*stm_init(); mod_ab_init(0, NULL)*/
#define TM_EXIT                         /*stm_exit()*/
#define TM_INIT_THREAD(p,s)             /*stm_init_thread();*/ //stm_log_init_bm(p,s)
#define TM_EXIT_THREAD                  /*stm_exit_thread()*/
#else /* USE_TSX_IMPL */
// redefine with TSX
#include "tsx_impl.h"
#define TM_START(tid, ro) 		  /*HTM_SGL_begin();*/
#define TM_COMMIT 		          /*HTM_SGL_commit();*/
#define TM_LOAD(addr)           *addr; /*HTM_SGL_read(addr)*/
#define TM_STORE(addr, value)   *addr = (__typeof__(*addr))value; /*HTM_SGL_write(addr, value)*/

#define TM_GET_LOG(p)           HeTM_get_log(&p)
#define TM_LOG(val)             // stm_log_add(0,val)
#define TM_LOG2(pos,val)        // stm_log_add(pos,val)
#define TM_FREE(point)          // stm_logel_free(point)

#define TM_INIT(nb_threads)    /* stm_init(); mod_ab_init(0, NULL)*/
#define TM_EXIT                /* stm_exit()*/
#define TM_INIT_THREAD(p,s)     /* HTM_SGL_init_thr(); stm_init(); stm_log_initBM(HeTM_log, p, s)*/
#define TM_EXIT_THREAD          /* printf("\n   EXIT TSX\n\n"); \
	HTM_SGL_exit_thr(); stm_exit_thread()*/

#undef HTM_SGL_after_write
#define HTM_SGL_after_write(addr, val) /* empty, no log instrumentation */

#endif

// TODO: add no logged TSX here

#endif /* Compile with explicit calls to tinySTM */

#define DEFAULT_DURATION                5000
#define DEFAULT_NB_ACCOUNTS             100000
#define DEFAULT_NB_THREADS              4
#define DEFAULT_READ_ALL                20
#define DEFAULT_SEED                    0
#define DEFAULT_WRITE_ALL               0
#define DEFAULT_READ_THREADS            0
#define DEFAULT_WRITE_THREADS           0
#define DEFAULT_DISJOINT                0
#define DEFAULT_OUTPUT_FILE             Bank_LOG
#define DEFAULT_ITERATIONS              1
#define DEFAULT_NB_TRFS                 2

#ifndef MAX_THREADS
#define MAX_THREADS                     128
#endif

#define MAX_ITER                        100

#ifndef DIFFERENT_POOLS
#define DIFFERENT_POOLS                        1       //Enable limiting to accounts starting with LimAcc
#endif
#ifndef LimAcc
#define LimAcc                          0
#endif
#if	DIFFERENT_POOLS==1
#ifndef Cuda_Confl
#define Cuda_Confl                      1       //Enable limited conflicts
#endif
#ifndef CudaConflVal
#define	CudaConflVal                    16
#endif
#ifndef CudaConflVal2
#define CudaConflVal2                   0.99999 //LC 0.9995 -1 : MC 0.95 -1
#endif
#ifndef CudaConflVal3
#define	CudaConflVal3                   32
#endif
#endif

#define GPUEn                           0       //Use GPU acceleration
#ifndef CPUEn
#define CPUEn                           1       //Use CPU processing
#endif

#define XSTR(s)                         STR(s)
#define STR(s)                          #s

#define NODEBUG                         0       //Set to 1 to kill prints for obtained results
#define RUN_ONCE                        0       //Set to 1 to run only one cuda cycle

//Uncomment to use a data structure instead of an array.
#define USE_ARRAY

//Set to 1 to enable compressed transfer
#define BM_TRANSF                       1

//Uncomment to allow CPU invalidation
//#define CPU_INV

//Uncomment to enable cpu/gpu balancing
//#define SYNC_BALANCING
#define	SYNC_BALANCING_VALS             20
#define SYNC_BALANCING_VALF             10

//Uncoment to uses single threaded comparison
#define USE_STREAM
#define STREAM_COUNT                    32

#if LOG_TYPE == 2
#ifdef USE_STREAM
	//#define NO_LOCK
	#define LOCK_VAL                      4
#endif
#endif

#define RESET_MUTEX(val,mutex) \
	pthread_mutex_lock(&mutex); \
	val = 0; \
	pthread_mutex_unlock(&mutex);


/* ################################################################### *
 * Sorting
 * ################################################################### */

int compare_int (const void *a, const void *b) {
  const int *da = (const int *) a;
  const int *db = (const int *) b;

  return (*da > *db) - (*da < *db);
}

int compare_double (const void *a, const void *b) {
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}


/* ################################################################### *
 * GLOBALS
 * ################################################################### */

static volatile int stop;
static volatile int cuda_stop;
static volatile int cpu_enable_flag;
static long long int global_fix;
#if GPUEn == 1
#ifdef USE_STREAM
static volatile int cmp_flag;
#endif
static volatile HeTM_CPULogNode_t *HeTM_CPULog[MAX_THREADS];
#ifdef CPU_INV
static int global_commits[MAX_THREADS];
#endif
#endif

/* ################################################################### *
 * BANK ACCOUNTS
 * ################################################################### */

 #ifndef USE_ARRAY
typedef struct account {
  long number;
  long balance;
} __attribute__((aligned (64))) account_t;
#else
typedef long __attribute__((aligned (64))) account_t;
#endif

typedef struct bank {
  account_t *accounts;
  long size;
} bank_t;

int transfer(account_t * accounts, int *positions, int count, int amount)
{
  long i;
  int z = 0;
  int n, src, dst;

  /* Allow overdrafts */
  TM_START(z, RW);

  for(n=0; n<count; n+=2) {
	 src=positions[n];
	 dst=positions[n+1];
	 #ifndef USE_ARRAY
	  i = TM_LOAD(&accounts[src]->balance);
	  i -= amount;
	  TM_STORE(&accounts[src]->balance, i);

	  i = TM_LOAD(&accounts[dst]->balance);
	  i += amount;
	  TM_STORE(&accounts[dst]->balance, i);
	#else
	  i = TM_LOAD(&accounts[src]);
	  i -= amount;
	  TM_STORE(&accounts[src], i);

	  i = TM_LOAD(&accounts[dst]);
	  i += amount;
	  TM_STORE(&accounts[dst], i);
	#endif
  }

  TM_COMMIT;

#if LOG_AUTO==0
  for(n=0; n<count; n+=2) {
	  TM_LOG(positions[n]);
	  TM_LOG(positions[n+1]);
	}
#endif
 /* TM_LOG2(&accounts[src],accounts[src]);
  TM_LOG2(&accounts[dst],accounts[dst]);*/
  //usleep(500);

  return amount;
}

int total(bank_t *bank, int transactional)
{
  long i;
  long total;

  if (!transactional) {
    total = 0;
    for (i = 0; i < bank->size; i++) {
#ifndef USE_ARRAY
      total += bank->accounts[i].balance;
#else
	  total += bank->accounts[i];
#endif
    }
  } else {
    TM_START(1, RO);
    total = 0;
    for (i = 0; i < bank->size; i++) {
#ifndef USE_ARRAY
      total += TM_LOAD(&bank->accounts[i].balance);
#else
	  total += TM_LOAD(&bank->accounts[i]);
#endif
    }
    TM_COMMIT;
  }

  return total;
}

static void reset(bank_t *bank)
{
  long i;

  TM_START(2, RW);
  for (i = 0; i < bank->size; i++) {
#ifndef USE_ARRAY
    TM_STORE(&bank->accounts[i].balance, 0);
#else
    TM_STORE(&bank->accounts[i], 0);
#endif
  }
  TM_COMMIT;
}

/* ################################################################### *
 * BARRIER
 * ################################################################### */

typedef struct barrier {
  pthread_cond_t complete;
  pthread_mutex_t mutex;
  int count;
  int crossing;
} barrier_t;

static void barrier_init(barrier_t *b, int n)
{
  pthread_cond_init(&b->complete, NULL);
  pthread_mutex_init(&b->mutex, NULL);
  b->count = n;
  b->crossing = 0;
}

static void barrier_cross(barrier_t *b)
{
  pthread_mutex_lock(&b->mutex);
  /* One more thread through */
  b->crossing++;
  /* If not all here, wait */
  if (b->crossing < b->count) {
    pthread_cond_wait(&b->complete, &b->mutex);
  } else {
    pthread_cond_broadcast(&b->complete);
    /* Reset for next time */
    b->crossing = 0;
  }
  pthread_mutex_unlock(&b->mutex);
}

/* ################################################################### *
 * THREAD SRTUCTURE
 * ################################################################### */

typedef struct thread_data {
  bank_t *bank;
  barrier_t *barrier;
  barrier_t *cuda_barrier;
#if GPUEn==1
  cuda_t * cd;
#endif
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
  int trfs;
  char padding[64];
} thread_data_t;

typedef enum {READ, STREAM, FINAL, END} cmp_status;


/* ################################################################### *
 * TRANSACTION THREADS
 * ################################################################### */

static void *test(void *data)
{
  int src, dst, nb;
  int accounts_vec[MAX_ITER], i;
  int rand_max, rand_min;
  thread_data_t *d = (thread_data_t *)data;
  int nb_accounts = d->bank->size;
  unsigned short seed[3];

	bindThread(d->id);

#if GPUEn == 1
  void * p = NULL;
  int total_el=0;
  HeTM_CPULogNode_t * logAux;
  cmp_status state=FINAL;
#ifdef USE_STREAM
  int n = 0, count = 0, res = 0;
  int * flag_check = (int*) CudaMallocWrapper(sizeof(int));
  stream_t *s = jobWithCuda_initStream(d->cd, d->id, STREAM_COUNT, flag_check);
  HeTM_CPULogEntry * logVec[STREAM_COUNT];
  for(count=0; count<STREAM_COUNT; count++)
	logVec[count] = (HeTM_CPULogEntry*) CudaMallocWrapper(LOG_SIZE*sizeof(HeTM_CPULogEntry));
#ifdef NO_LOCK
  state = READ;
#endif
#endif
#endif

#if DIFFERENT_POOLS == 1
  nb_accounts = nb_accounts >> 1;
#endif

  /* Initialize seed (use rand48 as rand is poor) */
  seed[0] = (unsigned short)rand_r(&d->seed);
  seed[1] = (unsigned short)rand_r(&d->seed);
  seed[2] = (unsigned short)rand_r(&d->seed);

  /* Prepare for disjoint access */
  if (d->disjoint) {
	rand_max = nb_accounts / d->nb_threads;
	rand_min = rand_max * d->id;
    if (rand_max <= 2) {
      fprintf(stderr, "can't have disjoint account accesses");
      return NULL;
    }
  } else {
#if LimAcc==0
    rand_max = nb_accounts;
#else
	rand_max= LimAcc;
#endif
	//rand_max= 150;
	//rand_max= 650;
    rand_min = 0;
  }

  /* Create transaction */
  TM_INIT_THREAD(d->bank->accounts, d->bank->size);
  /* Wait on barrier */
  barrier_cross(d->barrier);

  while (stop == 0 || cuda_stop!=2) {

#if GPUEn == 1
	if ( cuda_stop == 1 ) {

		if (state==FINAL) {

	#ifndef NO_LOCK
			//get stats
			TM_GET_LOG(p); // stm_get_stats("HeTM_CPULog", &p);
			HeTM_CPULog[d->id] = logAux = p;
			total_el = logAux->nb_el;
			DEBUG_PRINT("Thread %d stopping, to compare %d elements.\n",d->id, total_el);
			barrier_cross(d->cuda_barrier);				//Stop to synchronize with GPU
	#endif

			//DEBUG_PRINT("Thread %d locked.\n",d->id);
#ifdef USE_STREAM
			(*flag_check) = 0;
	#ifndef NO_LOCK
			barrier_cross(d->cuda_barrier);				//Wait for the GPU to be updated with CPU data
	#else
			state=READ;
	#endif
			//Main Comparison Loop
			RESET_MUTEX(s->count,s->mutex);
			while(logAux!=NULL && (*flag_check)==0 && (cmp_flag)==0 ){
				count=0;

				//(*flag_check )= jobWithCuda_checkStream(*d->cd, logVec, logAux->curPos, d->id, d->st,NULL, NULL);
				for(n=0; n<STREAM_COUNT && logAux!=NULL; n++) {
					memcpy(logVec[n], logAux->array, LOG_SIZE*sizeof(HeTM_CPULogEntry));

					res = jobWithCuda_checkStream(*d->cd, s, logVec[n], logAux->curPos, n);
					logAux = logAux->next;
					count++;
				}
				n = 0;

				/*Check if done*/
				while(n < count) {
					pthread_mutex_lock(&s->mutex);
					n = s->count;
					pthread_mutex_unlock(&s->mutex);
				}
				total_el-=n;

				//jobWithCuda_wait();

				if(*flag_check==-1) {
					cmp_flag = -1;
				} else
					jobWithCuda_checkStreamFinal(*d->cd,s,count);
			}
			DEBUG_PRINT("Thread %d finished cmp.\n",d->id);
			if(cmp_flag!=-1)
				cmp_flag |= *flag_check;

			barrier_cross(d->cuda_barrier);				//Signal end of comparisson
	#endif

			//wait for every thread to unlock			//Wait for synchronization to conclude
			barrier_cross(d->cuda_barrier);
			DEBUG_PRINT("Thread %d unlocked.\n",d->id);
			continue;
		}

		//No Lock launch
	#ifdef NO_LOCK
		if ( state!=FINAL) {

			if(state==READ) {			//READ DATA
				//get stats
				TM_GET_LOG(p); // stm_get_stats("HeTM_CPULog", &p);
				HeTM_CPULog[d->id] = logAux = p;
				total_el = logAux->nb_el;
				(*flag_check) = 0;

				if(total_el <= LOCK_VAL) {
					DEBUG_PRINT("Thread[%d]: Starting locking comparison. (With %d values).\n", d->id, logAux->curPos);
					state=FINAL;
				} else {
					DEBUG_PRINT("Thread[%d]: Starting comparison.\n", d->id);
					state=STREAM;
				}
				DEBUG_PRINT("Thread[%d]: %d logs\n", d->id,total_el);
				n=0; count=0;
				RESET_MUTEX(s->count,s->mutex);
			} else {					//CHECK CMP STATUS
				pthread_mutex_lock(&s->mutex);
				n = s->count;
				pthread_mutex_unlock(&s->mutex);
			}

			if(state==FINAL)
				continue;

			if(n == count) {
				//DEBUG_PRINT("Thread[%d]: Stream finished.\n", d->id);
				cmp_flag |= *flag_check;

				if(*flag_check!=0) {			//Check for colision
					state=FINAL;

				} else {
					total_el-=n;
					jobWithCuda_checkStreamFinal(*d->cd,s,count);

					if(total_el==0) {			//FINISHED A CYCLE
						state=READ;
						logAux = HeTM_CPULog[d->id];

						while(logAux!=NULL) {
							p=logAux;
							logAux = logAux->next;
							TM_FREE(p);
						}
						HeTM_CPULog[d->id] = p = logAux = NULL;

						DEBUG_PRINT("Thread[%d]: Finished comparing a set.\n", d->id);

					} else {				//CYCLE NOT FINISHED

						//DEBUG_PRINT("Thread[%d]: Queueing new log comparison, with %d remaining.\n", d->id , total_el);
						count=0;
						for(n=0; n<STREAM_COUNT && logAux!=NULL; n++) {
							memcpy(logVec[n], logAux->array, LOG_SIZE*sizeof(HeTM_CPULogEntry));

							jobWithCuda_checkStream(*d->cd, s, logVec[n], logAux->curPos, n);
							logAux = logAux->next;
							count++;
						}
						n=0;

						//DEBUG_PRINT("Thread[%d]: Queued new log comparisson.\n", d->id);
					}
				}
			}

			//Check if another threads comparison failed
			if(cmp_flag!=0 ) {
				state=FINAL;
				/*logAux = HeTM_CPULog[d->id];
				while(logAux!=NULL) {
					p=logAux;
					logAux = logAux->next;
					TM_FREE(p);
				}
				HeTM_CPULog[d->id] = p = logAux = NULL;*/
				continue;

			}
		}
	#endif
	}
#endif

	if(cpu_enable_flag == 1) {
		if (d->id < d->read_threads) {
		  /* Read all */
		  total(d->bank, 1);
		  d->nb_read_all++;
		} else if (d->id < d->read_threads + d->write_threads) {
		  /* Write all */
		  reset(d->bank);
		  d->nb_write_all++;
		} else {
		  nb = (int)(erand48(seed) * 100);
		  if (nb < d->read_all) {
			/* Read all */
			total(d->bank, 1);
			d->nb_read_all++;
		  } else if (nb < d->read_all + d->write_all) {
			/* Write all */
			reset(d->bank);
			d->nb_write_all++;
		  } else {
			/* Choose random accounts */
			for(i=0; i<d->trfs; i+=2) {
				src = (int)(erand48(seed) * rand_max) + rand_min;
				dst = (int)(erand48(seed) * rand_max) + rand_min;

				if (dst == src)
					dst = ((src + 1) % rand_max) + rand_min;
	#if DIFFERENT_POOLS==1
				src = src + nb_accounts;
				dst = dst + nb_accounts;
	#endif
				accounts_vec[i]=src;
				accounts_vec[i+1]=dst;

	#if Cuda_Confl==1
				float cfl = erand48(seed);
				if (cfl > CudaConflVal2) {
					int size = nb_accounts / CudaConflVal;
					int overlap = size / CudaConflVal3;
					src = erand48(seed)*size + nb_accounts - overlap;
				}
				accounts_vec[i]=src;
	#endif
			}

			transfer(d->bank->accounts, accounts_vec, d->trfs, 1);
			d->nb_transfer++;
#ifdef CPU_INV
			global_commits[d->id]++;
#endif
		 }
		}
	}
  }

  DEBUG_PRINT("Thread %d exiting!\n", d->id);
	#ifdef USE_TSX_IMPL
	d->nb_aborts = TM_get_error(ABORT);
	d->nb_aborts_1 = TM_get_error(CONFLICT);
	d->nb_aborts_2 = TM_get_error(CAPACITY);
	#endif
#if  !defined(TM_COMPILER) && !defined(USE_TSX_IMPL)
  // stm_get_stats("nb_aborts", &d->nb_aborts);
  // stm_get_stats("nb_aborts_1", &d->nb_aborts_1);
  // stm_get_stats("nb_aborts_2", &d->nb_aborts_2);
  // stm_get_stats("nb_aborts_locked_read", &d->nb_aborts_locked_read);
  // stm_get_stats("nb_aborts_locked_write", &d->nb_aborts_locked_write);
  // stm_get_stats("nb_aborts_validate_read", &d->nb_aborts_validate_read);
  // stm_get_stats("nb_aborts_validate_write", &d->nb_aborts_validate_write);
  // stm_get_stats("nb_aborts_validate_commit", &d->nb_aborts_validate_commit);
  // stm_get_stats("nb_aborts_invalid_memory", &d->nb_aborts_invalid_memory);
  // stm_get_stats("nb_aborts_killed", &d->nb_aborts_killed);
  // stm_get_stats("locked_reads_ok", &d->locked_reads_ok);
  // stm_get_stats("locked_reads_failed", &d->locked_reads_failed);
  // stm_get_stats("max_retries", &d->max_retries);
#endif /* ! TM_COMPILER */
  /* Free transaction */

#if GPUEn==1
#ifdef USE_STREAM
  CudaFreeWrapper(flag_check);
  for(n = 0; n<STREAM_COUNT; n++)
	CudaFreeWrapper(logVec[n]);
  jobWithCuda_exitStream(s);
#endif
#endif

  TM_EXIT_THREAD;

  return NULL;
}


/* ################################################################### *
 * CUDA THREAD
 * ################################################################### */

#if GPUEn == 1
static void *test_cuda(void *data) {
  thread_data_t *d = (thread_data_t *)data;
  int i, n, loop=0;
  int flag, flag_check = 0;
  int counter_aborts = 0, counter_commits = 0, counter_sync = 0;		//Counters
  int duration= 0, tx_duration = 0;										//Timers
  int trf_2cpu= 0, trf_2gpu = 0;										//Timers
  float	duration_cmp = 0, trans_cmp = 0;
  struct timeval tx_start, tx_end, lock_start, lock_end, cmp_start, cmp_end;				//Timer structure
  HeTM_CPULogNode_t *aux, *vec_log[MAX_THREADS];
  cuda_t *cd = d->cd;
  long * a = d->bank->accounts;
  long * helper = NULL;
  int sync_fail = 0, sync_sucess = 0;

#ifndef	USE_ARRAY
  a = (long *)malloc( d->bank->size * sizeof(long));
  for (i = 0; i < d->bank->size; i++) {
	a[i] =  d->bank->accounts[i].balance;
#endif

#ifdef CPU_INV
	int expected_commits=DEFAULT_TransEachThread*DEFAULT_threadNum*DEFAULT_blockNum;    					///FIXME
	for(i=0; i< d->nb_threads-1; i++)
		global_commits[i]=0;
#endif

  int * valid_vec = NULL;
#if CMP_APPLY==0
  valid_vec = (int *)malloc( d->bank->size * sizeof(int));
#endif
#if BM_TRANSF==1
  valid_vec = (int *)malloc( d->bank->size * sizeof(int));
#endif

  gettimeofday(&tx_start, NULL);

  DEBUG_PRINT("Starting GPU execution.\n");
  flag = jobWithCuda_run(cd,a);					//Start execution
  barrier_cross(d->barrier);						//Signal CPU threads to start


  if (flag == 0) {									//Cuda error detection
	printf("CUDA run failed.\n");
	gettimeofday(&tx_end, NULL);
	cuda_stop = 2;
  } else {
	DEBUG_PRINT("Starting transaction kernel!\n");
  }

  //Cicle till execution concludes
  while (flag) {									//MAIN LOOP
    loop++; helper = NULL;

	jobWithCuda_wait();								//Wait for GPU execution to conclude

	gettimeofday(&tx_end, NULL);
	DEBUG_PRINT("Transaction kernel concluded\n");
	tx_duration += (tx_end.tv_sec * 1000 + tx_end.tv_usec / 1000) - (tx_start.tv_sec * 1000 + tx_start.tv_usec / 1000);

#ifdef CPU_INV
	jobWithCuda_backup(cd);
#endif

#ifndef NO_LOCK
	//Lock threads
	cuda_stop = 1;
	barrier_cross(d->cuda_barrier);
#endif

	DEBUG_PRINT("All threads locked\n");								/* LOCKED */
	gettimeofday(&lock_start, NULL);
	flag_check = 0;

#if CMP_APPLY==1
	//Copy host data to device
	if ( !jobWithCuda_dupd(*cd, a) )
		flag_check=-1;
	gettimeofday(&cmp_start, NULL);
	trf_2gpu += (cmp_start.tv_sec * 1000 + cmp_start.tv_usec / 1000) - (lock_start.tv_sec * 1000 + lock_start.tv_usec / 1000);
#endif

#ifdef USE_STREAM
    struct timeval timerStream;				//Timer
	DEBUG_PRINT("Starting comparisson kernel!\n");

#ifdef NO_LOCK
	cuda_stop = 1;
#else
	barrier_cross(d->cuda_barrier);
#endif
#if BM_TRANSF==1
	jobWithCuda_bm(*cd, valid_vec);
#endif
	barrier_cross(d->cuda_barrier);

	flag_check = cmp_flag;

	gettimeofday(&timerStream, NULL);
	duration_cmp +=  (timerStream.tv_sec * 1000 + timerStream.tv_usec / 1000) - (cmp_start.tv_sec * 1000 + cmp_start.tv_usec / 1000);
#else
#if BM_TRANSF==1
	jobWithCuda_bm(*cd, valid_vec);
#endif
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

	DEBUG_PRINT("Final comparison result: %d\n", flag_check);

#ifdef CPU_INV
	//Check if GPU needs to invalidate CPU
	if ( flag_check == 1) {
		int global;
		global=0;
		for(i=0; i< d->nb_threads-1; i++)
			global+=global_commits[i];

		if (global < expected_commits) {
			DEBUG_PRINT("Invalidating CPU.\n");
			flag_check=0;
			global_fix+=global;
			counter_sync--;
			jobWithCuda_backupRestore(cd);
		}
	}
	for(i=0; i< d->nb_threads-1; i++)
		global_commits[i]=0;
#endif

	//Transfer back results to CPU
	gettimeofday(&cmp_start, NULL);
	if(flag_check == 0) {						//Successful execution
		sync_fail = 0; sync_sucess++; 			//Sequence Counter
		counter_sync++;							//Success Counter

		if ( jobWithCuda_hupd(cd, a, valid_vec ) == 1 ) {
#ifndef	USE_ARRAY
			//Update host data
			for (i = 0; i < d->bank->size; i++)		//Save value only if it is present in the compressed log
				d->bank->accounts[i].balance = valid_vec[i] <2 ? d->bank->accounts[i].balance : a[i];
#endif
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
		if(sync_fail == SYNC_BALANCING_VALF)
			cpu_enable_flag = 0;
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
	gettimeofday(&lock_end, NULL);
	barrier_cross(d->cuda_barrier);

	DEBUG_PRINT("All threads unlocked\n");

	//Rerun transaction kernel
	if(cuda_stop != 2) {
		if( flag_check == 0 )
			jobWithCuda_getStats(*cd,&counter_aborts,&counter_commits);						//Update stats
#if CMP_APPLY==0
		if ( jobWithCuda_run(cd,a ))
			DEBUG_PRINT("Running new transaction kernel!\n");
		gettimeofday(&tx_start, NULL);

		//Update timer
		trf_2gpu += (tx_start.tv_sec * 1000 + tx_start.tv_usec / 1000) - (lock_end.tv_sec * 1000 + lock_end.tv_usec / 1000);
#else
		if ( jobWithCuda_run(cd, helper ))
			DEBUG_PRINT("Running new transaction kernel!\n");

		gettimeofday(&tx_start, NULL);
#endif
	}

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

  d->nb_transfer = global_fix == 0 ? counter_sync * counter_commits : loop * counter_commits;
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
#ifndef	USE_ARRAY
  free(a);
#endif
#if CMP_APPLY==0
  free(valid_vec);
#endif
#if BM_TRANSF==1
  free(valid_vec);
#endif
  return NULL;
}
#endif

/* Signal Catcher */
static void catcher(int sig)
{
  static int nb = 0;
  printf("CAUGHT SIGNAL %d\n", sig);
  if (++nb >= 1)
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
	{"trfs",            		  required_argument, NULL, 't'},
	{"tperthread",            	  required_argument, NULL, 'T'},
	{"output-file",               required_argument, NULL, 'f'},
    {"disjoint",                  no_argument,       NULL, 'j'},
    {NULL, 0, NULL, 0}
  };

  bank_t *bank;
#if GPUEn == 1
  barrier_t cuda_barrier;
#endif
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
  int trfs = DEFAULT_NB_TRFS;
  int trans = DEFAULT_TransEachThread;
  double throughput, tot_throughput[MAX_ITER];
  unsigned int tot_duration[MAX_ITER], tot_duration2[MAX_ITER];
  unsigned long tot_commits[MAX_ITER], tot_aborts[MAX_ITER];
 #if GPUEn == 1
  cuda_t * cuda_st;
  double throughput_gpu, tot_throughput_gpu[MAX_ITER];
  unsigned int tot_comp[MAX_ITER], tot_tx[MAX_ITER], tot_cuda[MAX_ITER];
  unsigned int tot_loop[MAX_ITER], tot_loops[MAX_ITER], tot_trf2cpu[MAX_ITER], tot_trf2gpu[MAX_ITER], tot_trfcmp[MAX_ITER];;
  unsigned long tot_commits_gpu[MAX_ITER], tot_aborts_gpu[MAX_ITER];
#endif
  sigset_t block_set;
  char * fn = NULL;
  char filename[128];


  while(1) {
    i = 0;
    c = getopt_long(argc, argv, "ha:c:d:n:r:R:s:w:W:i:t:T:f:j", long_options, &i);

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
              "        Percentage of read-all transactions (default=" XSTR(DEFAULT_READ_ALL) ")\n"
              "  -R, --read-threads <int>\n"
              "        Number of threads issuing only read-all transactions (default=" XSTR(DEFAULT_READ_THREADS) ")\n"
              "  -s, --seed <int>\n"
              "        RNG seed (0=time-based, default=" XSTR(DEFAULT_SEED) ")\n"
              "  -w, --write-all-rate <int>\n"
              "        Percentage of write-all transactions (default=" XSTR(DEFAULT_WRITE_ALL) ")\n"
              "  -W, --write-threads <int>\n"
              "        Number of threads issuing only write-all transactions (default=" XSTR(DEFAULT_WRITE_THREADS) ")\n"
			  "  -i, --num-iterations <int>\n"
              "        Number of iterations to execute (default=" XSTR(DEFAULT_ITERATIONS) ")\n"
			  "  -t, --num-transfers <int>\n"
              "        Number of accounts to transfer between in each transaction (default=" XSTR(DEFAULT_NB_TRFS) ")\n"
			  "  -T, --num-txs-pergputhread <int>\n"
              "        Number of transactions per GPU thread (default=" XSTR(DEFAULT_TransEachThread) ")\n"
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
	 case 't':
       trfs = atoi(optarg);
       break;
	 case 'T':
       trans = atoi(optarg);
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
  assert(nb_threads > 0 && nb_threads<MAX_THREADS);
  assert(read_all >= 0 && write_all >= 0 && read_all + write_all <= 100);
  assert(read_threads + write_threads <= nb_threads);
  assert(iter <= MAX_ITER);
  assert(trfs <= MAX_ITER);
  assert(trans >= 0);

  printf("Nb accounts    : %d\n", nb_accounts);
#ifndef TM_COMPILER
  printf("CM             : %s\n", (cm == NULL ? "DEFAULT" : cm));
#endif /* ! TM_COMPILER */
  printf("Duration       : %d\n", duration);
  printf("Iterations     : %d\n", iter);
  printf("Nb threads     : %d\n", nb_threads);
  printf("Read-all rate  : %d\n", read_all);
  printf("Read threads   : %d\n", read_threads);
  printf("Seed           : %d\n", seed);
  printf("Write-all rate : %d\n", write_all);
  printf("Write threads  : %d\n", write_threads);
  printf("Transfers      : %d\n", trfs);
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

#if GPUEn == 1
  nb_threads++;
  barrier_init(&cuda_barrier, nb_threads );
#endif

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
  bank->accounts = (account_t *)malloc(nb_accounts * sizeof(account_t));
  bank->size = nb_accounts;
#ifndef USE_ARRAY
  for (i = 0; i < bank->size; i++) {
    bank->accounts[i].number = i;
    bank->accounts[i].balance = 0;
  }
#else
  memset(bank->accounts, 0, nb_accounts*sizeof(long));
#endif
  printf("Total before   : %d\n", total(bank, 0));

  for(j = 0; j<iter; j++) {				//Loop for testing purposes

	  //Init GPU
#if GPUEn == 1
	/*int logSize = LOG_SIZE;
#ifdef USE_ARRAY
	logSize = (LOG_SIZE*nb_threads);
#endif*/
	  DEBUG_PRINT("Initializing GPU.\n");

	  cuda_st = jobWithCuda_init(bank->accounts, bank->size, trans, 0, 0, 0);
	  //DEBUG_PRINT("Base: %lu %lu \n", bank->accounts, &bank->accounts);
	  if (cuda_st == NULL) {
		printf("CUDA init failed.\n");
		exit(-1);
	  }
#endif

	  //Clear flags
	  cuda_stop = stop = 0;
	  cpu_enable_flag = 1;
	  global_fix = 0;
#if CPUEn==0
	  cpu_enable_flag = 0;
#endif

	  /* Init STM */
	  printf("Initializing STM\n");
	  TM_INIT(nb_threads);

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
		data[i].trfs = trfs;
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
	#if GPUEn == 1
		data[i].cuda_barrier = &cuda_barrier;
		data[i].cd = cuda_st;
		if(i+1 == nb_threads){
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
	#else
		if (pthread_create(&threads[i], &attr, test, (void *)(&data[i])) != 0) {
		  fprintf(stderr, "Error creating thread\n");
		  exit(1);
		}
	#endif
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

	  ret = total(bank, 0);
	#if NODEBUG == 0
	  for (i = 0; i < nb_threadsCPU; i++) {
	#ifndef TM_COMPILER
	#ifdef DEBUG
		printf("Thread %d\n", i);
		printf("  #transfer   : %lu\n", data[i].nb_transfer);
		printf("  #read-all   : %lu\n", data[i].nb_read_all);
		printf("  #write-all  : %lu\n", data[i].nb_write_all);
		printf("  #aborts     : %lu\n", data[i].nb_aborts);
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
	  } updates -= global_fix;
	  /* Sanity check */
	  ret = total(bank, 0);
	  printf("Bank total    : %d (expected: 0)\n", ret);
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
	  printf("Real duration : %d (ms)\n", duration2);
	#endif

	/* Save info between iterations*/
	throughput = (reads + writes + updates) * 1000.0 / duration2;
#if GPUEn == 1
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
#endif
	tot_commits[j] = reads + writes + updates;
	tot_duration[j] = duration;
	tot_duration2[j] = duration2;
	tot_throughput[j] = throughput;
	tot_aborts[j] = aborts;

	/* Cleanup STM */
	TM_EXIT;

	//Clean up the bank
	for (i = 0; i < bank->size; i++) {
#ifndef USE_ARRAY
		bank->accounts[i].balance = 0;
#else
		bank->accounts[i] = 0;
#endif
	}
#if GPUEn==1
	/*Cleanup GPU*/
	jobWithCuda_exit(cuda_st);
	free(cuda_st);
#endif
  }

  DEBUG_PRINT("Sorting info.\n");

  /*Sort Arrays*/
  qsort(tot_duration,iter,sizeof(int),compare_int);
  qsort(tot_duration2,iter,sizeof(int),compare_int);
  qsort(tot_commits,iter,sizeof(int),compare_int);
  qsort(tot_aborts,iter,sizeof(int),compare_int);
#if GPUEn == 1
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
#endif

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
	fprintf(f,"Nb Accounts(1);Exp Duration(2);Real Duration(3);CPU Commits(4);GPU Commits(5);CPU Aborts(6);CPU Throughput(7);GPU Throughput(8);GPU Aborts(9);");
	fprintf(f,"Total Lock Time(10);Total Comp Time(11);Transf CPU(12);Transf GPU(13);Transf Cmp(14);Tx Time(15);Nb GPU runs(16);Nb success(17);Accts per TX(18)\r\n");
  }

  fprintf(f,"%d;%d;%d;%lu;",nb_accounts, tot_duration[iter/2], tot_duration2[iter/2], tot_commits[iter/2]);
#if GPUEn == 1
  fprintf(f,"%lu;%lu;%f;%f;%lu;",tot_commits_gpu[iter/2], tot_aborts[iter/2], tot_throughput[iter/2], tot_throughput_gpu[iter/2], tot_aborts_gpu[iter/2]);
  fprintf(f,"%d;%d;%d;%d;%d;%d;", tot_cuda[iter/2], tot_comp[iter/2], tot_trf2cpu[iter/2], tot_trf2gpu[iter/2], tot_trfcmp[iter/2], tot_tx[iter/2]);
  fprintf(f,"%d;%d;%d;%d\r\n",  tot_loop[iter/2], tot_loops[iter/2], trfs, trans);
  #else
  fprintf(f,"0;%lu;%f;0;0;0;0;0;0;0;0;0;0;%d;%d\r\n",tot_aborts[iter/2], tot_throughput[iter/2], trfs, trans);
#endif
  fclose(f);


  /* Delete bank and accounts */
#if GPUEn==0
  CudaFreeWrapper(bank->accounts);
#endif
  free(bank);

  free(threads);
  free(data);

  return ret;
  //printf("RETURN %d\n",ret);
  //return 0;
}
