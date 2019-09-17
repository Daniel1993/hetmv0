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

#define HETM_INSTRUMENT_CPU
#define HETM_LOG_TYPE 1
#define HETM_CMP_TYPE 1

#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

//#include "../common/rnd.h"
#include "../common/rdtsc.h"
#include "tbb/concurrent_queue.h"
#include "hetm.cuh"

#ifdef USE_HTM
#define USE_TSX_IMPL
#endif
#include "stm-wrapper.h"

#ifdef DEBUG
# define IO_FLUSH                       fflush(NULL)
/* Note: stdio is thread-safe */
#endif

#ifndef CPU_FREQ
#error "Pls define CPU_FREQ in kHz"
#endif

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

#endif /* Compile with explicit calls to tinySTM */

#define DEFAULT_DURATION                10000
#define DEFAULT_NB_ACCOUNTS             1024
#define DEFAULT_NB_THREADS              1
#define DEFAULT_READ_ALL                20
#define DEFAULT_SEED                    0
#define DEFAULT_WRITE_ALL               0
#define DEFAULT_READ_THREADS            0
#define DEFAULT_WRITE_THREADS           0
#define DEFAULT_DISJOINT                0

#define XSTR(s)                         STR(s)
#define STR(s)                          #s

#ifndef MAX_NB_THRS
#define MAX_NB_THRS 16
#endif /* MAX_NB_THRS */

/* ################################################################### *
 * GLOBALS
 * ################################################################### */

static volatile int stop __attribute__((aligned(64)));

static const char *STATS_FILE_PATH = "./bank_stats.csv";
static FILE *stats_file;

/* ################################################################### *
 * BANK ACCOUNTS
 * ################################################################### */

typedef struct account {
  long number;
  long balance;
} __attribute__((aligned(64))) account_t;

typedef struct bank {
  account_t *accounts;
  long size;
} bank_t;

typedef enum {
  TRANSFER,
  TOTAL,
  RESET
} REQUEST_TYPE;

typedef struct bank_request {
  REQUEST_TYPE req;
  volatile int isDone;
  int src;
  int dst;
  int amount;
  int transactional; // for total
  int result;
} __attribute__((aligned(64))) bank_request_t;

static void pinThisThread(int coreID)
{
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(coreID, &cpu_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
}

#define W1_SIZE_READ   4
#define W1_SIZE_WRITE  4

#define W2_SIZE_READ  40
#define W2_SIZE_WRITE  4

#if    USE_WORKLOAD == 1
#define SIZE_READ  W1_SIZE_READ
#define SIZE_WRITE W1_SIZE_WRITE
#elif  USE_WORKLOAD == 2
#define SIZE_READ  W2_SIZE_READ
#define SIZE_WRITE W2_SIZE_WRITE
#endif /* USE_WORKLOAD */

static int read_tx(bank_t *bank, int src_seed)
{
  long i;
  int result = 0;
  uint64_t seed = src_seed;
  int src;

  /* Allow overdrafts */
  TM_START(0, RW);
  for (i = 0; i < SIZE_READ; i++) {
    src = RAND_R_FNC(seed) % bank->size;
    result += TM_LOAD(&(bank->accounts[src]));
  }
  TM_COMMIT;

  return result;
}

static int write_tx(bank_t *bank, int src_seed)
{
  long i;
  int result = 0;
  uint64_t seed = src_seed;
  int src;

  /* Allow overdrafts */
  TM_START(0, RW);
  for (i = 0; i < SIZE_READ; i++) {
    src = RAND_R_FNC(seed) % bank->size;
    result += TM_LOAD(&(bank->accounts[src]));
  }
  seed = src_seed;
  for (i = 0; i < SIZE_WRITE; i++) {
    src = RAND_R_FNC(seed) % bank->size;
    TM_STORE(&(bank->accounts[src]), result);
  }
  TM_COMMIT;

  return result;
}

static int transfer(account_t *src, account_t *dst, int amount)
{
  long i;

  /* Allow overdrafts */
  TM_START(0, RW);
  i = TM_LOAD(&src->balance);
  i -= amount;
  TM_STORE(&src->balance, i);
  i = TM_LOAD(&dst->balance);
  i += amount;
  TM_STORE(&dst->balance, i);
  TM_COMMIT;

  return amount;
}

static int total(bank_t *bank, int transactional)
{
  long i, total;

  if (!transactional) {
    total = 0;
    for (i = 0; i < bank->size; i++) {
      total += bank->accounts[i].balance;
    }
  } else {
    TM_START(1, RO);
    total = 0;
    for (i = 0; i < bank->size; i++) {
      total += TM_LOAD(&bank->accounts[i].balance);
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
    TM_STORE(&bank->accounts[i].balance, 0);
  }
  TM_COMMIT;
}

/* ################################################################### *
 * STRESS TEST
 * ################################################################### */

typedef struct thread_data {
  bank_t *bank;
  barrier_t *barrier;
  unsigned int seed;
  int id;
  int read_all;
  int read_threads;
  int write_all;
  int write_threads;
  int disjoint;
  int nb_threads; // number of workers

  // 1 queue per worker
  tbb::concurrent_queue<bank_request_t*> **concQueue;
  char padding[64];
} thread_data_t;

typedef struct worker_data {
  int id;
  int nb_threads; // number of workers
  bank_t *bank;
  barrier_t *barrier;
  int read_all;
  int read_threads;
  int write_all;
  int write_threads;
  int disjoint;
  unsigned long nb_transfer;
  unsigned long nb_read_all;
  unsigned long nb_write_all;
  uint64_t in_tx_cycles;
  tbb::concurrent_queue<bank_request_t*> *concQueue;
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
  char padding[64];
} worker_data_t;

static void *worker_thread(void *data)
{
  worker_data_t *d = (worker_data_t*)data;
  uint64_t ts1, ts2;
  uint64_t seed = d->id * 1234567;

  pinThisThread(d->id);

  /* Create transaction */
  TM_INIT_THREAD(0,0);

  /* Wait on barrier */
  barrier_cross(*(d->barrier));

  while (stop == 0) {
    bank_request_t *req;
    // dequeue next request
    while (!d->concQueue->try_pop(req) && !stop) /* wait */ printf("thread %i waits requests\n", d->id);
    if (stop == 1) break;

    //
    // bank_request_t req1;
    // bank_request_t *req = &req1;
    //
    // int nb, src, dst;
    // req->isDone = 0;
    // req->transactional = 1;
    // nb = (int)(RAND_R_FNC(seed) % 100);
    // if (nb < d->read_all) {
    //   /* Read all */
    //   req->req = TOTAL;
    //   src = (int)RAND_R_FNC(seed);
    //   req->src = src;
    // } else if (nb < d->read_all + d->write_all) {
    //   /* Write all */
    //   req->req = RESET;
    // } else {
    //   /* Choose random accounts */
    //   req->req = TRANSFER;
    //   src = (int)RAND_R_FNC(seed);
    //   req->src = src;
    // }



    // TODO: if req == NULL --> SIGSEGV
    switch (req->req) {
      case TRANSFER:
        ts1 = rdtscp();
        // req->result = transfer(
        //   &d->bank->accounts[req->src],
        //   &d->bank->accounts[req->dst],
        //   req->amount
        // );
        req->result = write_tx(d->bank, req->src);
        ts2 = rdtsc();
        d->in_tx_cycles += ts2 - ts1;
        d->nb_transfer++;
        break;
      case TOTAL:
        ts1 = rdtscp();
        // req->result = total(d->bank, req->transactional);
        req->result = read_tx(d->bank, req->src);
        ts2 = rdtsc();
        d->in_tx_cycles += ts2 - ts1;
        d->nb_read_all++;
        break;
      case RESET:
        ts1 = rdtscp();
        reset(d->bank);
        ts2 = rdtsc();
        d->in_tx_cycles += ts2 - ts1;
        req->result = 0;
        d->nb_write_all++;
        break;
      default: /* empty */
        break;
    }

    // __atomic_store_n(&req->isDone, 1, __ATOMIC_RELEASE);
  }

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
  TM_EXIT_THREAD();

  return NULL;
}

static void *enqueuer_thread(void *data)
{
  int src, dst, nb;
  int rand_max, rand_min;
  thread_data_t *d = (thread_data_t *)data;
  // unsigned short seed[3];
  bank_request_t *req_buffer;
  const int SIZE_REQ_BUFFER = 1048576*50;
  int req_ptr;
  int target_queue;

  target_queue = d->id;
  pinThisThread(d->id + d->nb_threads);

  /* Initialize seed (use rand48 as rand is poor) */
  // seed[0] = (unsigned short)rand_r(&d->seed);
  // seed[1] = (unsigned short)rand_r(&d->seed);
  // seed[2] = (unsigned short)rand_r(&d->seed);

  req_buffer = (bank_request_t*)malloc(SIZE_REQ_BUFFER*sizeof(bank_request_t));
  req_ptr = 0;

  /* Prepare for disjoint access */
  if (d->disjoint) {
    rand_max = d->bank->size / d->nb_threads;
    rand_min = rand_max * d->id;
    if (rand_max <= 2) {
      fprintf(stderr, "can't have disjoint account accesses");
      return NULL;
    }
  } else {
    rand_max = d->bank->size;
    rand_min = 0;
  }

  for (int i = 0; i < SIZE_REQ_BUFFER; i++) {
    // Pre-fill some requests
    bank_request_t *req = &req_buffer[(req_ptr++) % SIZE_REQ_BUFFER];

    req->isDone = 0;
    req->transactional = 1;
    if (d->id < d->read_threads) {
      /* Read all */
      req->req = TOTAL;
      src = (int)RAND_R_FNC(d->seed);
      req->src = src;
    } else if (d->id < d->read_threads + d->write_threads) {
      /* Write all */
      req->req = RESET;
    } else {
      nb = (int)(RAND_R_FNC(d->seed) % 100);
      if (nb < d->read_all) {
        /* Read all */
        req->req = TOTAL;
        src = (int)RAND_R_FNC(d->seed);
        req->src = src;
      } else if (nb < d->read_all + d->write_all) {
        /* Write all */
        req->req = RESET;
      } else {
        /* Choose random accounts */
        req->req = TRANSFER;
        src = (int)RAND_R_FNC(d->seed);
        req->src = src;
      }
    }
    d->concQueue[target_queue]->push(req);
  }

  /* Wait on barrier */
  barrier_cross(*(d->barrier));

  return NULL; // done!

  while (stop == 0) {
    bank_request_t *req = &req_buffer[(req_ptr++) % SIZE_REQ_BUFFER];
    // target_queue = (target_queue + 1) % d->nb_threads; // always the same queue
    if (req_ptr > SIZE_REQ_BUFFER) {
      // check if done
      while (!req->isDone && !stop) /* wait */;
      if (stop) break;
    }
    req->isDone = 0;
    req->transactional = 1;
    if (d->id < d->read_threads) {
      /* Read all */
      req->req = TOTAL;
      src = (int)RAND_R_FNC(d->seed);
      req->src = src;
    } else if (d->id < d->read_threads + d->write_threads) {
      /* Write all */
      req->req = RESET;
    } else {
      // nb = (int)(erand48(seed) * 100);
      nb = (int)(RAND_R_FNC(d->seed) % 100);
      if (nb < d->read_all) {
        /* Read all */
        req->req = TOTAL;
        src = (int)RAND_R_FNC(d->seed);
        req->src = src;
      } else if (nb < d->read_all + d->write_all) {
        /* Write all */
        req->req = RESET;
      } else {
        /* Choose random accounts */
        req->req = TRANSFER;
        src = (int)RAND_R_FNC(d->seed);
        req->src = src;
        // // src = (int)(erand48(seed) * rand_max) + rand_min;
        // // dst = (int)(erand48(seed) * rand_max) + rand_min;
        // src = (int)(RAND_R_FNC(d->seed) % rand_max) + rand_min;
        // dst = (int)(RAND_R_FNC(d->seed) % rand_max) + rand_min;
        // if (dst == src)
        //   dst = ((src + 1) % rand_max) + rand_min;
        // req->src = src;
        // req->dst = dst;
        // req->amount = 1;
      }
    }
    d->concQueue[target_queue]->push(req);
  }

  // free(req_buffer); // TODO: must sync with worker!!!

  return NULL;
}

static void catcher(int sig)
{
  static int nb = 0;
  printf("CAUGHT SIGNAL %d\n", sig);
  if (++nb >= 3)
    exit(1);
}

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
    {"disjoint",                  no_argument,       NULL, 'j'},
    {"injectors",                 required_argument, NULL, 'i'},
    {NULL, 0, NULL, 0}
  };

  bank_t *bank;
  int i, c, ret;
  unsigned long reads, writes, updates;
  uint64_t in_tx_cycles;
#ifndef TM_COMPILER
  char *s;
  unsigned long aborts, aborts_1, aborts_2,
    aborts_locked_read, aborts_locked_write,
    aborts_validate_read, aborts_validate_write, aborts_validate_commit,
    aborts_invalid_memory, aborts_killed,
    locked_reads_ok, locked_reads_failed, max_retries;
  stm_ab_stats_t ab_stats;
  char *cm = NULL;
#endif /* ! TM_COMPILER */
  thread_data_t *data;
  worker_data_t *wdata;
  pthread_t *threads;
  pthread_attr_t attr;
  barrier_t barrier;
  struct timeval start, end;
  struct timespec timeout;
  int duration = DEFAULT_DURATION;
  int nb_accounts = DEFAULT_NB_ACCOUNTS;
  int nb_threads = DEFAULT_NB_THREADS;
  int nb_injectors = DEFAULT_NB_THREADS;
  int read_all = DEFAULT_READ_ALL;
  int read_threads = DEFAULT_READ_THREADS;
  int seed = DEFAULT_SEED;
  int write_all = DEFAULT_WRITE_ALL;
  int write_threads = DEFAULT_WRITE_THREADS;
  int disjoint = DEFAULT_DISJOINT;
  sigset_t block_set;

  while(1) {
    i = 0;
    c = getopt_long(argc, argv, "ha:c:d:n:r:R:s:w:W:ji:", long_options, &i);

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
              "  -j, --disjoint\n"
              "        Transfers do not cause conflicts\n"
              "  -i, --injectors\n"
              "        Number of threads that enqueue requests\n"
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
     case 'i':
       nb_injectors = atoi(optarg);
       break;
     case 'j':
       disjoint = 1;
       break;
     case '?':
       printf("Use -h or --help for help\n");
       exit(0);
     default:
       exit(1);
    }
  }

  assert(duration >= 0);
  assert(nb_accounts >= 2);
  assert(nb_threads > 0);
  assert(read_all >= 0 && write_all >= 0 && read_all + write_all <= 100);
  assert(read_threads + write_threads <= nb_threads);

  printf("Nb accounts    : %d\n", nb_accounts);
#ifndef TM_COMPILER
  printf("CM             : %s\n", (cm == NULL ? "DEFAULT" : cm));
#endif /* ! TM_COMPILER */
  printf("Duration       : %d\n", duration);
  printf("Nb threads     : %d\n", nb_threads);
  printf("Read-all rate  : %d\n", read_all);
  printf("Read threads   : %d\n", read_threads);
  printf("Seed           : %d\n", seed);
  printf("Write-all rate : %d\n", write_all);
  printf("Write threads  : %d\n", write_threads);
  printf("Type sizes     : int=%d/long=%d/ptr=%d/word=%d\n",
         (int)sizeof(int),
         (int)sizeof(long),
         (int)sizeof(void *),
         (int)sizeof(size_t));

  timeout.tv_sec = duration / 1000;
  timeout.tv_nsec = (duration % 1000) * 1000000;

  if ((data = (thread_data_t *)malloc(nb_threads * sizeof(thread_data_t))) == NULL) {
    perror("malloc");
    exit(1);
  }
  if ((wdata = (worker_data_t *)malloc(nb_threads * sizeof(worker_data_t))) == NULL) {
    perror("malloc");
    exit(1);
  }
  if ((threads = (pthread_t *)malloc(nb_threads*2 * sizeof(pthread_t))) == NULL) {
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
  for (i = 0; i < bank->size; i++) {
    bank->accounts[i].number = i;
    bank->accounts[i].balance = 0;
  }

  stop = 0;

  // HeTM_init((HeTM_init_s){
  //   .policy       = HETM_GPU_INV,
  //   .nbCPUThreads = parsedData.nb_threads,
  //   .nbGPUBlocks  = parsedData.GPUblockNum,
  //   .nbGPUThreads = parsedData.GPUthreadNum,
  //   .timeBudget   = parsedData.timeBudget,
  //   .isCPUEnabled = 1,
  //   .isGPUEnabled = 0
  // });

  /* Init STM */
  printf("Initializing STM\n");
  TM_INIT(MAX_NB_THRS);

#ifndef TM_COMPILER
  if (stm_get_parameter("compile_flags", &s))
    printf("STM flags      : %s\n", s);

  if (cm != NULL) {
    if (stm_set_parameter("cm_policy", cm) == 0)
      printf("WARNING: cannot set contention manager \"%s\"\n", cm);
  }
#endif /* ! TM_COMPILER */

  /* Access set from all threads */
  barrier_init(barrier, nb_threads+nb_injectors + 1);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // tbb::concurrent_queue<bank_request_t*> sharedConcQueue;
  tbb::concurrent_queue<bank_request_t*> **sharedConcQueue;
  sharedConcQueue = (tbb::concurrent_queue<bank_request_t*>**)
    malloc(sizeof(tbb::concurrent_queue<bank_request_t*>*)*nb_threads);

  for (i = 0; i < nb_threads; i++) {
    wdata[i].nb_transfer = 0;
    wdata[i].nb_read_all = 0;
    wdata[i].nb_write_all = 0;
    wdata[i].in_tx_cycles = 0;
    wdata[i].read_all = read_all;
    wdata[i].read_threads = read_threads;
    wdata[i].write_all = write_all;
    wdata[i].write_threads = write_threads;
    wdata[i].disjoint = disjoint;
#ifndef TM_COMPILER
    wdata[i].nb_aborts = 0;
    wdata[i].nb_aborts_1 = 0;
    wdata[i].nb_aborts_2 = 0;
    wdata[i].nb_aborts_locked_read = 0;
    wdata[i].nb_aborts_locked_write = 0;
    wdata[i].nb_aborts_validate_read = 0;
    wdata[i].nb_aborts_validate_write = 0;
    wdata[i].nb_aborts_validate_commit = 0;
    wdata[i].nb_aborts_invalid_memory = 0;
    wdata[i].nb_aborts_killed = 0;
    wdata[i].locked_reads_ok = 0;
    wdata[i].locked_reads_failed = 0;
    wdata[i].max_retries = 0;
#endif /* ! TM_COMPILER */
    wdata[i].id = i;
    wdata[i].nb_threads = nb_threads;
    wdata[i].bank = bank;
    wdata[i].barrier = &barrier;
    sharedConcQueue[i] = new tbb::concurrent_queue<bank_request_t*>();
    wdata[i].concQueue = sharedConcQueue[i];
    if (pthread_create(&threads[i], &attr, worker_thread, (void *)(&wdata[i])) != 0) {
      fprintf(stderr, "Error creating thread\n");
      exit(1);
    }
  }

  for (i = 0; i < nb_injectors; i++) {
    printf("Creating thread %d\n", i);
    data[i].id = i;
    data[i].read_all = read_all;
    data[i].read_threads = read_threads;
    data[i].write_all = write_all;
    data[i].write_threads = write_threads;
    data[i].disjoint = disjoint;
    data[i].nb_threads = nb_threads;
    data[i].seed = rand();
    data[i].bank = bank;
    data[i].barrier = &barrier;
    // data[i].concQueue = &sharedConcQueue;
    data[i].concQueue = sharedConcQueue;
    if (pthread_create(&threads[i+nb_threads], &attr, enqueuer_thread, (void *)(&data[i])) != 0) {
      fprintf(stderr, "Error creating thread\n");
      exit(1);
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
  barrier_cross(barrier);

  printf("STARTING...\n");
  gettimeofday(&start, NULL);
  if (duration > 0) {
    nanosleep(&timeout, NULL);
  } else {
    sigemptyset(&block_set);
    sigsuspend(&block_set);
  }
  stop = 1;
  gettimeofday(&end, NULL);
  printf("STOPPING...\n");

  /* Wait for thread completion */
  for (i = 0; i < nb_threads+nb_injectors; i++) {
    if (pthread_join(threads[i], NULL) != 0) {
      fprintf(stderr, "Error waiting for thread completion\n");
      exit(1);
    }
  }

  duration = (end.tv_sec * 1000 + end.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);
  in_tx_cycles = 0;
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
  for (i = 0; i < nb_threads; i++) {
    printf("Thread %d\n", i);
    printf("  #transfer   : %lu\n", wdata[i].nb_transfer);
    printf("  #read-all   : %lu\n", wdata[i].nb_read_all);
    printf("  #write-all  : %lu\n", wdata[i].nb_write_all);
#ifndef TM_COMPILER
    printf("  #aborts     : %lu\n", wdata[i].nb_aborts);
    printf("    #lock-r   : %lu\n", wdata[i].nb_aborts_locked_read);
    printf("    #lock-w   : %lu\n", wdata[i].nb_aborts_locked_write);
    printf("    #val-r    : %lu\n", wdata[i].nb_aborts_validate_read);
    printf("    #val-w    : %lu\n", wdata[i].nb_aborts_validate_write);
    printf("    #val-c    : %lu\n", wdata[i].nb_aborts_validate_commit);
    printf("    #inv-mem  : %lu\n", wdata[i].nb_aborts_invalid_memory);
    printf("    #killed   : %lu\n", wdata[i].nb_aborts_killed);
    printf("  #aborts>=1  : %lu\n", wdata[i].nb_aborts_1);
    printf("  #aborts>=2  : %lu\n", wdata[i].nb_aborts_2);
    printf("  #lr-ok      : %lu\n", wdata[i].locked_reads_ok);
    printf("  #lr-failed  : %lu\n", wdata[i].locked_reads_failed);
    printf("  Max retries : %lu\n", wdata[i].max_retries);
    aborts += wdata[i].nb_aborts;
    aborts_1 += wdata[i].nb_aborts_1;
    aborts_2 += wdata[i].nb_aborts_2;
    aborts_locked_read += wdata[i].nb_aborts_locked_read;
    aborts_locked_write += wdata[i].nb_aborts_locked_write;
    aborts_validate_read += wdata[i].nb_aborts_validate_read;
    aborts_validate_write += wdata[i].nb_aborts_validate_write;
    aborts_validate_commit += wdata[i].nb_aborts_validate_commit;
    aborts_invalid_memory += wdata[i].nb_aborts_invalid_memory;
    aborts_killed += wdata[i].nb_aborts_killed;
    locked_reads_ok += wdata[i].locked_reads_ok;
    locked_reads_failed += wdata[i].locked_reads_failed;
    if (max_retries < wdata[i].max_retries)
      max_retries = wdata[i].max_retries;
#endif /* ! TM_COMPILER */
    updates += wdata[i].nb_transfer;
    reads += wdata[i].nb_read_all;
    writes += wdata[i].nb_write_all;
    in_tx_cycles += wdata[i].in_tx_cycles;
  }
  /* Sanity check */
  ret = total(bank, 0);
  printf("Bank total    : %d (expected: 0)\n", ret);
  printf("Duration      : %d (ms)\n", duration);
  printf("#txs          : %lu (%f / s)\n", reads + writes + updates, (reads + writes + updates) * 1000.0 / duration);
  printf("#read txs     : %lu (%f / s)\n", reads, reads * 1000.0 / duration);
  printf("#write txs    : %lu (%f / s)\n", writes, writes * 1000.0 / duration);
  printf("#update txs   : %lu (%f / s)\n", updates, updates * 1000.0 / duration);
#ifndef TM_COMPILER
  printf("#aborts       : %lu (%f / s)\n", aborts, aborts * 1000.0 / duration);
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

  for (i = 0; stm_get_ab_stats(i, &ab_stats) != 0; i++) {
    printf("Atomic block  : %d\n", i);
    printf("  #samples    : %lu\n", ab_stats.samples);
    printf("  Mean        : %f\n", ab_stats.mean);
    printf("  Variance    : %f\n", ab_stats.variance);
    printf("  Min         : %f\n", ab_stats.min);
    printf("  Max         : %f\n", ab_stats.max);
    printf("  50th perc.  : %f\n", ab_stats.percentile_50);
    printf("  90th perc.  : %f\n", ab_stats.percentile_90);
    printf("  95th perc.  : %f\n", ab_stats.percentile_95);
  }
#endif /* ! TM_COMPILER */

  /* Delete bank and accounts */
  free(bank->accounts);
  free(bank);

  /* Cleanup STM */
  TM_EXIT();

  free(threads);
  free(data);

  /* Output stats */
  stats_file = fopen(STATS_FILE_PATH, "a");
  fseek(stats_file, 0, SEEK_END);
  if (ftell(stats_file) < 8) {
    // file is empty, write the header
    fprintf(stats_file,
      "NB_THREADS,"         \
      "NB_TRANSFER,"        \
      "NB_TOTAL,"           \
      "NB_RESET,"           \
      "DURATION,"           \
      "DURATION_IN_TX\n"    \
    );
  }
  fprintf(stats_file,
    "%i,%li,%li,%li,%i,%f\n",
    nb_threads,
    updates,
    reads,
    writes,
    duration,
    (float)in_tx_cycles / (float)CPU_FREQ
  );

  return ret;
}
