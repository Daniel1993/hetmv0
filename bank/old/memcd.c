#define _GNU_SOURCE
#define __USE_GNU

#include "bank.h"
#include "bank_aux.h"
#include "CheckAllFlags.h"


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
* GLOBALS
* ################################################################### */

static volatile int stop;
static volatile int cuda_stop;
static volatile int cpu_enable_flag;

static volatile int cmp_flag;

// global
thread_data_t parsedData;

int transfer(account_t * accounts, int *positions, int count, int amount)
{
  long i;
  int z = 0;
  int n, src, dst;

  /* Allow overdrafts */
  TM_START(z, RW);

  for (n = 0; n < count; n += 2) {
    src = positions[n];
    dst = positions[n+1];
    i = TM_LOAD(&accounts[src]);
    i -= amount;
    TM_STORE(&accounts[src], i);

    i = TM_LOAD(&accounts[dst]);
    i += amount;
    TM_STORE(&accounts[dst], i);
  }

  TM_COMMIT;

  return amount;
}

int total(bank_t *bank, int transactional)
{
  long i;
  long total;

  if (!transactional) {
    total = 0;
    for (i = 0; i < bank->size; i++) {
      total += bank->accounts[i];
    }
  } else {
    TM_START(1, RO);
    total = 0;
    for (i = 0; i < bank->size; i++) {
      total += TM_LOAD(&bank->accounts[i]);
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
    TM_STORE(&bank->accounts[i], 0);
  }
  TM_COMMIT;
}

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

  // INIT stream
  void * readFromTiny = NULL;
	int total_el = 0;
	HeTM_CPULogNode_t * logAux;
	cmp_status state = FINAL;
	int n = 0, count = 0, res = 0;
	stream_t *s = jobWithCuda_initStream(d->cd, d->id, STREAM_COUNT);
	HeTM_CPULogEntry *logVec[STREAM_COUNT];
	for (count = 0; count < STREAM_COUNT; count++) {
    logVec[count] = (HeTM_CPULogEntry*)CudaMallocWrapper(LOG_SIZE*sizeof(HeTM_CPULogEntry));
  }
	state = READ;

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
    rand_max = nb_accounts;
    rand_min = 0;
  }

  /* Create transaction */
  TM_INIT_THREAD(d->bank->accounts, d->bank->size);
  /* Wait on barrier */
  barrier_cross(d->barrier);

  while (stop == 0 || cuda_stop!=2) {

    TEST_GPU_IN_LOOP; // TODO: refactor the macros!!!

    if(cpu_enable_flag == 1) {
  		/* Fetch values from the queue */
  		for (i = 0; i < 1; i++) {
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
  BANK_GET_STATS(d);

  /* Free transaction */
  BANK_TEARDOWN_TX();

  TM_EXIT_THREAD;

  return NULL;
}

/* ################################################################### *
* CUDA THREAD
* ################################################################### */

static void *test_cuda(void *data) {
  thread_data_t *d = (thread_data_t *)data;
  int i, n, loop = 0;
  int flag, flag_check = 0;
  long counter_aborts = 0, counter_commits = 0, counter_sync = 0; //Counters
  double duration = 0, tx_duration = 0; //Timers
  double trf_2cpu = 0, trf_2gpu = 0; //Timers
  double	duration_cmp = 0, trans_cmp = 0;
  TIMER_T tx_start, tx_end, lock_start, lock_end, cmp_start, cmp_end; //Timer structure
  HeTM_CPULogNode_t *aux;

  cuda_t *cd   = d->cd;
  account_t *a = d->bank->accounts;
  account_t *helper = NULL;
  int sync_fail = 0, sync_sucess = 0;

  #if CPU_INV == 1
  long expected_commits = EXPECTED_COMMITS_GPU(cd);
  for (i = 0; i < d->nb_threads - 1; i++) {
    d->dthreads[i].global_commits = 0;
  }
  #endif /* CPU_INV */

  int * valid_vec = NULL;
  valid_vec = (int *)malloc(d->bank->size * sizeof(int)); // BM_TRANSF

  TIMER_READ(tx_start);

  DEBUG_PRINT("Starting GPU execution.\n");
  flag = jobWithCuda_run(cd, a); // Start execution
  barrier_cross(d->barrier);     // Signal CPU threads to start

  if (flag == 0) {  // Cuda error detection
    printf("CUDA run failed.\n");
    TIMER_READ(tx_end);
    cuda_stop = 2;
  } else {
    DEBUG_PRINT("Starting transaction kernel!\n");
  }

  while (flag) {
    // ------------------------------------
    // MAIN LOOP (CUDA thread)
    // Cicle till execution concludes
    // ------------------------------------
    loop++;
    helper = NULL;

    jobWithCuda_wait(); //Wait for GPU execution to conclude

    TIMER_READ(tx_end);
    DEBUG_PRINT("Transaction kernel concluded\n");
    tx_duration += TIMER_DIFF_SECONDS(tx_start, tx_end) * 1000;

    #if CPU_INV == 1
    jobWithCuda_backup(cd);
    #endif /* CPU_INV */

    DEBUG_PRINT("All threads locked\n"); /* LOCKED */
    TIMER_READ(lock_start);
    flag_check = 0;

    BANK_CMP_KERNEL();

    //Check for errors
    if (flag_check == -1)	{
      printf("Comparison crashed. CUDA thread exiting\n");
      flag = 0;
    }

    DEBUG_PRINT("Final comparison result: %d\n", flag_check);

    #if CPU_INV == 1
    //Check if GPU needs to invalidate CPU
    if (flag_check == 1) {
      long global = 0;
      for (i = 0; i < d->nb_threads - 1; i++) {
        global += d->dthreads[i].global_commits;
      }

      // TODO:  prefer faster
      // if (global < expected_commits) {
        DEBUG_PRINT("Invalidating CPU.\n");
        flag_check = 0; // invalida CPU
        global_fix += global;
        counter_sync--;
        jobWithCuda_backupRestore(cd);
      // }
    }
    for (i = 0; i < d->nb_threads - 1; i++) {
      d->dthreads[i].global_commits = 0;
    }
    #endif /* CPU_INV */

    // Transfer back results to CPU
    TIMER_READ(cmp_start);
    // TODO[Ricardo]: the INV_CPU sets this to 0, it means success?
    // flag_check --> se 0 a comparação foi sucesso
    // flag_check --> se 1 a comparação falhou
    if (flag_check == 0) { //Successful execution
      sync_fail = 0;
      sync_sucess++;       //Sequence Counter
      counter_sync++;      //Success Counter

      if (jobWithCuda_hupd(cd, a, valid_vec) != 1) { // transfere o WSet do GPU para o CPU
        printf("BUG\n");
        flag = 0;
      }
    } else {                           //Unsuccessful execution
      sync_fail++;
      sync_sucess = 0;                 //Sequence Counter
      helper = jobWithCuda_swap(cd);  //Update device data for re-rerun
    }
    TIMER_READ(cmp_end);

    //Check if we are re-running
    if (stop == 0 && flag_check !=-1) {
      cuda_stop = 0;
    } else {
      //Times up
      cuda_stop = 2;
      flag = 0;
    }

    cmp_flag = 0;
    TIMER_READ(lock_end);
    barrier_cross(d->cuda_barrier);  // TODO: check how this works

    DEBUG_PRINT("All threads unlocked\n");

    //Rerun transaction kernel
    if (cuda_stop != 2) {
      if (flag_check == 0) {
        // TODO[Ricardo]: Success?
        jobWithCuda_getStats(*cd, &counter_aborts, &counter_commits); // Update stats
      }

      // CMP_APPLY
      if (jobWithCuda_run(cd, helper)) {
        DEBUG_PRINT("Running new transaction kernel!\n");
      }

      TIMER_T(tx_start);
    }

    //Clean up the memory
    do {
      n = 0;
      for (i = 0; i < d->nb_threads - 1; i++) {
        if (d->dthreads[i].HeTM_CPULog == NULL) {
          n++;
          continue;
        }
        aux = d->dthreads[i].HeTM_CPULog;
        d->dthreads[i].HeTM_CPULog = d->dthreads[i].HeTM_CPULog->next;
        TM_FREE(aux);
        //printf("Free from thread %d\n", i);
      }
    } while (n != d->nb_threads - 1); //Reached the end of every list;

    duration += TIMER_DIFF_SECONDS(lock_start, lock_end) * 1000;
    trf_2cpu += TIMER_DIFF_SECONDS(cmp_start, cmp_end) * 1000;
    // ------------------------------------
    // MAIN LOOP (end)
    // ------------------------------------
  }

  printf("CUDA thread terminated after %d(%d successful) run(s). \nTotal cuda execution time: %f ms.\n", loop, (int)counter_sync, duration);
  if (flag_check == 0) // TODO: flag_check is 0 I need to know all flags
  jobWithCuda_getStats(*cd, &counter_aborts, &counter_commits); // counter commits has all commits

  // TODO:
  d->nb_transfer = global_fix == 0 ? counter_sync * (counter_commits/loop) : counter_commits;
  d->nb_aborts                 = counter_aborts;
  d->nb_aborts_1               = duration;
  d->nb_aborts_2               = duration_cmp;
  d->nb_aborts_locked_read     = loop;
  d->nb_aborts_locked_write    = counter_sync;
  d->nb_aborts_validate_read   = trf_2cpu;
  d->nb_aborts_validate_write  = trf_2gpu;
  d->nb_aborts_validate_commit = trans_cmp;
  d->nb_aborts_invalid_memory  = 0;
  d->nb_aborts_killed          = 0;
  d->locked_reads_ok           = 0;
  d->locked_reads_failed       = 0;
  d->max_retries               = tx_duration;

  DEBUG_PRINT("CUDA thread exiting.\n");
  // TODO: check free(a)
  return NULL;
}

/* ################################################################### *
*
* MAIN
*
* ################################################################### */

int main(int argc, char **argv)
{
  bank_t *bank;
  barrier_t cuda_barrier;
  int i, j, ret = -1;
  thread_data_t *data;
  pthread_t *threads;
  barrier_t barrier;
  sigset_t block_set;
  char filename[512];

  memset(&parsedData, 0, sizeof(thread_data_t));

  parsedData.filename = filename;

  PRINT_FLAGS();

  // ##########################################
  // ### Input management
  bank_parseArgs(argc, argv, &parsedData);

  // TODO: these barriers are a CANCER!!!!
#if CPUEn == 0
  // disable CPU usage
  parsedData.nb_threadsCPU = 0;
  parsedData.nb_threads = 1;
#else
  parsedData.nb_threadsCPU = parsedData.nb_threads;
  parsedData.nb_threads++;
#endif

  barrier_init(&cuda_barrier, parsedData.nb_threads);
  barrier_init(&barrier, parsedData.nb_threads + 1); // also waits for the main thread?

  malloc_or_die(data, parsedData.nb_threads);
  parsedData.dthreads     = data;
  parsedData.barrier      = &barrier;
  parsedData.cuda_barrier = &cuda_barrier;
  // ##########################################

  jobWithCuda_exit(NULL); // Reset Cuda Device

  bank_check_params(&parsedData);

  malloc_or_die(threads, parsedData.nb_threads);

  malloc_or_die(bank, 1);
  malloc_or_die(bank->accounts, parsedData.nb_accounts);
  bank->size = parsedData.nb_accounts;
  memset(bank->accounts, 0, parsedData.nb_accounts * sizeof(account_t));
  printf("Total before   : %d\n", total(bank, 0));
  parsedData.bank = bank;

  DEBUG_PRINT("Initializing GPU.\n");

  cuda_t *cuda_st;
  cuda_st = jobWithCuda_init(bank->accounts, bank->size, parsedData.trans, 0,
    parsedData.GPUthreadNum, parsedData.GPUblockNum);
  parsedData.cd = cuda_st;
  //DEBUG_PRINT("Base: %lu %lu \n", bank->accounts, &bank->accounts);
  if (cuda_st == NULL) {
    printf("CUDA init failed.\n");
    exit(-1);
  }

  /* Init STM */
  printf("Initializing STM\n");

  TM_INIT(parsedData.nb_threads);
  // ############################################################################
  // ### Start iterations #######################################################
  // ############################################################################
  for(j = 0; j < parsedData.iter; j++) { // Loop for testing purposes
    //Clear flags
    cuda_stop = stop = 0;
    cpu_enable_flag = 1;
    global_fix = 0;

    #ifndef TM_COMPILER
    if (parsedData.cm != NULL) {
      if (stm_set_parameter("cm_policy", parsedData.cm) == 0)
      printf("WARNING: cannot set contention manager \"%s\"\n", parsedData.cm);
    }
    #endif /* ! TM_COMPILER */

    // ##############################################
    // ### create threads
    // ##############################################
    for (i = 0; i < parsedData.nb_threads; i++) {
      /*SET CPU AFFINITY*/
      /*INIT DATA STRUCTURE*/
      printf("Creating thread %d\n", i);
      // remove last iter status
      parsedData.reads = parsedData.writes = parsedData.updates = 0;
      parsedData.nb_aborts = 0;
      parsedData.nb_aborts_2 = 0;
      memcpy(&data[i], &parsedData, sizeof(thread_data_t));
      data[i].id      = i;
      data[i].seed    = rand();
      data[i].cd      = cuda_st;
      if(i == parsedData.nb_threadsCPU) {
        thread_create_or_die(&threads[i], NULL, test_cuda, &data[i]);
      } else {
        data[i].HeTM_CPULog = NULL;
        thread_create_or_die(&threads[i], NULL, test, &data[i]);
      }
    }
    // ### end create threads

    /* Start threads */
    barrier_cross(&barrier);

    printf("STARTING...(Run %d)\n", j);

    TIMER_READ(parsedData.start);

    if (parsedData.duration > 0) {
      nanosleep(&parsedData.timeout, NULL);
    } else {
      sigemptyset(&block_set);
      sigsuspend(&block_set);
    }
    stop = 1;

    TIMER_READ(parsedData.end);
    printf("STOPPING...\n");

    /* Wait for thread completion */
    for (i = 0; i < parsedData.nb_threads; i++) {
      thread_join_or_die(threads[i], NULL);
    }

    // reset accounts
    memset(bank->accounts, 0, bank->size * sizeof(account_t));

    TIMER_READ(parsedData.final);

    bank_printStats(&parsedData);

    bank_between_iter(&parsedData, j);
  }

  //////////////////////////////////////////////////////////////////////////////
  // TODO: move this elsewhere
  DEBUG_KERNEL();
  //////////////////////////////////////////////////////////////////////////////

  //Clean up the bank
  for (i = 0; i < bank->size; i++) {
    bank->accounts[i] = 0;
  }

  /* Cleanup STM */
  TM_EXIT;
  /*Cleanup GPU*/
  jobWithCuda_exit(cuda_st);
  free(cuda_st);
  // ### End iterations #########################################################

  bank_statsFile(&parsedData);

  /* Delete bank and accounts */
  free(bank);

  free(threads);
  free(data);

  return EXIT_SUCCESS;
}
