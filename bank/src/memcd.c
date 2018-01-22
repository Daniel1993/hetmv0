#define _GNU_SOURCE
#define __USE_GNU

#include "bank.h"
#include "bank_aux.h"
#include "CheckAllFlags.h"

/* ################################################################### *
* GLOBALS
* ################################################################### */

// global
thread_data_t parsedData;

static volatile int global_ts;

packet_t search_entry(memcd_t *b, unsigned int *positions)
{
  int z = 0;
  int n, base, goal;
  int val, vers;
  packet_t response;

  /* Allow overdrafts */
  TM_START(z, RW);

  base = positions[0]*b->ways;
  goal = (positions[0] & 0xffff) % b->ways;

  for (n = 0; n < goal; n++) {
	  val  = TM_LOAD(&b->accounts[base+n]);
	  vers = TM_LOAD(&b->version[base+n]);

	  response.key  = val;
	  response.vers = vers;
  }
  n -= 1;
  TM_STORE(&b->version[base+n], global_ts);

  TM_COMMIT;

  int pos = (base + n) >> BM_HASH;
  b->version_bm[pos] = 1;

  return response;
}

packet_t new_entry(memcd_t *b, unsigned int *positions)
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

static void test(int id, void *data)
{
  int src, dst, nb;
  int accounts_vec[MAX_ITER], i;
  thread_data_t *d = &((thread_data_t *)data)[id];
  int nb_accounts = d->memcd->size;
  unsigned short seed[3];
  int queue_size = 0, queue_current = 0;
  unsigned int *queue_pointer = NULL;

  bindThread(d->id);

  // INIT stream
	int n = 0, count = 0;
	stream_t *s = jobWithCuda_initStream(d->cd, d->id, STREAM_COUNT);

  /* Initialize seed (use rand48 as rand is poor) */
  seed[0] = (unsigned short)rand_r(&d->seed);
  seed[1] = (unsigned short)rand_r(&d->seed);
  seed[2] = (unsigned short)rand_r(&d->seed);

  /* Create transaction */
  TM_INIT_THREAD(d->memcd->accounts, d->memcd->size);

  while (!HeTM_is_stop() || HeTM_get_GPU_status() != HETM_IS_EXIT) {

    TEST_GPU_IN_LOOP; // TODO: refactor the macros!!!

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
			search_entry(d->memcd, accounts_vec);
			d->nb_read_all++;
		} else {
			/* Write (SET) */
			new_entry(d->memcd, accounts_vec);
			d->nb_write_all++;
		}
  }

  DEBUG_PRINT("Thread %d exiting!\n", d->id);
  BANK_GET_STATS(d);

  /* Free transaction */
  BANK_TEARDOWN_TX();

  TM_EXIT_THREAD;
}

/* ################################################################### *
* CUDA THREAD
* ################################################################### */

static void test_cuda(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];
  int i, n, loop = 0;
  int flag;
  long counter_aborts = 0, counter_commits = 0, counter_sync = 0; //Counters
  double duration = 0, tx_duration = 0; //Timers
  double trf_2cpu = 0, trf_2gpu = 0; //Timers
  double	duration_cmp = 0, trans_cmp = 0;
  TIMER_T tx_start, tx_end, lock_start, lock_end, cmp_start, cmp_end; //Timer structure
  HeTM_CPULogNode_t *aux;

  cuda_t *cd = d->cd;
  account_t *accounts = d->memcd->accounts;
  account_t *helper = NULL;
  int sync_fail = 0, sync_sucess = 0;

#if CPU_INV == 1
  long expected_commits = EXPECTED_COMMITS_GPU(cd);
  for (i = 0; i < d->nb_threads - 1; i++) {
    d->dthreads[i].global_commits = 0;
  }
#endif /* CPU_INV */

  int * valid_vec = NULL;
  valid_vec = (int*)malloc(d->memcd->size * sizeof(int)); // BM_TRANSF

  TIMER_READ(tx_start);

  DEBUG_PRINT("Starting GPU execution.\n");
  flag = jobWithCuda_run(cd, accounts); // Start execution

  if (flag == 0) {  // Cuda error detection
    printf("CUDA run failed.\n");
    TIMER_READ(tx_end);
    HeTM_set_GPU_status(HETM_IS_EXIT);
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

    BANK_CMP_KERNEL(); // waits while the comparison is running

    //Check for errors
    if (HeTM_is_interconflict() == -1)	{
      printf("Comparison crashed. CUDA thread exiting\n");
      flag = 0;
    }

    DEBUG_PRINT("Final comparison result: %d\n", HeTM_is_interconflict());

    #if CPU_INV == 1
    //Check if GPU needs to invalidate CPU
    if (HeTM_is_interconflict() == 1) {
      long global = 0;
      for (i = 0; i < d->nb_threads - 1; i++) {
        global += d->dthreads[i].global_commits;
      }

      // TODO:  prefer faster
      // if (global < expected_commits) {
        DEBUG_PRINT("Invalidating CPU.\n");
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
    if (HeTM_is_interconflict() == 0) { // Successful execution
      sync_fail = 0;
      sync_sucess++;    // Sequence Counter
      counter_sync++;   // Success Counter

      // transfers the GPU WSet to CPU
      // TODO: bitmap is not in use (motivation use bitmap to reduce the copyback size)
      if (jobWithCuda_hupd(cd, accounts, NULL) != 1) {
        printf("BUG\n");
        flag = 0;
      }
#if HETM_CPU_EN == 0
      // copy twice, i.e., H->D and D->H
      jobWithCuda_cpyDatasetToGPU(cd, accounts); // kind of unfair
#endif
    } else {               // Unsuccessful execution
      sync_fail++;
      sync_sucess = 0;                 // Sequence Counter
      helper = jobWithCuda_swap(cd);  // Update device data for re-rerun
    }
    TIMER_READ(cmp_end);

    //Check if we are re-running
    if (!HeTM_is_stop() && HeTM_is_interconflict() != -1) {
      HeTM_set_GPU_status(HETM_BATCH_RUN);
    } else {
      //Times up
      HeTM_set_GPU_status(HETM_IS_EXIT);
      flag = 0;
    }

    HeTM_is_interconflict(0);
    TIMER_READ(lock_end);

    jobWithCuda_resetGPUSTMState(cd); // flags/locks
    HeTM_GPU_wait(); // wakes the threads, GPU will re-run

    DEBUG_PRINT("All threads unlocked\n");

    //Rerun transaction kernel
    if (HeTM_get_GPU_status() != HETM_IS_EXIT) {
      // CMP_APPLY
      if (jobWithCuda_run(cd, helper)) {
        DEBUG_PRINT("Running new transaction kernel!\n");
      }

      TIMER_T(tx_start);
    }

    //Clean up the memory
    do {
      n = 0;
      // TODO: double free
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
  jobWithCuda_getStats(cd, &counter_aborts, &counter_commits); // counter commits has all commits

  // TODO:
  d->nb_transfer = global_fix == 0 ? counter_sync * (counter_commits/loop) : counter_commits;
  d->nb_transfer_gpu_only      = counter_commits;
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
  // TODO: check free(accounts)
  // return NULL;
}

/* ################################################################### *
*
* MAIN
*
* ################################################################### */

int main(int argc, char **argv)
{
  memcd_t *memcd;
  // barrier_t cuda_barrier;
  int i, j, ret = -1;
  thread_data_t *data;
  pthread_t *threads;
  barrier_t barrier;
  sigset_t block_set;

  memset(&parsedData, 0, sizeof(thread_data_t));

  PRINT_FLAGS();

  // ##########################################
  // ### Input management
  bank_parseArgs(argc, argv, &parsedData);

  HeTM_init(parsedData.nb_threads, parsedData.GPUblockNum, parsedData.GPUthreadNum);

  // TODO:
  parsedData.nb_threadsCPU = HeTM_shared_data.nbCPUThreads;
  parsedData.nb_threads    = HeTM_shared_data.nbThreads;

  malloc_or_die(data, parsedData.nb_threads + 1);
  memset(data, 0, (parsedData.nb_threads + 1)*sizeof(thread_data_t)); // safer
  parsedData.dthreads     = data;
  // ##########################################

  jobWithCuda_exit(NULL); // Reset Cuda Device

  bank_check_params(&parsedData);

  malloc_or_die(threads, parsedData.nb_threads);

  malloc_or_die(memcd, 1);
  malloc_or_die(memcd->accounts, parsedData.nb_accounts);
  memcd->size = parsedData.nb_accounts;
  memset(memcd->accounts, 0, parsedData.nb_accounts * sizeof(account_t));
  // printf("Total before   : %d\n", total(memcd, 0)); // TODO
  parsedData.memcd = memcd;

  DEBUG_PRINT("Initializing GPU.\n");

  cuda_t *cuda_st;
  cuda_st = jobWithCuda_init(memcd->accounts, memcd->size, parsedData.trans, 0,
    parsedData.GPUthreadNum, parsedData.GPUblockNum);

  jobWithCuda_initMemcd(cuda_st, parsedData.num_ways,
    parsedData.set_percent, parsedData.shared_percent);
  memcd->ways = parsedData.num_ways;

  parsedData.cd = cuda_st;
  //DEBUG_PRINT("Base: %lu %lu \n", memcd->accounts, &memcd->accounts);
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
    HeTM_set_is_stop(0);
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
      /* SET CPU AFFINITY */
      /* INIT DATA STRUCTURE */
      printf("Creating thread %d\n", i);
      // remove last iter status
      parsedData.reads = parsedData.writes = parsedData.updates = 0;
      parsedData.nb_aborts = 0;
      parsedData.nb_aborts_2 = 0;
      memcpy(&data[i], &parsedData, sizeof(thread_data_t));
      data[i].id      = i;
      data[i].seed    = rand();
      data[i].cd      = cuda_st;
    }
    // ### end create threads

    /* Start threads */
    HeTM_start(test, test_cuda, data);

    printf("STARTING...(Run %d)\n", j);

    TIMER_READ(parsedData.start);

    if (parsedData.duration > 0) {
      nanosleep(&parsedData.timeout, NULL);
    } else {
      sigemptyset(&block_set);
      sigsuspend(&block_set);
    }
    HeTM_set_is_stop(1);

    TIMER_READ(parsedData.end);
    printf("STOPPING...\n");

    /* Wait for thread completion */
    HeTM_join_CPU_threads();

    // reset accounts
    memset(memcd->accounts, 0, memcd->size * sizeof(account_t));

    TIMER_READ(parsedData.final);
    bank_printStats(&parsedData);
    bank_between_iter(&parsedData, j);
  }

  //////////////////////////////////////////////////////////////////////////////
  // TODO: move this elsewhere
  DEBUG_KERNEL();
  //////////////////////////////////////////////////////////////////////////////

  //Clean up the memcd
  for (i = 0; i < memcd->size; i++) {
    memcd->accounts[i] = 0;
  }

  /* Cleanup STM */
  TM_EXIT;
  /*Cleanup GPU*/
  jobWithCuda_exit(cuda_st);
  free(cuda_st);
  HeTM_destroy();
  // ### End iterations #########################################################

  bank_statsFile(&parsedData);

  /* Delete memcd and accounts */
  free(memcd);

  free(threads);
  free(data);

  return EXIT_SUCCESS;
}
