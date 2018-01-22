#include "bank.h"
#include "bank_aux.h"

// -----------------------------------------------------------------------------
static thread_data_t *threadData;
static thread_data_t parsedData;
static barrier_t barrier, cuda_barrier;
static cuda_t *cuda_state;
static int isExit  = 0; /* set to one to kill the thread loop */
static int isGPUEn = 0; /* if set to 1, then last thread does GPU sync */

(void*)(*CPU_worker_callback)(int);
(void*)(*GPU_worker_callback)(int);

// -----------------------------------------------------------------------------
// local functions
static void threadWait();

// -----------------------------------------------------------------------------
// header implementation

int bank_init_CPU_worker(void *args)
{

}

void bank_do_CPU(int id)
{

}

void *bank_destroy_CPU_worker()
{

}

int bank_init_GPU_worker(void *args)
{

}

void bank_do_GPU(int id)
{
}

void *bank_destroy_GPU_worker()
{

}

void bank_process_args(int argc, char **argv)
{
  bank_parseArgs(argc, argv, &parsedData);
}

void bank_init_main()
{
  parsedData.nb_threadsCPU = parsedData.nb_threads;

#if GPUEn == 1
  isGPUEn = 1;
  parsedData.nb_threads++;
  barrier_init(&cuda_barrier, parsedData.nb_threads);
#endif /* GPUEn */
  barrier_init(&barrier, parsedData.nb_threads + 1);

  // per thread arguments
	malloc_or_die(bank_threads_args, parsedData.nb_threads);
	parsedData.dthreads     = bank_threads_args;
	parsedData.cuda_barrier = &cuda_barrier;
	parsedData.barrier      = &barrier;
  bank_check_params(&parsedData);

  jobWithCuda_exit(NULL); // Reset Cuda Device

	malloc_or_die(threads, parsedData.nb_threads);

  // TODO: bank init
  malloc_or_die(bank, 1);
  malloc_or_die(bank->accounts, parsedData.nb_accounts);
  bank->size = parsedData.nb_accounts;
  memset(bank->accounts, 0, parsedData.nb_accounts * sizeof(account_t));
  printf("Total before   : %d\n", total(bank, 0));
  parsedData.bank = bank;

  cuda_state = jobWithCuda_init(bank->accounts, bank->size, parsedData.trans, 0, 0, 0);
  parsedData.cd = cuda_state;
}

void bank_init_threading()
{
  int i;

  for (i = 0; i < parsedData.nb_threads; i++) {
    DEBUG_PRINT("Creating thread %d\n", i);
    memcpy(&threadData[i], parsedData, sizeof(thread_data_t));
    threadData[i].id = i;
    threadData[i].seed = rand();
  #if GPUEn == 1
    if(i == parsedData.nb_threadsCPU) { // last thread
      thread_create_or_die(&threads[i], NULL, threadWait, (void*)i);
    } else {
      HeTM_CPULog[i] = NULL;
      thread_create_or_die(&threads[i], NULL, threadWait, (void*)i);
    }
  #else /* GPUEn != 1 */
    thread_create_or_die(&threads[i], &attr, test, &data[i]);
  #endif /* GPUEn */
  }
}

void bank_CPU_worker((void*)(*CPU_worker)(void*))
{
  CPU_worker_callback = CPU_worker;
}

void bank_GPU_worker((void*)(*GPU_worker)(void*))
{
  GPU_worker_callback = GPU_worker;
}

void bank_do_bench()
{

}

void bank_check_iter()
{

}

void bank_print_stats()
{

}

int bank_destroy_main()
{

}

thread_data_t *bank_get_parsed_data() { return &parsedData; }

// -----------------------------------------------------------------------------
// local functions implementation

static void threadWait (void* argPtr)
{
  int threadId = *(int*)argPtr;

  bindThread(threadId);

  // think twice before changing anything!!!

  do {
    barrier_cross(barrier); /* wait for start parallel */
    if (isExit) {
      break; // after shutdown thread 0 call this
    }
    if (threadId == parsedData.nb_threadsCPU) {
      // GPU thread
      while (GPU_worker_callback == NULL) {
        pthread_yield(); // to block or to execute NULL ptr function
      }

      barrier_cross(barrier); /* wait for start parallel */
      GPU_worker_callback(threadId);
      barrier_cross(barrier); /* wait for end parallel */
    } else {
      // normal CPU worker
      while (CPU_worker_callback == NULL) {
        pthread_yield(); // to block or to execute NULL ptr function
      }

      barrier_cross(barrier); /* wait for start parallel */
      CPU_worker_callback(threadId);
      barrier_cross(barrier); /* wait for end parallel */
    }
  } while (1);
}
