#include "hetm.cuh"

HeTM_s HeTM_shared_data; // global
static barrier_t wait_callback;
static int isCreated = 0;

static void pinThread(long threadId);
static void* threadWait(void* argPtr);

int HeTM_init(int nbCPUThreads, int nbGPUBlocks, int nbGPUThreads)
{
  HeTM_shared_data.isCPUEnabled = 1;
  HeTM_shared_data.isGPUEnabled = 1;
  HeTM_shared_data.nbCPUThreads = nbCPUThreads;
  HeTM_shared_data.nbGPUBlocks  = nbGPUBlocks;
  HeTM_shared_data.nbGPUThreads = nbGPUThreads;
  HeTM_set_GPU_status(HETM_BATCH_RUN);
#if HETM_CPU_EN == 0
  // disable CPU usage
  HeTM_shared_data.nbCPUThreads = 0;
  HeTM_shared_data.isCPUEnabled = 0;
  HeTM_shared_data.nbThreads    = 1;
  barrier_init(HeTM_shared_data.GPUBarrier, 1); // only the controller thread
  barrier_init(wait_callback, 2);
#elif HETM_GPU_EN == 0
  // disable GPU usage
  HeTM_set_GPU_status(HETM_IS_EXIT);
  HeTM_shared_data.isGPUEnabled = 0;
  HeTM_shared_data.nbThreads    = nbCPUThreads;
  barrier_init(HeTM_shared_data.GPUBarrier, nbCPUThreads); // 0 controller
  barrier_init(wait_callback, nbCPUThreads+1);
#else
  // both on
  HeTM_shared_data.nbThreads    = nbCPUThreads + 1;
  barrier_init(HeTM_shared_data.GPUBarrier, nbCPUThreads+1); // 1 controller
  barrier_init(wait_callback, nbCPUThreads+2);
#endif

  malloc_or_die(HeTM_shared_data.threadsInfo, HeTM_shared_data.nbThreads);

  return 0;
}

int HeTM_destroy()
{
  barrier_destroy(HeTM_shared_data.GPUBarrier);
  barrier_destroy(wait_callback);
  free(HeTM_shared_data.threadsInfo);
  HeTM_set_is_stop(1);
  return 0;
}

int HeTM_start(HeTM_callback CPUclbk, HeTM_callback GPUclbk, void *args)
{
  for (int i = 0; i < HeTM_shared_data.nbThreads; i++) {
    if(i == HeTM_shared_data.nbCPUThreads) {
      // last thread is the GPU
      HeTM_shared_data.threadsInfo[i].callback = GPUclbk;
    } else {
      HeTM_shared_data.threadsInfo[i].callback = CPUclbk;
    }
    HeTM_shared_data.threadsInfo[i].id = i;
    HeTM_shared_data.threadsInfo[i].args = args;
    __sync_synchronize(); // makes sure threads see the new callback
    if (!isCreated) {
      thread_create_or_die(&HeTM_shared_data.threadsInfo[i].thread,
        NULL, threadWait, &HeTM_shared_data.threadsInfo[i]);
    }
  }

  barrier_cross(wait_callback); // all start sync
  isCreated = 1;
  return 0;
}

int HeTM_join_CPU_threads()
{
  // WARNING: HeTM_set_is_stop(1) must be called before this point
  barrier_cross(wait_callback);
  for (int i = 0; i < HeTM_shared_data.nbThreads; i++) {
    thread_join_or_die(HeTM_shared_data.threadsInfo[i].thread, NULL);
  }
  return 0;
}

void HeTM_set_is_stop(int isStop)
{
  HeTM_shared_data.stopFlag = isStop;
  __sync_synchronize();
}

int HeTM_is_stop()
{
  return HeTM_shared_data.stopFlag;
}

void HeTM_set_is_interconflict(int isInterconflict)
{
  HeTM_shared_data.isInterconflict = isInterconflict;
  __sync_synchronize();
}

int HeTM_is_interconflict()
{
  return HeTM_shared_data.isInterconflict;
}

void HeTM_set_GPU_status(HETM_GPU_STATE status)
{
  HeTM_shared_data.statusGPU = status;
  __sync_synchronize();
}

HETM_GPU_STATE HeTM_get_GPU_status()
{
  return HeTM_shared_data.statusGPU;
}

int HeTM_GPU_wait()
{
  barrier_cross(HeTM_shared_data.GPUBarrier);
  return 0;
}

static void pinThread(long threadId)
{
  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  int offset = threadId;
  int id = threadId;

  // intel14
  offset = id % 56;
  if (id >= 14 && id < 28)
    offset += 14;
  if (id >= 28 && id < 42)
    offset -= 14;

  CPU_SET(offset, &my_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
}

static void* threadWait(void* argPtr)
{
  thread_args_s *args = (thread_args_s*)argPtr;
  int threadId = args->id;
  HeTM_callback callback = NULL;
  void *clbkArgs = NULL;

  pinThread(threadId);
	while (1) {
		barrier_cross(wait_callback); /* wait for start parallel */
		while (callback == NULL) {
      callback = args->callback;
      clbkArgs = args->args;
      if (HeTM_is_stop()) break;
      pthread_yield();
    }
    if (HeTM_is_stop()) break;
    callback(threadId, clbkArgs);
    args->callback = callback = NULL;
	};

  return NULL;
}
