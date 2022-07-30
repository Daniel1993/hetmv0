#include "hetm.cuh"
#include "hetm-cmp-kernels.cuh"
#include "hetm-producer-consumer.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "pr-stm-wrapper.cuh" // TODO: check if linkage breaks
#include "knlman.h"

#ifndef SYS_CPU_NB_CORES
#define SYS_CPU_NB_CORES 56
#endif

#define thread_create_or_die(thr, attr, callback, arg) \
  if (pthread_create(thr, attr, callback, (void *)(arg)) != 0) { \
    fprintf(stderr, "Error creating thread at " __FILE__ ":%i\n", __LINE__); \
    exit(EXIT_FAILURE); \
  } \
//

#define thread_join_or_die(thr, res) \
  if (pthread_join(thr, res)) { \
    fprintf(stderr, "Error joining thread at " __FILE__ ":%i\n", __LINE__); \
    exit(EXIT_FAILURE); \
  } \
//

// -------------------- // TODO: organize them
// Functions
static int pinThread(int tid);
static void* threadWait(void* argPtr);
static void* offloadThread(void*);
static void emptyRequest(void*);
// --------------------

// --------------------
// Variables
__thread HeTM_thread_s *HeTM_thread_data;
static barrier_t wait_callback;
static int isCreated = 0;
static HeTM_map_tid_to_core threadMapFunction = pinThread;
// --------------------

int HeTM_set_thread_mapping_fn(HeTM_map_tid_to_core fn)
{
  threadMapFunction = fn;
  return 0;
}

int HeTM_start(HeTM_callback CPUclbk, HeTM_callback GPUclbk, void *args)
{
  HeTM_set_is_stop(0);
  // Inits threading sync barrier
  if (!HeTM_shared_data.isCPUEnabled) {
    barrier_init(wait_callback, 2); // GPU on
  } else if (!HeTM_shared_data.isGPUEnabled) {
    barrier_init(wait_callback, HeTM_shared_data.nbCPUThreads+1); // CPU on
  } else {
    barrier_init(wait_callback, HeTM_shared_data.nbCPUThreads+2); // both on
  }

  int i;
  for (i = 0; i < HeTM_shared_data.nbThreads; i++) {
    if(i == HeTM_shared_data.nbCPUThreads) {
      // last thread is the GPU
      HeTM_shared_data.threadsInfo[i].callback = GPUclbk;
    } else {
      HeTM_shared_data.threadsInfo[i].callback = CPUclbk;
    }
    HeTM_shared_data.threadsInfo[i].id = i;
    HeTM_shared_data.threadsInfo[i].args = args;
    HeTM_shared_data.threadsInfo[i].isFirstChunk = 1;
    __sync_synchronize(); // makes sure threads see the new callback
    if (!isCreated) {
      thread_create_or_die(&HeTM_shared_data.threadsInfo[i].thread,
        NULL, threadWait, &HeTM_shared_data.threadsInfo[i]);
    }
  }

  if (!isCreated) {
    thread_create_or_die(&HeTM_shared_data.asyncThread,
      NULL, offloadThread, NULL);
  }

  HETM_DEB_THREADING("Signal threads to start");
  barrier_cross(wait_callback); // all start sync
  isCreated = 1;
  return 0;
}

int HeTM_join_CPU_threads()
{
  // WARNING: HeTM_set_is_stop(1) must be called before this point
  HeTM_flush_barrier();
  barrier_cross(wait_callback);
  int i;
  for (i = 0; i < HeTM_shared_data.nbThreads; i++) {
    HETM_DEB_THREADING("Joining with thread %i ...", i);
    HeTM_flush_barrier();
    thread_join_or_die(HeTM_shared_data.threadsInfo[i].thread, NULL);
  }
  barrier_destroy(wait_callback);
  HeTM_async_set_is_stop(1); // TODO: do we want to stop the thread here?
  HETM_DEB_THREADING("Joining with offload thread ...");
  HeTM_async_request((HeTM_async_req_s){ // unblocks the poor guy
    .args = NULL,
    .fn = emptyRequest
  });
  thread_join_or_die(HeTM_shared_data.asyncThread, NULL);
  isCreated = 0;
  return 0;
}

static int pinThread(int tid)
{
  int coreId = tid;

  // TODO: this is hardware dependent
  // intel14
  if (tid >= 14 && tid < 28)
    coreId += 14;
  if (tid >= 28 && tid < 42)
    coreId -= 14;

  coreId = tid % SYS_CPU_NB_CORES;
  return coreId;
}

static void emptyRequest(void*) { }

// Producer-Consumer thread: consumes requests from the other ones
static void* offloadThread(void*)
{
  HeTM_async_req_s *req;

  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  int core = threadMapFunction(55); // last thread
  CPU_SET(core, &my_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &my_set);

  while (!HeTM_async_is_stop() || !hetm_pc_is_empty(HeTM_offload_pc)) {
    hetm_pc_consume(HeTM_offload_pc, (void**)&req);
    req->fn(req->args);
    HeTM_free_async_request(req);
  }
  HETM_DEB_THREADING("Offload thread exit");
  return NULL;
}

static void* threadWait(void *argPtr)
{
  HeTM_thread_s *args = (HeTM_thread_s*)argPtr;
  int threadId = args->id;
  int idGPU = HeTM_shared_data.nbCPUThreads; // GPU is last
  HeTM_callback callback = NULL;
  int core = -1;
  cpu_set_t my_set;

  CPU_ZERO(&my_set);
  HeTM_thread_data = args;
  // TODO: each thread runs on one core only (more complex schedules not allowed)
  if (threadId != idGPU) { // only pin STM threads
    core = threadMapFunction(threadId);
    CPU_SET(core, &my_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
  }

  if (threadId == idGPU) {
    core = threadMapFunction(54); // last thread
    CPU_SET(core, &my_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
  }

  HETM_DEB_THREADING("Thread %i started on core %i", threadId, core);
  HETM_DEB_THREADING("Thread %i wait start barrier", threadId);
  barrier_cross(wait_callback); /* wait for start parallel */
	while (1) {
		while (callback == NULL) {
      // COMPILER_FENCE();
      __sync_synchronize();
      callback = args->callback;
      if (HeTM_is_stop()) break;
      // pthread_yield();
    }
    if (HeTM_is_stop()) break;

    // Runs the corresponding thread type (worker or controller)
    if (threadId == idGPU) {
      HeTM_gpu_thread(); // TODO
    } else {
      HeTM_cpu_thread(); // TODO
    }
    // callback(threadId, args->args);
    args->callback = callback = NULL;
	}
  HETM_DEB_THREADING("Thread %i wait join barrier", threadId);
  barrier_cross(wait_callback); /* join threads also call the barrier */
  HETM_DEB_THREADING("Thread %i exit", threadId);

  return NULL;
}
