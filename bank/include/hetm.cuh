#ifndef HETM_H_GUARD_
#define HETM_H_GUARD_

#include "shared.h"

#ifndef HETM_GPU_EN
// Set to 0 to ignore GPU
#define HETM_GPU_EN 1
#endif

#ifndef HETM_CPU_EN
// Set to o to ignore CPU
#define HETM_CPU_EN 1
#endif

typedef enum {
  HETM_BATCH_RUN  = 0,
  HETM_BATCH_DONE = 1,
  HETM_IS_EXIT    = 2
} HETM_GPU_STATE;

// the GPU callback launches the kernel inside of it
typedef void(*HeTM_callback)(int, void*);

typedef struct thread_args_ {
  int id;
  HeTM_callback callback;
  void *args;
  pthread_t thread;
} thread_args_s;

// sigleton
typedef struct HeTM_ {
  HETM_GPU_STATE statusGPU;
  int stopFlag;
  int nbCPUThreads, nbGPUBlocks, nbGPUThreads;
  int isCPUEnabled, isGPUEnabled, nbThreads;
  int isInterconflict; // set to 1 when a CPU-GPU conflict is found
  barrier_t GPUBarrier;
  thread_args_s *threadsInfo;
} HeTM_s;

extern HeTM_s HeTM_shared_data;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int HeTM_init(int nbCPUThreads, int nbGPUBlocks, int nbGPUThreads);
int HeTM_destroy();

// register two callbacks, first for the CPU, second for the GPU,
// and the global arguments (GPU thread will have the last id)
int HeTM_start(HeTM_callback, HeTM_callback, void *args);

// Waits the threads. Note: HeTM_set_is_stop(1) must be called
// before so the threads know it is time to stop
int HeTM_join_CPU_threads();

// TODO: convert into MACROs!
// ---
void HeTM_set_is_stop(int isStop);
int HeTM_is_stop();

void HeTM_set_is_interconflict(int isInterconflict);
int HeTM_is_interconflict();
// ---

void HeTM_set_GPU_status(HETM_GPU_STATE status);
HETM_GPU_STATE HeTM_get_GPU_status();

int HeTM_GPU_wait();

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HETM_H_GUARD_ */
