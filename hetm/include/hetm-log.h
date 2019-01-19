#ifndef HETM_LOG_H_GUARD_
#define HETM_LOG_H_GUARD_

#include <stdio.h>
#include <stdlib.h>
#include "memman.h"
// #include "pr-stm-wrapper.cuh"

#include "chunked-log.h"

// TODO: these don't go through
#define HETM_VERS_LOG  1
#define HETM_ADDR_LOG  2
#define HETM_BMAP_LOG  3
#define HETM_VERS2_LOG 4

#if HETM_LOG_TYPE == HETM_VERS2_LOG
#define LOG_ACTUAL_SIZE      4093 // 8191 /* closest Prime number */
#define LOG_GPU_THREADS      4096 // 8192 /* LOG_THREADS_IN_BLOCK*... */
#define LOG_THREADS_IN_BLOCK 128
#define LOG_SIZE             32
#define STM_LOG_BUFFER_SIZE  4 /* buffer capacity (times LOG_SIZE) */
#else /* !HETM_VERS2_LOG */
#define LOG_SIZE             131072 // 32768 // 65536 // 131072 /* Initial log size */
#define STM_LOG_BUFFER_SIZE  1 // 4 /* buffer capacity (times LOG_SIZE) */
#endif /* HETM_VERS2_LOG */

// BMAP only
#define DEFAULT_BITMAP_GRANULARITY_BITS (10) // 1kB --> then I use a smart copy
#define DEFAULT_BITMAP_GRANULARITY (0x1<<10)
// #define DEFAULT_BITMAP_GRANULARITY_BITS (12) // 4kB --> then I use a smart copy
// #define DEFAULT_BITMAP_GRANULARITY (0x1<<12)
// #define DEFAULT_BITMAP_GRANULARITY_BITS (14) // 16kB --> then I use a smart copy
// #define DEFAULT_BITMAP_GRANULARITY (0x1<<14)

#ifndef HETM_LOG_TYPE
#define HETM_LOG_TYPE HETM_VERS_LOG
#endif /* HETM_LOG_TYPE */

#if HETM_LOG_TYPE == HETM_VERS_LOG || HETM_LOG_TYPE == HETM_VERS2_LOG
typedef struct HeTM_CPULogEntry_t {
  volatile uint32_t pos;         /* Address written to --> offset */
  volatile uint32_t time_stamp;  /* Version counter    */
  volatile int32_t  val;              /* Value Written      */
} __attribute__((packed)) HeTM_CPULogEntry; // in the GPU does not work!
#elif HETM_LOG_TYPE == HETM_ADDR_LOG
typedef struct HeTM_CPULogEntry_t {
  int *pos;       /* Address written to */
} HeTM_CPULogEntry;
#else
// no log entries
#endif /* HETM_LOG_TYPE == HETM_VERS_LOG */

#if HETM_LOG_TYPE == HETM_VERS2_LOG
#define HETM_LOG_T mod_chunked_log_s
#else /* HETM_LOG_TYPE != HETM_VERS2_LOG */
#define HETM_LOG_T chunked_log_s
#endif  /* HETM_LOG_TYPE */

extern __thread HETM_LOG_T *stm_thread_local_log;

extern void *stm_devMemPoolBackupBmap;
extern void *stm_devMemPoolBmap;
extern void *stm_baseMemPool;
extern void *stm_wsetCPU; // only for HETM_BMAP_LOG
extern void *stm_wsetCPUCache; // only for HETM_BMAP_LOG
extern size_t stm_wsetCPUCacheBits; // only for HETM_BMAP_LOG
extern long *hetm_batchCount;

/* ################################################################### *
 * inline FUNCTIONS
 * ################################################################### */

/* Init log: must be called per thread (thread local log) */
static inline HETM_LOG_T* stm_log_init()
{
  HETM_LOG_T *res = NULL;
#if HETM_LOG_TYPE == HETM_VERS2_LOG
  if (stm_thread_local_log != NULL) return stm_thread_local_log; // already init
  res = (HETM_LOG_T*)malloc(sizeof(HETM_LOG_T));

  MOD_CHUNKED_LOG_INIT(res, sizeof(HeTM_CPULogEntry), LOG_SIZE, LOG_ACTUAL_SIZE);
  stm_thread_local_log = res;
#elif HETM_LOG_TYPE != HETM_BMAP_LOG /* ADDR or VERS */
  if (stm_thread_local_log != NULL) return stm_thread_local_log; // already init
  res = (HETM_LOG_T*)malloc(sizeof(HETM_LOG_T));

  CHUNKED_LOG_INIT(res, sizeof(HeTM_CPULogEntry), LOG_SIZE);
  stm_thread_local_log = res;
#endif /* HETM_LOG_TYPE */
  // printf("new log=%p\n", res);
  // COMPILER_FENCE(); // some error
  __sync_synchronize();
  return res;
};

static inline size_t stm_log_size(chunked_log_node_s *node)
{
  size_t res = 0;
  chunked_log_node_s *tmp = node;

  if (tmp == NULL) return res;

  while (tmp->next != NULL) {
    res++;
    tmp = tmp->next;
  }

  return res;
}

/* Read the log */
#if HETM_LOG_TYPE != HETM_VERS2_LOG
static inline chunked_log_node_s* stm_log_truncate(HETM_LOG_T *log)
{
  HETM_LOG_T truncated = CHUNKED_LOG_TRUNCATE(log, STM_LOG_BUFFER_SIZE);
  return truncated.first;
}
#endif

static inline void stm_log_node_free(chunked_log_node_s *node)
{
  CHUNKED_LOG_FREE(node); // recycles
}

/* Clean up */
static inline void stm_log_free(HETM_LOG_T *log)
{
  if (log != NULL) {
#if HETM_LOG_TYPE == HETM_VERS2_LOG
    MOD_CHUNKED_LOG_DESTROY(log);
#else
    CHUNKED_LOG_DESTROY(log);
#endif
  }
  // CHUNKED_LOG_TEARDOWN(); // TODO: implement thread-local chunk_nodes
  stm_thread_local_log = NULL;
}

// TODO: keep this inline for performance reasons
/* Add value to log */
static inline int
stm_log_newentry(HETM_LOG_T *log, long* volatile pos, int volatile val, long vers)
{
#if !defined(HETM_DISABLE_CHUNKS)
  memman_access_addr(stm_devMemPoolBackupBmap, pos, 1);
#endif /* HETM_DISABLE_CHUNKS */

#if HETM_LOG_TYPE == HETM_BMAP_LOG
  /* ********************************************** */
  // set bitmap
  // TODO: 2 ==> PR_LOCK_GRAN_BITS
  memman_access_addr_gran(stm_wsetCPU, stm_baseMemPool, pos, 1, 2, *hetm_batchCount);
  memman_access_addr_gran(stm_wsetCPUCache, stm_baseMemPool, pos, 1, (stm_wsetCPUCacheBits+2), *hetm_batchCount);

  /* ********************************************** */
#else /* HETM_LOG_TYPE != HETM_BMAP_LOG */
  /* ********************************************** */
  // add entry in log

#if HETM_LOG_TYPE == HETM_VERS_LOG || HETM_LOG_TYPE == HETM_VERS2_LOG
  uintptr_t base_addr = (uintptr_t)stm_baseMemPool;
  uintptr_t pos_addr = (uintptr_t)pos;
  uintptr_t pos_idx;
  base_addr >>= 2; // 32 bits --> remove the offset within a word
  pos_addr >>= 2;

  pos_idx = ((pos_addr - base_addr) + 1);

  // the 0 is the default value for empty
  // printf("val = %i\n", val);
  volatile HeTM_CPULogEntry entry = {
    .pos = (uint32_t) pos_idx,
    .time_stamp = (uint32_t)vers,
    .val = (int32_t)val,
  };
  // if (entry.pos == 29) { /* don't forget the +1 */
  //   printf("entry.pos =%i \n", entry.pos);
  // }
#elif HETM_LOG_TYPE == HETM_ADDR_LOG
  volatile HeTM_CPULogEntry entry;
  entry.pos = (int*)pos;
#endif /* HETM_LOG_TYPE == HETM_VERS_LOG */

#if HETM_LOG_TYPE == HETM_VERS2_LOG
// TODO: sizeof(PR_GRANULE_T)
  uintptr_t wordAddr = ((uintptr_t)pos) / sizeof(int);
  MOD_CHUNKED_LOG_APPEND(log, (void*)&entry, wordAddr);
#else /* HETM_VERS_LOG */
  CHUNKED_LOG_APPEND(log, (void*)&entry);
#endif /* HETM_LOG_TYPE == HETM_VERS2_LOG */

#endif /* HETM_LOG_TYPE == HETM_BMAP_LOG */
  return 1;
}

#endif /* HETM_LOG_H_GUARD_ */
