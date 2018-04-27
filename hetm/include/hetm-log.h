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
#define LOG_ACTUAL_SIZE      8191 /* closest Prime number 8191 */
#define LOG_GPU_THREADS      8192 /* 256*32 */
#define LOG_THREADS_IN_BLOCK 64
#define LOG_SIZE             16
#define STM_LOG_BUFFER_SIZE  1 /* buffer capacity (times LOG_SIZE) */
#else /* !HETM_VERS2_LOG */
#define LOG_SIZE             65536 // 131072 /* Initial log size */
#define STM_LOG_BUFFER_SIZE  4 /* buffer capacity (times LOG_SIZE) */
#endif /* HETM_VERS2_LOG */

#ifndef HETM_LOG_TYPE
#define HETM_LOG_TYPE HETM_VERS_LOG
#endif /* HETM_LOG_TYPE */

#if HETM_LOG_TYPE == HETM_VERS_LOG || HETM_LOG_TYPE == HETM_VERS2_LOG
typedef struct HeTM_CPULogEntry_t {
  long *pos;       /* Address written to */
  long val;        /* Value Written      */
  long time_stamp; /* Version counter    */
} HeTM_CPULogEntry;
#elif HETM_LOG_TYPE == HETM_ADDR_LOG
typedef struct HeTM_CPULogEntry_t {
  long *pos;       /* Address written to */
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
extern void *stm_baseMemPool;
#if HETM_LOG_TYPE == HETM_BMAP_LOG
extern void *stm_wsetCPU;
#endif

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
    CHUNKED_LOG_TEARDOWN(); // deletes the freed nodes
  }
  stm_thread_local_log = NULL;
}

// TODO: keep this inline for performance reasons
/* Add value to log */
static inline int
stm_log_newentry(HETM_LOG_T *log, long *pos, int val, long vers)
{
#if !defined(HETM_DISABLE_CHUNKS) && HETM_LOG_TYPE != HETM_VERS_LOG && HETM_LOG_TYPE != HETM_VERS2_LOG
  memman_access_addr(stm_devMemPoolBackupBmap, pos);
#endif /* HETM_DISABLE_CHUNKS */

#if HETM_LOG_TYPE == HETM_BMAP_LOG
  /* ********************************************** */
  // set bitmap
  // TODO: 2 ==> PR_LOCK_GRAN_BITS
  memman_access_addr_gran(stm_wsetCPU, stm_baseMemPool, pos, 1, 2);

  /* ********************************************** */
#else /* HETM_LOG_TYPE != HETM_BMAP_LOG */
  /* ********************************************** */
  // add entry in log

  HeTM_CPULogEntry entry;
#if HETM_LOG_TYPE == HETM_VERS_LOG || HETM_LOG_TYPE == HETM_VERS2_LOG
  entry.pos        = pos;
  entry.val        = val;
  entry.time_stamp = vers;
#elif HETM_LOG_TYPE == HETM_ADDR_LOG
  entry.pos = pos;
#endif /* HETM_LOG_TYPE == HETM_VERS_LOG */

#if HETM_LOG_TYPE == HETM_VERS2_LOG
// TODO: sizeof(PR_GRANULE_T)
  uintptr_t wordAddr = ((uintptr_t)pos) / sizeof(int);
  MOD_CHUNKED_LOG_APPEND(log, (void*)&entry, wordAddr);
#else
  CHUNKED_LOG_APPEND(log, (void*)&entry);
#endif

#endif /* HETM_LOG_TYPE == HETM_VERS_LOG */
  return 1;
}

#endif /* HETM_LOG_H_GUARD_ */
