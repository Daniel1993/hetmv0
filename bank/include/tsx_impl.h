#ifndef TSX_IMPL_GUARD_H_
#define TSX_IMPL_GUARD_H_

#ifdef USE_TSX_IMPL

#include "htm_retry_template.h"
#include "log.h"
#include "rdtsc.h"

#ifndef MAX_THREADS
#define MAX_THREADS 128 // TODO: used somewhere else
#endif
#define HETM_BUFFER_MAXSIZE 1024

extern __thread HeTM_CPULog_t *HeTM_log;
extern __thread void* HeTM_bufAddrs[HETM_BUFFER_MAXSIZE];
extern __thread uintptr_t HeTM_bufVal[HETM_BUFFER_MAXSIZE];
extern __thread uintptr_t HeTM_bufVers[HETM_BUFFER_MAXSIZE];
extern __thread uintptr_t HeTM_version;
extern __thread size_t HeTM_ptr;
extern int errors[MAX_THREADS][HTM_NB_ERRORS];

// updates the statistics
#undef HTM_INC
#define HTM_INC(status) \
  TM_inc_error(HTM_SGL_tid, status) \
//

#define TM_inc_error(tid, error) \
  HTM_ERROR_INC(error, errors[tid])

#define TM_get_error(error) ({ \
  int res = errors[HTM_SGL_tid][error]; \
  res; \
}) \

// #undef AFTER_ABORT
// #define AFTER_ABORT(tid, budget, status) printf(".")

#undef BEFORE_TRANSACTION
#define BEFORE_TRANSACTION(tid, budget) \
  HeTM_ptr = 0; \
//

#undef BEFORE_COMMIT
#define BEFORE_COMMIT(tid, budget, status) ({ \
  HeTM_version = rdtsc(); \
})

#undef AFTER_TRANSACTION
#define AFTER_TRANSACTION(tid, budget) ({ \
  int i; \
  for (i = 0; i < HeTM_ptr; ++i) { \
    stm_log_newentry(HeTM_log, (long*)HeTM_bufAddrs[i], HeTM_bufVal[i], HeTM_version); \
  } \
})

#undef HTM_SGL_after_write
#define HTM_SGL_after_write(addr, val) \
  HeTM_bufAddrs[HeTM_ptr] = addr; \
  HeTM_bufVal[HeTM_ptr]   = val; \
  HeTM_bufVers[HeTM_ptr]  = 0; \
  HeTM_ptr++; \
//

#undef AFTER_SGL_BEGIN
#define AFTER_SGL_BEGIN(tid) \
  errors[tid][HTM_FALLBACK]++;

// not defined in the template
#define HTM_SGL_init_thr() ({ HTM_thr_init(); HeTM_log = stm_log_init(); })
#define HTM_SGL_exit_thr() ({ HTM_thr_exit(); stm_log_free(HeTM_log); })

#define HeTM_get_log(ptr) ({ *(HeTM_CPULogNode_t **)ptr = stm_log_read(HeTM_log); })

#undef HTM_SGL_before_read
#define HTM_SGL_before_read(addr) /* empty */

#endif /* USE_TSX_IMPL */

#endif /* TSX_IMPL_GUARD_H_ */
