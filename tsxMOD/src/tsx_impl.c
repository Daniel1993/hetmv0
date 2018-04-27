#include "tsx_impl.h"

 __thread HETM_LOG_T *HeTM_log;
 __thread void* HeTM_bufAddrs[HETM_BUFFER_MAXSIZE];
 __thread uintptr_t HeTM_bufVal[HETM_BUFFER_MAXSIZE];
 __thread uintptr_t HeTM_bufVers[HETM_BUFFER_MAXSIZE];
 __thread size_t HeTM_ptr;
 __thread uintptr_t HeTM_version;

int errors[MAX_THREADS][HTM_NB_ERRORS];
