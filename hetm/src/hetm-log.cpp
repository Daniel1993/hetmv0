#include <stdio.h>
#include <stdlib.h>
#include "hetm-log.h"

__thread HETM_LOG_T *stm_thread_local_log = NULL;

void *stm_devMemPoolBackupBmap;
void *stm_baseMemPool;
#if HETM_LOG_TYPE == HETM_BMAP_LOG
void *stm_wsetCPU;
#endif
