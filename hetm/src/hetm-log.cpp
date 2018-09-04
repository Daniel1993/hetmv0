#include <stdio.h>
#include <stdlib.h>
#include "hetm-log.h"

__thread HETM_LOG_T *stm_thread_local_log = NULL;

void *stm_devMemPoolBackupBmap;
void *stm_baseMemPool;
void *stm_wsetCPU; // only for HETM_BMAP_LOG
void *stm_wsetCPUCache; // only for HETM_BMAP_LOG
size_t stm_wsetCPUCacheBits; // only for HETM_BMAP_LOG

__thread chunked_log_node_s *chunked_log_node_recycled[SIZE_OF_FREE_NODES];
__thread unsigned long chunked_log_free_ptr = 0;
__thread unsigned long chunked_log_alloc_ptr = 0;
__thread unsigned long chunked_log_free_r_ptr = 0;
__thread unsigned long chunked_log_alloc_r_ptr = 0;
