#include <stdio.h>
#include <stdlib.h>
#include "hetm.cuh"

__thread HETM_LOG_T *stm_thread_local_log = NULL;

void *stm_devMemPoolBackupBmap;
void *stm_devMemPoolBmap;
void *stm_baseMemPool;

void *stm_wsetCPU; // only for HETM_BMAP_LOG
void *stm_wsetCPUCache; // only for HETM_BMAP_LOG
void *stm_wsetCPUCache_x64; // only for HETM_BMAP_LOG
size_t stm_wsetCPUCacheBits; // only for HETM_BMAP_LOG

__thread chunked_log_node_s *chunked_log_node_recycled[SIZE_OF_FREE_NODES];
__thread unsigned long chunked_log_free_ptr = 0;
__thread unsigned long chunked_log_alloc_ptr = 0;
__thread unsigned long chunked_log_free_r_ptr = 0;
__thread unsigned long chunked_log_alloc_r_ptr = 0;
long *hetm_batchCount = 0;

HETM_LOG_T* stm_log_init()
{
  HETM_LOG_T *res = NULL;
#if HETM_LOG_TYPE != HETM_BMAP_LOG /* ADDR or VERS */
  if (stm_thread_local_log != NULL) return stm_thread_local_log; // already init
  res = (HETM_LOG_T*)malloc(sizeof(HETM_LOG_T));

  CHUNKED_LOG_INIT(res, sizeof(HeTM_CPULogEntry), LOG_SIZE);
  stm_thread_local_log = res;

  const int nbChunksPreallocate = 2000;
  chunked_log_node_s *chunks[nbChunksPreallocate];
  for (int i = 0; i < nbChunksPreallocate; ++i) {
    // allocate some chunks
    chunks[i] = CHUNKED_LOG_ALLOC(res->gran, res->nb_elements);
  }
  for (int i = 0; i < nbChunksPreallocate; ++i) {
    CHUNKED_LOG_FREE(chunks[i]);
  }

#endif /* HETM_LOG_TYPE */
  // printf("new log=%p\n", res);
  // COMPILER_FENCE(); // some error
  __sync_synchronize();
  return res;
};

void
stm_log_newentry(HETM_LOG_T *log, long* pos, int val, long vers)
{
#if HETM_LOG_TYPE == HETM_BMAP_LOG
  /* ********************************************** */
// TODO: this thing is useless now!!! check what this actually affects
  // set bitmap
  // TODO: 2 ==> PR_LOCK_GRAN_BITS
  memman_access_addr_gran(stm_wsetCPU, stm_baseMemPool, pos, 1, 2, *hetm_batchCount);
  // memman_access_addr_gran(stm_wsetCPUCache, stm_baseMemPool, pos, 1, (stm_wsetCPUCacheBits+2), *hetm_batchCount);
  memman_access_addr_gran_x64(stm_wsetCPUCache_x64, stm_baseMemPool, pos, 1, (stm_wsetCPUCacheBits+2), *hetm_batchCount);

  /* ********************************************** */
#else /* HETM_LOG_TYPE != HETM_BMAP_LOG */
  /* ********************************************** */
  // add entry in log

#if HETM_LOG_TYPE == HETM_VERS_LOG
  uintptr_t base_addr = (uintptr_t)stm_baseMemPool;
  uintptr_t pos_addr = (uintptr_t)pos;
  uintptr_t pos_idx;
  base_addr >>= 2; // 32 bits --> remove the offset within a word
  pos_addr >>= 2;

  pos_idx = ((pos_addr - base_addr) + 1);

  // the 0 is the default value for empty
  // printf("val = %i\n", val);
  HeTM_CPULogEntry entry = {
    .pos = (uint32_t) pos_idx,
    .time_stamp = (uint32_t)vers,
    .val = (int32_t)val,
  };
  // if (entry.pos == 29) { /* don't forget the +1 */
  //   printf("entry.pos =%i \n", entry.pos);
  // }
#endif /* HETM_LOG_TYPE == HETM_VERS_LOG */

  CHUNKED_LOG_APPEND(log, (void*)&entry);

#endif /* HETM_LOG_TYPE == HETM_BMAP_LOG */
}
