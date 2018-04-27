#ifndef STM_WRAPPER_H_GUARD_
#define STM_WRAPPER_H_GUARD_

/*
 * Useful macros to work with transactions. Note that, to use nested
 * transactions, one should check the environment returned by
 * stm_get_env() and only call sigsetjmp() if it is not null.
 */

 /*
  * When a thread enters its routine, one must call TM_INIT_THREAD with
  * the dataset base pointer and size (located in HeTM_shared_data.hostMemPool)
  */

// TODO: requires to compile HeTM with the correct set of flags
#ifndef USE_TSX_IMPL
// goes for TinySTM
#define TM_START(tid, ro)     { stm_tx_attr_t _a = {{.id = tid, .read_only = ro}}; sigjmp_buf *_e = stm_start(_a); if (_e != NULL) sigsetjmp(*_e, 0)
#define TM_LOAD(addr)         stm_load((stm_word_t *)addr)
#define TM_STORE(addr, value) stm_store((stm_word_t *)addr, (stm_word_t)value)
#define TM_COMMIT             stm_commit(); }

#define TM_GET_LOG(p)         p = stm_thread_local_log
#define TM_LOG(val)           stm_log_add(0, val)
#define TM_LOG2(pos,val)      stm_log_add(pos, val)
#define TM_FREE(point)        free(point)

#define TM_INIT(nb_threads)   stm_init(); mod_ab_init(0, NULL)
#define TM_EXIT()             stm_exit()
#define TM_INIT_THREAD(p,s)   stm_init_thread(); /*stm_log_init_bm(p,s)*/
#define TM_EXIT_THREAD()      stm_exit_thread()
#else /* USE_TSX_IMPL */
// redefine with TSX
#define GRANULE_TYPE int
#include "tsx_impl.h"
#define TM_START(tid, ro) 		HTM_SGL_begin();
#define TM_COMMIT 		        HTM_SGL_commit();
#define TM_LOAD(addr)         HTM_SGL_read(addr)
#define TM_STORE(addr, value) HTM_SGL_write(addr, value)

#define TM_GET_LOG(p)         HeTM_get_log(&p)
#define TM_LOG(val)           /*stm_log_add(0,val)*/
#define TM_LOG2(pos,val)      /*stm_log_add(pos,val)*/
#define TM_FREE(point)        free(point)

#define TM_INIT(nb_threads)		HTM_init(nb_threads) /*stm_init(); mod_ab_init(0, NULL)*/
#define TM_EXIT()             HTM_exit(); stm_exit()
#define TM_INIT_THREAD(p,s)   HTM_SGL_init_thr(); stm_init_thread(); /*stm_log_init_bm(p, s)*/
#define TM_EXIT_THREAD()      HTM_SGL_exit_thr() /*stm_exit_thread()*/

#endif /* USE_TSX_IMPL */


#endif /* STM_WRAPPER_H_GUARD_ */
