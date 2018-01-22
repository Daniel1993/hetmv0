#ifndef BANK_AUX_H_
#define BANK_AUX_H_

// TODO: refactor this code else where
/* ################################################################### *
 * GPU ENABLE
 * ################################################################### */
// LOCK --> usa a cuda_barrier para não premitir avançar para o próximo batch (depois da cmp)

#define TEST_GPU_IN_LOOP_STREAM_CHECK_DONE \
	if (s->isCmpDone == 1) { \
		s->status = READ; \
		if (s->isCudaError) { \
			HeTM_set_is_interconflict(-1); \
		} else { \
			HeTM_set_is_interconflict(jobWithCuda_checkStreamFinal(d->cd, s, count)); \
		} \
		if (s->isCudaError) { \
			HeTM_set_is_interconflict(-1); \
		} \
		if (HeTM_is_interconflict() || s->count == NB_CMP_RETRIES) { \
			s->isThreshold = 1; \
		} \
	} \
//

#define TEST_GPU_IN_LOOP \
	if (HeTM_get_GPU_status() == HETM_BATCH_DONE) { \
		jobWithCuda_threadCheck(d->cd, s); \
		TEST_GPU_IN_LOOP_STREAM_CHECK_DONE; \
	} \
//

#ifdef USE_TSX_IMPL
  #define BANK_GET_STATS_TSX(d) \
    d->nb_aborts = TM_get_error(HTM_ABORT); \
    d->nb_aborts_1 = TM_get_error(HTM_CONFLICT); \
    d->nb_aborts_2 = TM_get_error(HTM_CAPACITY); \
//
#else
  #define BANK_GET_STATS_TSX(d) /* empty */
#endif /* USE_TSX_IMPL */

#if !defined(TM_COMPILER) && !defined(USE_TSX_IMPL)
  #define BANK_GET_STATS(d) \
    stm_get_stats("nb_aborts", &d->nb_aborts); \
    stm_get_stats("nb_aborts_1", &d->nb_aborts_1); \
    stm_get_stats("nb_aborts_2", &d->nb_aborts_2); \
		printf("aborts=%lu\n", d->nb_aborts); \
    stm_get_stats("nb_aborts_locked_read", &d->nb_aborts_locked_read); \
    stm_get_stats("nb_aborts_locked_write", &d->nb_aborts_locked_write); \
    stm_get_stats("nb_aborts_validate_read", &d->nb_aborts_validate_read); \
    stm_get_stats("nb_aborts_validate_write", &d->nb_aborts_validate_write); \
    stm_get_stats("nb_aborts_validate_commit", &d->nb_aborts_validate_commit); \
    stm_get_stats("nb_aborts_invalid_memory", &d->nb_aborts_invalid_memory); \
    stm_get_stats("nb_aborts_killed", &d->nb_aborts_killed); \
    stm_get_stats("locked_reads_ok", &d->locked_reads_ok); \
    stm_get_stats("locked_reads_failed", &d->locked_reads_failed); \
    stm_get_stats("max_retries", &d->max_retries); \
//
#elif !defined(TM_COMPILER) && defined(USE_TSX_IMPL)
#define BANK_GET_STATS(d) \
	BANK_GET_STATS_TSX(d); \
	printf("aborts=%lu\n", d->nb_aborts); \
//
#else
  #define BANK_GET_STATS(d) /* empty */
#endif /* ! TM_COMPILER */

#define BANK_TEARDOWN_TX() \
  jobWithCuda_exitStream(s); \
//

//   Memory access layout
// +------------+--------------------+
// | NOT ACCESS |      CPU_PART      |
// +------------+--------------------+
// +------------------+--------------+
// |     GPU_PART     |  NOT ACCESS  |
// +------------------+--------------+
//

#define BANK_PREPARE_TRANSFER(id, total, account_vec, nb_accounts) ({ \
  int i; \
  for (i = 0; i < d->trfs; i += 2) { \
    int is_intersect = IS_INTERSECT_HIT( erand48(seed)*100000.0 ); \
		if (!is_intersect) { \
			src = CPU_ACCESS(erand48(seed)*nb_accounts, nb_accounts); \
			dst = CPU_ACCESS(erand48(seed)*nb_accounts, nb_accounts); \
		} else { \
			src = INTERSECT_ACCESS(erand48(seed)*nb_accounts, nb_accounts); \
			dst = INTERSECT_ACCESS(erand48(seed)*nb_accounts, nb_accounts); \
		} \
    if (dst == src) \
      dst = (src + 1) % nb_accounts; /* TODO: check this */ \
    accounts_vec[i]   = src; \
    accounts_vec[i+1] = dst; \
  } \
}) \
//

#define BANK_CMP_KERNEL_NO_LOCK   HeTM_set_GPU_status(HETM_BATCH_DONE)
#define BANK_CMP_KERNEL_BM_TRANSF jobWithCuda_bm(*cd, valid_vec)

// primeira barreira sinaliza comparação
// segunda barreira espera finalização da comparação

#define BANK_CMP_KERNEL() \
	TIMER_T timerStream; \
	DEBUG_PRINT("Starting comparisson kernel!\n"); \
	BANK_CMP_KERNEL_NO_LOCK; /* TODO[Ricardo]: barrier_wait if !NO_LOCK */\
	HeTM_GPU_wait(); /*barrier_cross(*d->cuda_barrier);*/ \
  TIMER_READ(timerStream); \
  duration_cmp += TIMER_DIFF_SECONDS(cmp_start, timerStream) * 1000; \
//

#endif /* BANK_AUX_H_ */
