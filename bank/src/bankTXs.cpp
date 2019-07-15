#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

#include "bank.h"
#include <cmath>
#include "hetm-cmp-kernels.cuh"
#include "setupKernels.cuh"
#include "bank_aux.h"
#include "CheckAllFlags.h"
#include "input_handler.h"

#ifndef HETM_BANK_PART_SCALE
#define HETM_BANK_PART_SCALE 10
#endif /* HETM_BANK_PART_SCALE */

thread_local static unsigned someSeed = 0x034e3115;

int transfer2(account_t *accounts, volatile unsigned *positions, int isInter, int count, int tid, int nbAccounts)
{
	volatile int i;
	int z = 0;
	int n;

#if BANK_PART == 10
	void *pos[count*HETM_BANK_PART_SCALE];
#else
	void *pos[count];
#endif
	void *pos_write;
	int accountIdx;
	unsigned seedCopy = someSeed;

	double count_amount = 0.0;

	unsigned input = positions[0];
	unsigned randNum;

	unsigned seedState;

	seedCopy += tid ^ 1234;
	RAND_R_FNC(someSeed);

	seedCopy  = someSeed;
	seedState = someSeed;

	// for (n = 0; n < count; n++) {
	// 	randNum = RAND_R_FNC(seedCopy);
	// 	if (!isInter) {
	// 		accountIdx = CPU_ACCESS(randNum, nbAccounts);
	// 	} else {
	// 		accountIdx = randNum % nbAccounts;
	// 	}
	// 	pos[n] = accounts + accountIdx;
	// 	// __builtin_prefetch(pos[n], 1, 1);
	// }

  TM_START(z, RW);

	seedCopy = seedState;

	int forLoop = count;

#if BANK_PART == 10
	forLoop *= HETM_BANK_PART_SCALE;
#endif

	for (n = 0; n < forLoop; n++) {
		randNum = RAND_R_FNC(seedCopy);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			accountIdx = /*(n == 0) ? tid * 64 : */INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		pos[n] = accounts + accountIdx;
		int someLoad = TM_LOAD(accounts + accountIdx);
		count_amount += someLoad;
		// int someLoad = TM_LOAD(pos[n]);
		// count_amount += someLoad;
  }

	seedCopy = seedState;

#if BANK_PART == 10
	forLoop = count;
#endif

	// TODO: store must be controlled with parsedData.access_controller
	// -----------------
	for (n = 0; n < count; n++) {
		randNum = RAND_R_FNC(seedCopy);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			accountIdx = /*(n == 0) ? tid * 64 : */INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		TM_STORE(accounts + accountIdx, count_amount * input);
		// TM_STORE(pos[n], count_amount * input);
	}

	TM_COMMIT;

	someSeed = seedCopy;

	return 0;
}

int readOnly2(account_t *accounts, volatile unsigned *positions, int isInter, int count, int tid, int nbAccounts)
{
	volatile int i;
	int z = 0;
	int n;
	void *pos[count];
	void *pos_write;
	int accountIdx;

	double count_amount = 0.0;

	unsigned input = positions[0];
	unsigned randNum;

	someSeed += tid ^ 1234;
	RAND_R_FNC(someSeed);

  TM_START(z, RW);
#if BANK_PART == 10
	count *= HETM_BANK_PART_SCALE;
#endif

	for (n = 0; n < count; n++) {
		randNum = RAND_R_FNC(someSeed);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			accountIdx = /*(n == 0) ? tid * 64 : */INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		int someLoad = TM_LOAD(accounts + accountIdx);
		count_amount += someLoad * input;
  }

	TM_COMMIT;

	return count_amount;
}
