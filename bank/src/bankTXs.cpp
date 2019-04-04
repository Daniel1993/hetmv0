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

thread_local static unsigned someSeed = 0x034e3115;

int transfer2(account_t *accounts, volatile unsigned *positions, int isInter, int count, int tid, int nbAccounts)
{
	volatile int i;
	int z = 0;
	int n;
	void *pos[count];
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

	for (n = 0; n < count; n++) {
		randNum = RAND_R_FNC(seedCopy);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
				accountIdx = randNum % nbAccounts;
		}
		pos[n] = accounts + accountIdx;
		int someLoad = TM_LOAD(accounts + accountIdx);
		count_amount += someLoad;
		// int someLoad = TM_LOAD(pos[n]);
		// count_amount += someLoad;
  }

	seedCopy = seedState;

	// TODO: store must be controlled with parsedData.access_controller
	// -----------------
	for (n = 0; n < count; n++) {
		randNum = RAND_R_FNC(seedCopy);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
			accountIdx = randNum % nbAccounts;
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

	for (n = 0; n < count; n++) {
		randNum = RAND_R_FNC(someSeed);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
			accountIdx = randNum % nbAccounts;
		}
		int someLoad = TM_LOAD(accounts + accountIdx);
		count_amount += someLoad * input;
  }

	TM_COMMIT;

	return count_amount;
}
