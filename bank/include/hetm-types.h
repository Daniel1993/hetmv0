#ifndef HETM_TYPES_H_GUARD_
#define HETM_TYPES_H_GUARD_

// TODO: this is Benchmark stuff

/* ################################################################### *
* BANK ACCOUNTS
* ################################################################### */
typedef int /*__attribute__((aligned (64)))*/ account_t;

typedef struct bank {
  account_t *accounts;
  long size;
} bank_t;

typedef struct memcd {
  account_t *accounts;
  int *version;
  long size;
  int ways;
  int *version_bm;
  int version_bm_size;
} memcd_t;

typedef struct packet {
	int vers;
	long key;
} packet_t;

#endif /* HETM_TYPES_H_GUARD_ */
