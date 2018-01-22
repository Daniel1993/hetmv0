#ifndef SETUP_KERNELS_H_
#define SETUP_KERNELS_H_

#include "knlman.h"

int HeTM_setup_bankTx(int nbBlocks, int nbThreads);
int HeTM_bankTx_cpy_IO();
int HeTM_teardown_bankTx();

int HeTM_setup_checkTxCompressed();
int HeTM_teardown_checkTxCompressed();

int HeTM_setup_checkTxExplicit();
int HeTM_teardown_checkTxExplicit();

int HeTM_setup_finalTxLog2();
int HeTM_teardown_finalTxLog2();

#endif /* SETUP_KERNELS_H_ */
