#!/bin/bash

GPU_PART="0.67"
CPU_PART="0.66"
P_INTERSECT="0.0" # 0.10 0.30 0.5 0.8 1.0"
DATASET="10000000"
GPU_THREADS="4 8 16 32 64 128 256 512 1024"
BLOCKS="4 8 16 32 64 128 256 512 1024"
BATCH_SIZE="1 2 4 8 16 32 64 128 256"
DATASET_SIZE="10000000 100000000 200000000 400000000 600000000"
THR_GPU_THREADS="4 1024"
THR_BLOCKS="4 1024"
THR_BATCH_SIZE="1 256"

DEFAULT_BATCH="1"
DEFAULT_GPU_THREADS="512"
DEFAULT_BLOCKS="512"

#"1 2 4 6 8 10 12 14 16 22 30 42 56"
THREADS="14"
DURATION=10000
SAMPLES=3

rm -f Bank.csv

### TODO: comment this
# sleep 20m

for s in `seq $SAMPLES`
do
	##################################
	# GPU-only
	make clean ; make CMP_TYPE=DISABLED GPU_PART=1.0 CPU_PART=0.0 \
		P_INTERSECT=0.0 CPUEn=0 USE_TSX_IMPL=0 BANK_PART=1 -j 14
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_GPU_THREADS
	for a in $DATASET_SIZE
	do
		for T in $BATCH_SIZE
		do
			timeout 20s ./bank -n $THREADS -a $a -d $DURATION -b $b -x $t -T $T
		done
		mv Bank.csv GPU_NO_HACKS_BTCH_${a}acc_s${s}
	done

	##################################
done
