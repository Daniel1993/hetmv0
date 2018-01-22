#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="0.67"
CPU_PART="0.66"
P_INTERSECT="0.0 0.05 0.15 0.30"
DATASET="10000000"
#BLOCKS="256"
#GPU_THREADS="128"
BATCH_SIZE="50 100 150 250 500 750 1000 2000 3000"

# THREADS="1 2 4 6 8 10 12 14 16 22 30 42 56"
THREADS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 22 30 42 56"
DURATION=5000
SAMPLES=5
#./makeTM.sh

rm -f Bank.csv

for s in `seq $SAMPLES`
do
	# GPU-only
	make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=0.5 CPUEn=0 USE_TSX_IMPL=0 -j 14
	for t in $THREADS
	do
		timeout 20s ./bank -n $t -a $DATASET -d $DURATION
	done
	mv Bank.csv HeTM_GPU_only_s${s}.csv

	# TSX
	make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=0.5 USE_TSX_IMPL=1 -j 14
	for t in $THREADS
	do
		timeout 20s ./bank -n $t -a $DATASET -d $DURATION
	done
	mv Bank.csv HeTM_TSX_s${s}.csv

	# Tiny
	make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=0.5 USE_TSX_IMPL=0 -j 14
	for t in $THREADS
	do
		timeout 20s ./bank -n $t -a $DATASET -d $DURATION
	done
	mv Bank.csv HeTM_Tiny_s${s}.csv
done
