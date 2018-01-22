#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="0.67"
CPU_PART="0.66"
P_INTERSECT="0.0 0.05 0.15 0.30"
DATASET="50000000"
#BLOCKS="256"
#GPU_THREADS="128"
BATCH_SIZE="25 50 75 100 125 150 250 500 750 1000 2000"

#"1 2 4 6 8 10 12 14 16 22 30 42 56"
THREADS="14"
DURATION=5000
SAMPLES=10
#./makeTM.sh

rm -f Bank.csv

for s in `seq $SAMPLES`
do
	# GPU-only
	make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=0.5 CPUEn=0 USE_TSX_IMPL=0 -j 14
	for b in $BATCH_SIZE
	do
		timeout 20s ./bank -n $THREADS -a $DATASET -d $DURATION -T $b
	done
	mv Bank.csv HeTM_GPU_only_s${s}.csv

	# TSX-only
	make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=0.5 GPUEn=0 USE_TSX_IMPL=1 -j 14
	for b in $BATCH_SIZE
	do
		timeout 20s ./bank -n $THREADS -a $DATASET -d $DURATION -T $b
	done
	mv Bank.csv HeTM_TSX_only_s${s}.csv

	# Tiny-only
	make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=0.5 GPUEn=0 USE_TSX_IMPL=0 -j 14
	for b in $BATCH_SIZE
	do
		timeout 20s ./bank -n $THREADS -a $DATASET -d $DURATION -T $b
	done
	mv Bank.csv HeTM_Tiny_only_s${s}.csv

	for i in $P_INTERSECT
	do
		# TSX --- CPU_INV
		make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$i CPU_INV=1 USE_TSX_IMPL=1 -j 14
		for b in $BATCH_SIZE
		do
			timeout 20s ./bank -n $THREADS -a $DATASET -d $DURATION -T $b
		done
		mv Bank.csv HeTM_TSX_CPU_INV_I${i}_s${s}.csv

		# TSX --- GPU_INV
		make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=1 -j 14
		for b in $BATCH_SIZE
		do
			timeout 20s ./bank -n $THREADS -a $DATASET -d $DURATION -T $b
		done
		mv Bank.csv HeTM_TSX_GPU_INV_I${i}_s${s}.csv

		# Tiny --- CPU_INV
		make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$i CPU_INV=1 USE_TSX_IMPL=0 -j 14
		for b in $BATCH_SIZE
		do
			timeout 20s ./bank -n $THREADS -a $DATASET -d $DURATION -T $b
		done
		mv Bank.csv HeTM_Tiny_CPU_INV_I${i}_s${s}.csv

		# Tiny --- GPU_INV
		make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=0 -j 14
		for b in $BATCH_SIZE
		do
			timeout 20s ./bank -n $THREADS -a $DATASET -d $DURATION -T $b
		done
		mv Bank.csv HeTM_Tiny_GPU_INV_I${i}_s${s}.csv
	done
done
