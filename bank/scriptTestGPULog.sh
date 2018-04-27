#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=5000
DATASET="1000 10000 100000 1000000 10000000 50000000" #2621440 # 90 000 000 is the max for my home machine
BLOCKS="2 4 8 16 32 64 128 256 512 1024" # 512
THREADS="2 4 8 16 32 64 128 256 512 1024"
BATCH_SIZE="1 2 4 8 16 32 64 128 256 512 1024"
DEFAULT_BATCH_SIZE="16"
DEFAULT_DATASET="1000000"
DEFAULT_BLOCKS="512"
DEFAULT_THREADS="512"
SAMPLES=5
#./makeTM.sh

rm -f Bank.csv

make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	T=$DEFAULT_BATCH_SIZE
	for t in $THREADS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_THRS_COMPRESSED_s${s}
	a=$DEFAULT_DATASET
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for b in $BLOCKS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_BLKS_COMPRESSED_s${s}
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for a in $DATASET
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_DTST_COMPRESSED_s${s}
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	for T in $BATCH_SIZE
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_BTCH_COMPRESSED_s${s}
done

make clean ; make CMP_TYPE=EXPLICIT GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	T=$DEFAULT_BATCH_SIZE
	for t in $THREADS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_THRS_EXPLICIT_s${s}
	a=$DEFAULT_DATASET
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for b in $BLOCKS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_BLKS_EXPLICIT_s${s}
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for a in $DATASET
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_DTST_EXPLICIT_s${s}
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	for T in $BATCH_SIZE
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_BTCH_EXPLICIT_s${s}
done

make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	T=$DEFAULT_BATCH_SIZE
	for t in $THREADS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_THRS_DISABLED_s${s}
	a=$DEFAULT_DATASET
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for b in $BLOCKS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_BLKS_DISABLED_s${s}
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for a in $DATASET
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_DTST_DISABLED_s${s}
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	for T in $BATCH_SIZE
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPUonly_BTCH_DISABLED_s${s}
done
