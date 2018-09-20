#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="0.52"
CPU_PART="0.52"
P_INTERSECT="0.0"
DURATION=5000
DATASET="50000 100000 1000000 10000000 50000000" #2621440 # 90 000 000 is the max for my home machine
BLOCKS="2 4 8 16 32 64 128 256 512 1024" # 512
THREADS="2 4 8 16 32 64 128 256 512 1024"
BATCH_SIZE="1 2 4 8 16 32 64 128 256 512 1024"
DEFAULT_BATCH_SIZE="4"
DEFAULT_DATASET="1000000"
DEFAULT_BLOCKS="512"
DEFAULT_THREADS="512"
SAMPLES=1
#./makeTM.sh

rm -f Bank.csv

function doExperiment {
	for s in `seq $SAMPLES`
	do
		a=$DEFAULT_DATASET
		b=$DEFAULT_BLOCKS
		T=$DEFAULT_BATCH_SIZE
		for t in $THREADS
		do
			timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
		done
		mv Bank.csv GPUonly_THRS_${1}_s${s}
		a=$DEFAULT_DATASET
		t=$DEFAULT_THREADS
		T=$DEFAULT_BATCH_SIZE
		for b in $BLOCKS
		do
			timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
		done
		mv Bank.csv GPUonly_BLKS_${1}_s${s}
		b=$DEFAULT_BLOCKS
		t=$DEFAULT_THREADS
		T=$DEFAULT_BATCH_SIZE
		for a in $DATASET
		do
			timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
		done
		mv Bank.csv GPUonly_DTST_${1}_s${s}
		a=$DEFAULT_DATASET
		b=$DEFAULT_BLOCKS
		t=$DEFAULT_THREADS
		for T in $BATCH_SIZE
		do
			timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
		done
		mv Bank.csv GPUonly_BTCH_${1}_s${s}
	done
}

make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=$GPU_PART \
	CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT PROFILE=1 BENCH=BANK CPUEn=0 -j 14
doExperiment COMPRESSED

make clean ; make CMP_TYPE=EXPLICIT BANK_PART=1 GPU_PART=$GPU_PART \
	CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT PROFILE=1 BENCH=BANK CPUEn=0 -j 14
doExperiment EXPLICIT

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 GPU_PART=$GPU_PART \
	CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT PROFILE=1 BENCH=BANK CPUEn=0 -j 14
doExperiment DISABLED

make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=$GPU_PART \
	CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT LOG_TYPE=VERS PROFILE=1 BENCH=BANK -j 14
doExperiment COMPRESSED_W_VERS

make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=$GPU_PART \
	CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT LOG_TYPE=BMAP PROFILE=1 BENCH=BANK -j 14
doExperiment COMPRESSED_W_BMAP

# BLOCKS="2 4 8 16 32 64 128 256" # 512
# THREADS="2 4 8 16 32 64 128"
# BATCH_SIZE="1 2 4 8"

make clean ; make CMP_TYPE=EXPLICIT BANK_PART=1 GPU_PART=$GPU_PART \
	CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT LOG_TYPE=VERS PROFILE=1 BENCH=BANK -j 14
doExperiment EXPLICIT_W_VERS

make clean ; make CMP_TYPE=EXPLICIT BANK_PART=1 GPU_PART=$GPU_PART \
	CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT LOG_TYPE=BMAP PROFILE=1 BENCH=BANK -j 14
doExperiment EXPLICIT_W_BMAP