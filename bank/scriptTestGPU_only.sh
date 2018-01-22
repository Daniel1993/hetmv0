#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DATASET=100000 #2621440 # 90 000 000 is the max for my home machine
DURATION=5000
BLOCKS="512"#"2 4 8 16 32 64 256 512 1024 2048 4096"
THREADS="2 4 8 16 32 64 96 256 320 512 640 768 1024"
SAMPLES=5
#./makeTM.sh

rm -f Bank.csv

make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	for b in $BLOCKS
	do
		for t in $THREADS
		do
			./bank -b $b -x $t -a $DATASET -d $DURATION
		done
	done
	mv Bank.csv GPUonly_${b}blks_COMPRESSED_s${s}
done

make clean ; make CMP_TYPE=EXPLICIT GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	for b in $BLOCKS
	do
		for t in $THREADS
		do
			./bank -b $b -x $t -a $DATASET -d $DURATION
		done
	done
	mv Bank.csv GPUonly_${b}blks_EXPLICIT_s${s}
done

make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	for b in $BLOCKS
	do
		for t in $THREADS
		do
			./bank -b $b -x $t -a $DATASET -d $DURATION
		done
	done
	mv Bank.csv GPUonly_${b}blks_DISABLED_s${s}
done
