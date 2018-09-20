#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=12000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=8
#./makeTM.sh

CPU_THREADS=4
LOW_CPU_THREADS=10
HIGH_CPU_THREADS=20

rm -f Bank.csv

LARGE_DATASET=100000000 #2621440 # 90 000 000 is the max for my home machine
# LARGE_DATASET=20000000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P20=61000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P20=122000 #2621440 # 90 000 000 is the max for my home machine

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 7 $SAMPLES`
	do
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 16 -l 25 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 16 -l 50 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 16 -l 75 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT
		mv Bank.csv ${1}${s}
	done
}

############### LARGE
###########################################################################
### Fixed the amount of CPU threads
CPU_THREADS=8

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14
doRunLargeDTST GPUonly_rand_sep_BMAP_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=1 PROFILE=1 -j 14
doRunLargeDTST GPUonly_rand_sep_DISABLED_s

# ############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
		BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_DISABLED_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_VERS_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_BMAP_s
#
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
		BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_DISABLED_STM_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_VERS_STM_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_BMAP_STM_s

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST VERS_rand_sep_s

############## BMAP
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST BMAP_rand_sep_s
###########################################################################

############### SMALL
###########################################################################
CPU_THREADS=8
LARGE_DATASET=10000000

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14
doRunLargeDTST GPUonly_rand_sep_BMAP_SMALL_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=1 PROFILE=1 -j 14
doRunLargeDTST GPUonly_rand_sep_DISABLED_SMALL_s

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
		BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_DISABLED_SMALL_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_VERS_SMALL_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_BMAP_SMALL_s

make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
		BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_DISABLED_STM_SMALL_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_VERS_STM_SMALL_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_BMAP_STM_SMALL_s

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST VERS_rand_sep_SMALL_s

############## BMAP
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST BMAP_rand_sep_SMALL_s
###########################################################################


################ Zipf


############### LARGE
###########################################################################
LARGE_DATASET=49999999 # half the size
CPU_THREADS=4

cd java_zipf
GPU_INPUT="GPU_input_${LARGE_DATASET}_099_4194304.txt"
CPU_INPUT="CPU_input_${LARGE_DATASET}_099_57344.txt"

if [ ! -f ../$GPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 4194304 > ../$GPU_INPUT
fi
if [ ! -f ../$CPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 57344 > ../$CPU_INPUT
fi
cd ..

LARGE_DATASET=100000000 # half the size

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTST GPUonly_zipf_sep_BMAP_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTST GPUonly_zipf_sep_DISABLED_s

# ############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
		BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_DISABLED_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_VERS_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_BMAP_s

make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_DISABLED_STM_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_VERS_STM_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_BMAP_STM_s

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST VERS_zipf_sep_s

############## BMAP
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST BMAP_zipf_sep_s
###########################################################################

############### SMALL
###########################################################################
LARGE_DATASET=4999999 # half the size
CPU_THREADS=4

cd java_zipf
GPU_INPUT="GPU_input_${LARGE_DATASET}_099_4194304.txt"
CPU_INPUT="CPU_input_${LARGE_DATASET}_099_57344.txt"

if [ ! -f ../$GPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 4194304 > ../$GPU_INPUT
fi
if [ ! -f ../$CPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 57344 > ../$CPU_INPUT
fi
cd ..

LARGE_DATASET=10000000 # half the size

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 CPUEn=0 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTST GPUonly_zipf_sep_BMAP_SMALL_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTST GPUonly_zipf_sep_DISABLED_SMALL_s

# ############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
		BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_DISABLED_SMALL_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_VERS_SMALL_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_BMAP_SMALL_s

make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_DISABLED_STM_SMALL_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_VERS_STM_SMALL_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_zipf_sep_BMAP_STM_SMALL_s

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST VERS_zipf_sep_SMALL_s

############## BMAP
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST BMAP_zipf_sep_SMALL_s
###########################################################################
