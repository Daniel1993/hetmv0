#!/bin/bash

SAMPLES=3
DURATION=30000
#./makeTM.sh
# DURATION_ORG=12000
# DURATION_GPU=6000
# DURATION=$DURATION_ORG

DATA_FOLDER=$(pwd)/data
# mkdir -p $DATA_FOLDER
# cd ../../bank
# rm -f Bank.csv

mkdir -p data/

L_DATASET=150000000
S_DATASET=15000000
# CPU_BACKOFF=250

CPU_THREADS=8
# CPU_THREADS=2
# GPU_BLOCKS=80
GPU_BLOCKS=80
GPU_THREADS=256

TRANSACTION_SIZE=4
CPU_BACKOFF=0
PROB_WRITE=100
USE_TSX=1

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.02
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.04
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.12
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.30
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.40
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.50
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.60
		### TODO: larger batches
		###
		mv Bank.csv $DATA_FOLDER/${1}_w${PROB_WRITE}_s${s}
	done
}

function doRunLargeDTST_CPU {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.60
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		# for i in `seq 1 7`
		for i in `seq 1 11`
		do
			cat /tmp/BankLastLine.csv >> Bank.csv
		done
		mv Bank.csv $DATA_FOLDER/${1}_w${PROB_WRITE}_s${s}
	done
}

DATASET=$L_DATASET
# DURATION=$DURATION_GPU


# GPU_BACKOFF=200000
GPU_BACKOFF=0

###########################################################################

############## CPU-only
make clean ; make \
	CMP_TYPE=COMPRESSED \
	LOG_TYPE=VERS \
	USE_TSX_IMPL=1 \
	GPUEn=0 \
	CPUEn=1 \
	INST_CPU=0 \
	DISABLE_RS=1 \
	DISABLE_WS=1 \
	PR_MAX_RWSET_SIZE=50 \
	BANK_PART=9 \
	BANK_INTRA_CONFL=0 \
	GPU_PART=0.55 \
	CPU_PART=0.55 \
	P_INTERSECT=0.00 \
	PROFILE=1 \
	BMAP_GRAN_BITS=13 \
	DISABLE_NON_BLOCKING=1 \
	OVERLAP_CPY_BACK=0 \
	LOG_SIZE=4096 \
	DISABLE_EARLY_VALIDATION=1 \
	STM_LOG_BUFFER_SIZE=256 \
	BANK_PART_SCALE=1 -j >/dev/null
### 90% writes
doRunLargeDTST CPUonly

############## GPU-only
make clean ; make \
	CMP_TYPE=COMPRESSED \
	LOG_TYPE=VERS \
	USE_TSX_IMPL=1 \
	GPUEn=1 \
	CPUEn=0 \
	INST_CPU=0 \
	DISABLE_RS=1 \
	DISABLE_WS=1 \
	PR_MAX_RWSET_SIZE=50 \
	BANK_PART=9 \
	BANK_INTRA_CONFL=0 \
	GPU_PART=0.55 \
	CPU_PART=0.55 \
	P_INTERSECT=0.00 \
	PROFILE=1 \
	BMAP_GRAN_BITS=13 \
	DISABLE_NON_BLOCKING=1 \
	OVERLAP_CPY_BACK=0 \
	LOG_SIZE=4096 \
	DISABLE_EARLY_VALIDATION=1 \
	STM_LOG_BUFFER_SIZE=256 \
	BANK_PART_SCALE=1 -j >/dev/null
### 90% writes
doRunLargeDTST GPUonly

# ############## VERS
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 \
# 	LOG_SIZE=4096 DISABLE_EARLY_VALIDATION=1 STM_LOG_BUFFER_SIZE=256 BANK_PART_SCALE=1 >/dev/null
# ### 90% writes
# doRunLargeDTST SHeTM_basic

# ############## VERS
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
# 	LOG_SIZE=4096 DISABLE_EARLY_VALIDATION=0 STM_LOG_BUFFER_SIZE=256 BANK_PART_SCALE=1 >/dev/null
# ### 90% writes
# doRunLargeDTST SHeTM_opt

mkdir -p data/no_conflicts_inc_duration
mv data/*_s* data/no_conflicts_inc_duration/
