#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=5
#./makeTM.sh

rm -f Bank.csv

DURATION=12000
DATASET=150000000 #250000000
PROB_WRITE=10 # test also with 50 and 10
# CPU_BACKOFF=250
CPU_BACKOFF=35
GPU_BACKOFF=2000

CPU_THREADS=8
# TRANSACTION_SIZE=16 # default
TRANSACTION_SIZE=4

function doRunLargeDTST_1 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.01
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.04
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.12
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.60
		### TODO: larger batches
		###
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_100 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.01
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.04
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.12
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.60
		### TODO: larger batches
		###
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_CPU_1 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF -X 0.3
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		# for i in `seq 1 7`
		for i in `seq 1 5`
		do
			cat /tmp/BankLastLine.csv >> Bank.csv
		done
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_CPU_100 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 1 CPU_BACKOFF=$CPU_BACKOFF -X 0.3
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		# for i in `seq 1 7`
		for i in `seq 1 5`
		do
			cat /tmp/BankLastLine.csv >> Bank.csv
		done
		mv Bank.csv ${1}${s}
	done
}

### Fixed the amount of CPU threads
CPU_THREADS=10
CPU_BACKOFF=35

###########################################################################
############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 >/dev/null
doRunLargeDTST_1 GPUonly_rand_sep_DISABLED_1_s
doRunLargeDTST_100 GPUonly_rand_sep_DISABLED_100_s

make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 OVERLAP_CPY_BACK=1 >/dev/null
# doRunLargeDTST_1 GPUonly_rand_sep_DISABLED_OVERLAP_1_s
# doRunLargeDTST_100 GPUonly_rand_sep_DISABLED_OVERLAP_100_s
### it is basically flat
doRunLargeDTST_1 GPUonly_rand_sep_DISABLED_OVERLAP_1_s
doRunLargeDTST_100 GPUonly_rand_sep_DISABLED_OVERLAP_100_s

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 >/dev/null
doRunLargeDTST_CPU_1 CPUonly_rand_sep_DISABLED_1_s
doRunLargeDTST_CPU_100 CPUonly_rand_sep_DISABLED_100_s

# LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 \
	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
doRunLargeDTST_1 VERS_BLOC_rand_sep_1_s
doRunLargeDTST_100 VERS_BLOC_rand_sep_100_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 \
	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
doRunLargeDTST_1 VERS_NON_BLOC_rand_sep_1_s
doRunLargeDTST_100 VERS_NON_BLOC_rand_sep_100_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
doRunLargeDTST_1 VERS_NON_BLOC_OVER_rand_sep_1_s
doRunLargeDTST_100 VERS_NON_BLOC_OVER_rand_sep_100_s

mkdir -p array_batch_duration_4
mv *_s* array_batch_duration_4/

###############################################################################
###############################################################################

#### TinySTM

CPU_BACKOFF=35
CPU_THREADS=10

############### GPU-only
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=0 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	DEFAULT_BITMAP_GRANULARITY_BITS=13 >/dev/null
# doRunLargeDTST_1 GPUonly_rand_sep_DISABLED_1_s
# doRunLargeDTST_100 GPUonly_rand_sep_DISABLED_100_s
cp array_batch_duration_4/GPUonly_rand_sep_DISABLED_1_s* array_batch_duration_4_b/
cp array_batch_duration_4/GPUonly_rand_sep_DISABLED_100_s* array_batch_duration_4_b/

# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=0 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	DEFAULT_BITMAP_GRANULARITY_BITS=13 OVERLAP_CPY_BACK=1 >/dev/null
# doRunLargeDTST_1 GPUonly_rand_sep_DISABLED_OVERLAP_1_s
# doRunLargeDTST_100 GPUonly_rand_sep_DISABLED_OVERLAP_100_s
### it is basically flat
# doRunLargeDTST_1 GPUonly_rand_sep_DISABLED_OVERLAP_1_s
# doRunLargeDTST_100 GPUonly_rand_sep_DISABLED_OVERLAP_100_s
cp array_batch_duration_4/GPUonly_rand_sep_OVERLAP_DISABLED_1_s* array_batch_duration_4_b/
cp array_batch_duration_4/GPUonly_rand_sep_OVERLAP_DISABLED_100_s* array_batch_duration_4_b/

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 >/dev/null
doRunLargeDTST_CPU_1 CPUonly_rand_sep_DISABLED_1_s
doRunLargeDTST_CPU_100 CPUonly_rand_sep_DISABLED_100_s

# LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 \
	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
doRunLargeDTST_1 VERS_BLOC_rand_sep_1_s
doRunLargeDTST_100 VERS_BLOC_rand_sep_100_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 \
	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
doRunLargeDTST_1 VERS_NON_BLOC_rand_sep_1_s
doRunLargeDTST_100 VERS_NON_BLOC_rand_sep_100_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	DEFAULT_BITMAP_GRANULARITY_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
doRunLargeDTST_1 VERS_NON_BLOC_OVER_rand_sep_1_s
doRunLargeDTST_100 VERS_NON_BLOC_OVER_rand_sep_100_s

mkdir -p array_batch_duration_4_b
mv *_s* array_batch_duration_4_b/
###############################################################################
