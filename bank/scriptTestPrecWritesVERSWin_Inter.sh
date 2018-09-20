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
rm -f File10.csv
rm -f File90.csv

DATASET=100000000 #2621440 # 90 000 000 is the max for my home machine
# DATASET=20000000 #2621440 # 90 000 000 is the max for my home machine
DATASET_P20=61000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P20=122000 #2621440 # 90 000 000 is the max for my home machine

function doRunLargeDTST_GPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
			GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=${2} PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
			GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 BANK_PART=${2} PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
			GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 BANK_PART=${2} PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
			GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 BANK_PART=${2} PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
			GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 BANK_PART=${2} PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
	done
}

function doRunLargeDTST_CPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
				BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
				BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
				BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
				BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
				BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
	done
}

function doRunLargeDTST_VERS {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
	done
}

function doRunLargeDTST_BMAP {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		rm -f bank *.o src/*.o ; make simple CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File10
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 2 -G $GPU_INPUT -C $CPU_INPUT -f File90
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
	done
}

############### LARGE
###########################################################################
### Fixed the amount of CPU threads
CPU_THREADS=8
DATASET=100000000
############### GPU-only
doRunLargeDTST_GPUonly inter_GPUonly_rand_sep 1

############## CPU-only
doRunLargeDTST_CPUonly inter_CPUonly_rand_sep 1

############## VERS
doRunLargeDTST_VERS inter_VERS_rand_sep 1

############## BMAP
doRunLargeDTST_BMAP inter_BMAP_rand_sep 1
###########################################################################

############### SMALL
###########################################################################
CPU_THREADS=8
DATASET=10000000
############### GPU-only
doRunLargeDTST_GPUonly inter_GPUonly_rand_sep_SMALL 1

############## CPU-only
doRunLargeDTST_CPUonly inter_CPUonly_rand_sep_SMALL 1

############## VERS
doRunLargeDTST_VERS inter_VERS_rand_sep_SMALL 1

############## BMAP
doRunLargeDTST_BMAP inter_BMAP_rand_sep_SMALL 1
###########################################################################

################ Zipf

############### LARGE
###########################################################################
DATASET=49999999 # half the size
CPU_THREADS=4

cd java_zipf
GPU_INPUT="GPU_input_${DATASET}_099_4194304.txt"
CPU_INPUT="CPU_input_${DATASET}_099_57344.txt"

if [ ! -f ../$GPU_INPUT ]
then
	java Main $DATASET 0.99 4194304 > ../$GPU_INPUT
fi
if [ ! -f ../$CPU_INPUT ]
then
	java Main $DATASET 0.99 57344 > ../$CPU_INPUT
fi
cd ..

DATASET=100000000 # half the size

############### GPU-only
doRunLargeDTST_GPUonly inter_GPUonly_zipf_sep 4

############## CPU-only
doRunLargeDTST_CPUonly inter_CPUonly_zipf_sep 4

############## VERS
doRunLargeDTST_VERS inter_VERS_zipf_sep 4

############## BMAP
doRunLargeDTST_BMAP inter_BMAP_zipf_sep 4
###########################################################################

############### SMALL
###########################################################################
DATASET=4999999 # half the size
CPU_THREADS=4

cd java_zipf
GPU_INPUT="GPU_input_${DATASET}_099_4194304.txt"
CPU_INPUT="CPU_input_${DATASET}_099_57344.txt"

if [ ! -f ../$GPU_INPUT ]
then
	java Main $DATASET 0.99 4194304 > ../$GPU_INPUT
fi
if [ ! -f ../$CPU_INPUT ]
then
	java Main $DATASET 0.99 57344 > ../$CPU_INPUT
fi
cd ..

DATASET=10000000 # half the size

############### GPU-only
doRunLargeDTST_GPUonly inter_GPUonly_zipf_sep_SMALL 4

############## CPU-only
doRunLargeDTST_CPUonly inter_CPUonly_zipf_sep_SMALL 4

############## VERS
doRunLargeDTST_VERS inter_VERS_zipf_sep_SMALL 4

############## BMAP
doRunLargeDTST_BMAP inter_BMAP_zipf_sep_SMALL 4
###########################################################################
