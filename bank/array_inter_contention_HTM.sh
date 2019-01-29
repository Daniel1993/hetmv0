#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=20000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=2
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

# sleep 2h

function actualRun {
	timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
	if [ $? -ne 0 ]
	then
		timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION \
		-R 0 -S 16 -l ${1} -N 1 -T 32 -f ${2} CPU_BACKOFF=180
	fi
}

function doRunLargeDTST_GPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		# timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 32 -f File10 CPU_BACKOFF=180
		# timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 32 -f File90 CPU_BACKOFF=180

		# all the same point (no inter-confl with 1 device)
		tail -n 1 File10.csv > /tmp/auxFile10.csv
		tail -n 1 File90.csv > /tmp/auxFile90.csv
###
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
###
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv

		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
	done
}

function doRunLargeDTST_CPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
				BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		# timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 32 -f File10 CPU_BACKOFF=180
		# timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 32 -f File90 CPU_BACKOFF=180
		###
		tail -n 1 File10.csv > /tmp/auxFile10.csv
		tail -n 1 File90.csv > /tmp/auxFile90.csv

		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		###
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		###

		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
	done
}

function doRunLargeDTST_VERS {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		# timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 32 -f File10 CPU_BACKOFF=180
		# timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 32 -f File90 CPU_BACKOFF=180
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
	done
}

function doRunLargeDTST_BMAP {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		# timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 32 -f File10 CPU_BACKOFF=180
		# timeout 30s ./bank -n $CPU_THREADS -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 32 -f File90 CPU_BACKOFF=180
		actualRun 10 File10
		actualRun 90 File90
		###
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		###
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		###
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		###
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
		actualRun 10 File10
		actualRun 90 File90
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
	done
}

### Fixed the amount of CPU threads
CPU_THREADS=14

############### LARGE
###########################################################################
### 1GB
DATASET=250000000
############### GPU-only
# doRunLargeDTST_GPUonly inter_GPUonly_rand_sep 1

############## CPU-only
# doRunLargeDTST_CPUonly inter_CPUonly_rand_sep 1

############## VERS
# doRunLargeDTST_VERS inter_VERS_rand_sep 1

############## BMAP
doRunLargeDTST_BMAP inter_BMAP_rand_sep 1
###########################################################################

############### SMALL
###########################################################################
### 100MB
DATASET=25000000
############### GPU-only
# doRunLargeDTST_GPUonly inter_GPUonly_rand_sep_SMALL 1

############## CPU-only
# doRunLargeDTST_CPUonly inter_CPUonly_rand_sep_SMALL 1

############## VERS
# doRunLargeDTST_VERS inter_VERS_rand_sep_SMALL 1

############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_rand_sep_SMALL 1
###########################################################################
#
# ################ Zipf
#
# ############### LARGE
# ###########################################################################
# ### 1GB
# DATASET=250000000
#
# ############### GPU-only
# doRunLargeDTST_GPUonly inter_GPUonly_cont_sep 5
#
# ############## CPU-only
# doRunLargeDTST_CPUonly inter_CPUonly_cont_sep 5
#
# ############## VERS
# doRunLargeDTST_VERS inter_VERS_cont_sep 5
#
# ############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_cont_sep 5
# ###########################################################################
#
# ############### SMALL
# ###########################################################################
# ### 100MB
# DATASET=25000000
#
# ############### GPU-only
# doRunLargeDTST_GPUonly inter_GPUonly_cont_sep_SMALL 5
#
# ############## CPU-only
# doRunLargeDTST_CPUonly inter_CPUonly_cont_sep_SMALL 5
#
# ############## VERS
# doRunLargeDTST_VERS inter_VERS_cont_sep_SMALL 5
#
# ############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_cont_sep_SMALL 5
# ###########################################################################

mkdir -p array_inter_HTM
mv *_s* array_inter_HTM
