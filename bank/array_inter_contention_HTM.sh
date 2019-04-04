#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=10000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=3
#./makeTM.sh

rm -f Bank.csv
rm -f File1.csv
rm -f File50.csv
rm -f File10.csv
rm -f File90.csv
rm -f File100.csv

CPU_BACKOFF=35
GPU_BACKOFF=2000
TX_SIZE=4
# sleep 2h

function actualRun {
	timeout 40s ./bank -n $CPU_THREADS -b 160 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 160 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 160 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 160 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 160 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
	fi
}

function doRunLargeDTST_GPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=1 \
			OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1  File1
		# actualRun 50 File50

		# all the same point (no inter-confl with 1 device)
		tail -n 1 File10.csv > /tmp/auxFile10.csv
		tail -n 1 File90.csv > /tmp/auxFile90.csv
		# tail -n 1 File1.csv  > /tmp/auxFile1.csv
		# tail -n 1 File50.csv > /tmp/auxFile50.csv
###
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
###
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
###
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
###
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv

		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}

		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_CPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=1 \
			OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		tail -n 1 File10.csv > /tmp/auxFile10.csv
		tail -n 1 File90.csv > /tmp/auxFile90.csv
		# tail -n 1 File1.csv  > /tmp/auxFile1.csv
		# tail -n 1 File50.csv > /tmp/auxFile50.csv

		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		###
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		cat /tmp/auxFile10.csv >> File10.csv
		###
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		###
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		cat /tmp/auxFile90.csv >> File90.csv
		###

		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}

		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_VERS {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_VERS_OVER {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_VERS_BLOC {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 10 File10
		actualRun 90 File90
		# actualRun 1 File1
		# actualRun 50 File50
		###
		mv File10.csv ${1}_10_s${s}
		mv File90.csv ${1}_90_s${s}
		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

# function doRunLargeDTST_BMAP {
# 	# Seq. access, 18 items, prob. write {5..95}, writes 1%
# 	for s in `seq $SAMPLES`
# 	do
# 		make clean ; make \
# 			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
# 		# timeout 40s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T 32 -f File10 CPU_BACKOFF=180
# 		# timeout 40s ./bank -n $CPU_THREADS -b 160 -x 256 -a $DATASET -d $DURATION -R 0 -S 16 -l 90 -N 1 -T 32 -f File100 CPU_BACKOFF=180
# 		actualRun 10 File10
# 		actualRun 100 File100
# 		# actualRun 1 File1
# 		# actualRun 50 File50
# 		###
# 		make clean ; make \
# 			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 \
# 			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
# 		actualRun 10 File10
# 		actualRun 100 File100
# 		# actualRun 1 File1
# 		# actualRun 50 File50
# 		###
# 		make clean ; make \
# 			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 \
# 			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
# 		actualRun 10 File10
# 		actualRun 100 File100
# 		# actualRun 1 File1
# 		# actualRun 50 File50
# 		###
# 		make clean ; make \
# 			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 \
# 			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
# 		actualRun 10 File10
# 		actualRun 100 File100
# 		# actualRun 1 File1
# 		# actualRun 50 File50
# 		###
# 		make clean ; make \
# 			CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
# 			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
# 		actualRun 10 File10
# 		actualRun 100 File100
# 		# actualRun 1 File1
# 		# actualRun 50 File50
# 		###
# 		mv File10.csv ${1}_10_s${s}
# 		mv File100.csv ${1}_100_s${s}
# 		# mv File1.csv ${1}_1_s${s}
# 		# mv File50.csv ${1}_50_s${s}
# 	done
# }

### Fixed the amount of CPU threads
CPU_THREADS=10

############### LARGE
###########################################################################
### 600MB
DATASET=150000000
############### GPU-only
doRunLargeDTST_GPUonly inter_GPUonly_rand_sep 9

############## CPU-only
doRunLargeDTST_CPUonly inter_CPUonly_rand_sep 9

############## VERS
doRunLargeDTST_VERS inter_VERS_rand_sep 9

############## VERS_OVER
doRunLargeDTST_VERS_OVER inter_VERS_OVER_rand_sep 9

############## VERS_BLOC
doRunLargeDTST_VERS_BLOC inter_VERS_BLOC_rand_sep 9

############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_rand_sep 1
###########################################################################

############### SMALL
###########################################################################
### 60MB
DATASET=15000000
# ############### GPU-only
doRunLargeDTST_GPUonly inter_GPUonly_rand_sep_SMALL 9

############## CPU-only
doRunLargeDTST_CPUonly inter_CPUonly_rand_sep_SMALL 9

############## VERS
doRunLargeDTST_VERS inter_VERS_rand_sep_SMALL 9

############## VERS_OVER
doRunLargeDTST_VERS_OVER inter_VERS_OVER_rand_sep_SMALL 9

############## VERS_BLOC
doRunLargeDTST_VERS_BLOC inter_VERS_BLOC_rand_sep_SMALL 9

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
