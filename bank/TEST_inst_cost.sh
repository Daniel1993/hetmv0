#!/bin/bash

SAMPLES=5
DURATION=12000
#./makeTM.sh
# DURATION_ORG=12000
# DURATION_GPU=6000
# DURATION=$DURATION_ORG

DATA_FOLDER=$(pwd)/data
# mkdir -p $DATA_FOLDER
# cd ../../bank
# rm -f Bank.csv

mkdir -p data/

rm Bank.csv

CPU_THREADS=8
GPU_BLOCKS=80
GPU_THREADS=256

function run_may_fail {
  echo "timeout 40s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET \
		-d $DURATION -R 0 -S 4 -l 90 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0 -X 0.08"
	timeout 40s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x $GPU_THREADS -a $DATASET \
		-d $DURATION -R 0 -S 4 -l 90 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0 -X 0.08 >/dev/null
}

function actualRun {
	run_may_fail
	if [ $? -ne 0 ]
	then
		run_may_fail
	fi
	if [ $? -ne 0 ]
	then
		run_may_fail
	fi
	if [ $? -ne 0 ]
	then
		run_may_fail
	fi
	if [ $? -ne 0 ]
	then
		run_may_fail
	fi
}

function doRun_CPU {
	make clean ; make \
    CMP_TYPE=COMPRESSED \
    CPUEn=1 \
    GPUEn=0 \
    INST_CPU=$INST_CPU \
    LOG_TYPE=VERS \
    USE_TSX_IMPL=$USE_TSX \
    PR_MAX_RWSET_SIZE=50 \
    BANK_PART=9 \
    BANK_INTRA_CONFL=0 \
    GPU_PART=0.55 \
    CPU_PART=0.55 \
    P_INTERSECT=0.00 \
    PROFILE=1 \
    BMAP_GRAN_BITS=13 \
    DISABLE_NON_BLOCKING=0 \
    OVERLAP_CPY_BACK=1 \
    LOG_SIZE=4096 \
    DISABLE_EARLY_VALIDATION=0 \
    STM_LOG_BUFFER_SIZE=256 \
    BANK_PART_SCALE=1 -j >/dev/null
  for s in `seq 1 $SAMPLES`
  do
    actualRun
  done
  mv Bank.csv "TSX${USE_TSX}_INST_CPU${INST_CPU}_${CPU_THREADS}THR_${DATASET}.csv"
}
function doRun_GPU {
	make clean ; make \
    CMP_TYPE=COMPRESSED \
    CPUEn=0 \
    GPUEn=1 \
    INST_CPU=1 \
    DISABLE_RS=$DISABLE_INST_GPU \
    DISABLE_WS=$DISABLE_INST_GPU \
    LOG_TYPE=VERS \
    USE_TSX_IMPL=0 \
    PR_MAX_RWSET_SIZE=50 \
    BANK_PART=9 \
    BANK_INTRA_CONFL=0 \
    GPU_PART=0.55 \
    CPU_PART=0.55 \
    P_INTERSECT=0.00 \
    PROFILE=1 \
    BMAP_GRAN_BITS=13 \
    DISABLE_NON_BLOCKING=0 \
    OVERLAP_CPY_BACK=1 \
    LOG_SIZE=4096 \
    DISABLE_EARLY_VALIDATION=0 \
    STM_LOG_BUFFER_SIZE=256 \
    BANK_PART_SCALE=1 -j >/dev/null
  for s in `seq 1 $SAMPLES`
  do
    actualRun
  done
  mv Bank.csv "DISABLE_INST_GPU${DISABLE_INST_GPU}_${GPU_BLOCKS}B${GPU_THREADS}T_${DATASET}.csv"
}
function doRun_BOTH {
  if [ $INST == 1 ]
  then
    DISABLE_INST_GPU=0
    INST_CPU=1
  else
    DISABLE_INST_GPU=1
    INST_CPU=0
  fi
	make clean ; make \
    CMP_TYPE=COMPRESSED \
    CPUEn=1 \
    GPUEn=1 \
    INST_CPU=$INST_CPU \
    DISABLE_RS=$DISABLE_INST_GPU \
    DISABLE_WS=$DISABLE_INST_GPU \
    LOG_TYPE=VERS \
    USE_TSX_IMPL=$USE_TSX \
    PR_MAX_RWSET_SIZE=50 \
    BANK_PART=9 \
    BANK_INTRA_CONFL=0 \
    GPU_PART=0.55 \
    CPU_PART=0.55 \
    P_INTERSECT=0.00 \
    PROFILE=1 \
    BMAP_GRAN_BITS=13 \
    DISABLE_NON_BLOCKING=1 \
    DISABLE_EARLY_VALIDATION=1 \
    OVERLAP_CPY_BACK=0 \
    LOG_SIZE=4096 \
    STM_LOG_BUFFER_SIZE=256 \
    BANK_PART_SCALE=1 -j >/dev/null
  CPU_THREADS=8
  GPU_BLOCKS=80
  GPU_THREADS=256
  for s in `seq 1 $SAMPLES`
  do
    actualRun
  done
  mv Bank.csv "BOTH_INST${INST}_TSX${USE_TSX}_${CPU_THREADS}THR_${GPU_BLOCKS}B${GPU_THREADS}T_${DATASET}.csv"
}

# for USE_TSX in 0 1
# do
#   for CPU_THREADS in 1 8
#   do
#     for DATASET in 1000000 150000000
#     do
#       for INST_CPU in 1 0
#       do
#         doRun_CPU
#       done
#     done
#   done
# done

# mkdir -p data/instrument
# mv TSX0_INST_CPU1_1THR_1000000.csv data/instrument/STM_4MB_1THR_INST.csv
# mv TSX0_INST_CPU0_1THR_1000000.csv data/instrument/STM_4MB_1THR_NO_INST.csv
# mv TSX0_INST_CPU1_1THR_150000000.csv data/instrument/STM_600MB_1THR_INST.csv
# mv TSX0_INST_CPU0_1THR_150000000.csv data/instrument/STM_600MB_1THR_NO_INST.csv
# mv TSX1_INST_CPU1_1THR_1000000.csv data/instrument/HTM_4MB_1THR_INST.csv
# mv TSX1_INST_CPU0_1THR_1000000.csv data/instrument/HTM_4MB_1THR_NO_INST.csv
# mv TSX1_INST_CPU1_1THR_150000000.csv data/instrument/HTM_600MB_1THR_INST.csv
# mv TSX1_INST_CPU0_1THR_150000000.csv data/instrument/HTM_600MB_1THR_NO_INST.csv

# mv TSX0_INST_CPU1_8THR_1000000.csv data/instrument/STM_4MB_8THR_INST.csv
# mv TSX0_INST_CPU0_8THR_1000000.csv data/instrument/STM_4MB_8THR_NO_INST.csv
# mv TSX0_INST_CPU1_8THR_150000000.csv data/instrument/STM_600MB_8THR_INST.csv
# mv TSX0_INST_CPU0_8THR_150000000.csv data/instrument/STM_600MB_8THR_NO_INST.csv
# mv TSX1_INST_CPU1_8THR_1000000.csv data/instrument/HTM_4MB_8THR_INST.csv
# mv TSX1_INST_CPU0_8THR_1000000.csv data/instrument/HTM_4MB_8THR_NO_INST.csv
# mv TSX1_INST_CPU1_8THR_150000000.csv data/instrument/HTM_600MB_8THR_INST.csv
# mv TSX1_INST_CPU0_8THR_150000000.csv data/instrument/HTM_600MB_8THR_NO_INST.csv

# for GPU_BLOCKS in 20 80
for GPU_BLOCKS in 80
do
  for DATASET in 1000000 150000000
  do
    for DISABLE_INST_GPU in 1 0
    do
      doRun_GPU
    done
  done
done

mkdir -p data/instrument
mv DISABLE_INST_GPU1_20B256T_1000000.csv   data/instrument/GPU_4MB_20B256T_INST.csv
mv DISABLE_INST_GPU0_20B256T_1000000.csv   data/instrument/GPU_4MB_20B256T_NO_INST.csv
mv DISABLE_INST_GPU1_20B256T_150000000.csv data/instrument/GPU_600MB_20B256T_INST.csv
mv DISABLE_INST_GPU0_20B256T_150000000.csv data/instrument/GPU_600MB_20B256T_NO_INST.csv
mv DISABLE_INST_GPU1_20B256T_1000000.csv   data/instrument/GPU_4MB_20B256T_INST.csv
mv DISABLE_INST_GPU0_20B256T_1000000.csv   data/instrument/GPU_4MB_20B256T_NO_INST.csv
mv DISABLE_INST_GPU1_20B256T_150000000.csv data/instrument/GPU_600MB_20B256T_INST.csv
mv DISABLE_INST_GPU0_20B256T_150000000.csv data/instrument/GPU_600MB_20B256T_NO_INST.csv

mv DISABLE_INST_GPU1_80B256T_1000000.csv   data/instrument/GPU_4MB_80B256T_INST.csv
mv DISABLE_INST_GPU0_80B256T_1000000.csv   data/instrument/GPU_4MB_80B256T_NO_INST.csv
mv DISABLE_INST_GPU1_80B256T_150000000.csv data/instrument/GPU_600MB_80B256T_INST.csv
mv DISABLE_INST_GPU0_80B256T_150000000.csv data/instrument/GPU_600MB_80B256T_NO_INST.csv
mv DISABLE_INST_GPU1_80B256T_1000000.csv   data/instrument/GPU_4MB_80B256T_INST.csv
mv DISABLE_INST_GPU0_80B256T_1000000.csv   data/instrument/GPU_4MB_80B256T_NO_INST.csv
mv DISABLE_INST_GPU1_80B256T_150000000.csv data/instrument/GPU_600MB_80B256T_INST.csv
mv DISABLE_INST_GPU0_80B256T_150000000.csv data/instrument/GPU_600MB_80B256T_NO_INST.csv

USE_TSX=1
for DATASET in 1000000 150000000
do
  for INST in 1 0
  do
    doRun_BOTH
  done
done

mkdir -p data/instrument
mv BOTH_INST0_TSX1_8THR_80B256T_1000000.csv   data/instrument/BOTH_4MB_NO_INST.csv
mv BOTH_INST1_TSX1_8THR_80B256T_1000000.csv   data/instrument/BOTH_4MB_INST.csv
mv BOTH_INST0_TSX1_8THR_80B256T_150000000.csv data/instrument/BOTH_600MB_NO_INST.csv
mv BOTH_INST1_TSX1_8THR_80B256T_150000000.csv data/instrument/BOTH_600MB_INST.csv

