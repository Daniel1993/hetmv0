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

CPU_THREADS=4
# LOW_CPU_THREADS=2
LARGE_HIGH_CPU_THREADS=4
LARGE_VERY_HIGH_CPU_THREADS=14
SMALL_HIGH_CPU_THREADS=4
SMALL_VERY_HIGH_CPU_THREADS=14

rm -f Bank.csv GPUload.csv CPUload.csv

# LARGE_DATASET=46656
# SMALL_DATASET=7776
# VSMALL_DATASET=1296
#LARGE_DATASET=279936
# LARGE_DATASET=1679616
#VSMALL_DATASET=46656
# VSMALL_DATASET=7776
#VSMALL_DATASET=7776

LARGE_DATASET=1000000
VSMALL_DATASET=100000

#GPU_BLOCKS=512
GPU_BLOCKS=320

GPU_THREADS=256

CPU_THREADS=14

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"

if [ ! -f $GPU_INPUT ]
then
	cd java_zipf
	java Main $SIZE_ZIPF 0.99 25165824 > ../$GPU_INPUT
	cd ..
fi
if [ ! -f $CPU_INPUT ]
then
	cd java_zipf
	java Main $SIZE_ZIPF 0.99 1310720 > ../$CPU_INPUT
	cd ..
fi
#
# SIZE_ZIPF=80000000
# GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
# CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"
#
# if [ ! -f $GPU_INPUT ]
# then
# 	cd java_zipf
# 	java Main $SIZE_ZIPF 0.99 25165824 > ../$GPU_INPUT
# 	cd ..
# fi
# if [ ! -f $CPU_INPUT ]
# then
# 	cd java_zipf
# 	java Main $SIZE_ZIPF 0.99 1310720 > ../$CPU_INPUT
# 	cd ..
# fi

# memcached read only

### TODO: use shared 0, 30, 50

### TODO: find the best combination of parameters
function doRun_HeTM {
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD
	for s in `seq 1 $SAMPLES`
	do
		### DO GPU load
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.5 REQUEST_CPU=0.5
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		cp GPUload.csv CPUload.csv
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.6 REQUEST_CPU=0.4
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.7 REQUEST_CPU=0.3
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.8 REQUEST_CPU=0.2
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.9 REQUEST_CPU=0.1
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=1.0 REQUEST_CPU=0.0
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		### DO CPU load
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.6 REQUEST_GPU=0.4
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.7 REQUEST_GPU=0.3
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.8 REQUEST_GPU=0.2
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.9 REQUEST_GPU=0.1
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=1.0 REQUEST_GPU=0.0
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		###
		mv GPUload.csv ${1}_GPUload_s${s}
		mv CPUload.csv ${1}_CPUload_s${s}
	done
}

# reduces the load in one device, it goes into the shared queue
function doRun_HeTM_SHARED_noConfl {
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		### DO GPU load
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.5 REQUEST_CPU=0.5 >/dev/null
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		cp GPUload.csv CPUload.csv
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.5 REQUEST_CPU=0.4 >/dev/null
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.5 REQUEST_CPU=0.3 >/dev/null
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.5 REQUEST_CPU=0.2
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.5 REQUEST_CPU=0.1
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_GPU=0.5 REQUEST_CPU=0.0
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		### DO CPU load
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.5 REQUEST_GPU=0.4
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.5 REQUEST_GPU=0.3
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.5 REQUEST_GPU=0.2
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.5 REQUEST_GPU=0.1
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		rm memcd *.o src/*.o ; make simple CMP_TYPE=COMPRESSED BANK_PART=1 \
			PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD \
			REQUEST_GRANULARITY=$REQUEST_GRANULARITY REQUEST_CPU=0.5 REQUEST_GPU=0.0
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f CPUload CPU_BACKOFF=200
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		if [ $? -ne 0 ]
		then
			timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
			-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f GPUload CPU_BACKOFF=200
		fi
		###
		mv GPUload.csv ${1}_GPUload_s${s}
		mv CPUload.csv ${1}_CPUload_s${s}
	done
}

function actualRun {
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=200 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3}
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=200 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=200 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=200 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=200 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=200 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=200 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=200 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3}
	fi
}

### TODO: no longer use REQUEST_GRANULARITY
# --> prob of stealing (at the begining of the batch)
#    CPU steal 10% of batches, 20%, 30%, 50%, 80% 100%
#    GPU steal 10% of batches, 20%, 30%, 50%, 80% 100%

# SETs (0.1%)/GETs(99.9%) (DELETEs next paper)
# -------------------
# CPU --> do round-robin
# GPU --> do the same --> the change is complecated

# reduces the load in one device, it goes into the shared queue
function doRun_HeTM_SHARED_steal {
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 \
		BENCH=MEMCD CPU_STEAL_ONLY_GETS=0 >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		# ### DO GPU load
		actualRun GPUload 0.0 0.0
		cp GPUload.csv CPUload.csv
		cp GPUload.csv CPUload_GETs.csv
		actualRun GPUload 0.0 0.1
		actualRun GPUload 0.0 0.2
		actualRun GPUload 0.0 0.3
		actualRun GPUload 0.0 0.4
		# actualRun GPUload 0.0 0.5
		actualRun GPUload 0.0 0.6
		# actualRun GPUload 0.0 0.7
		actualRun GPUload 0.0 0.8
		# actualRun GPUload 0.0 0.9
		actualRun GPUload 0.0 1.0
		### DO CPU load
		actualRun CPUload 0.1 0.0
		actualRun CPUload 0.2 0.0
		actualRun CPUload 0.3 0.0
		actualRun CPUload 0.4 0.0
		# actualRun CPUload 0.5 0.0
		actualRun CPUload 0.6 0.0
		# actualRun CPUload 0.7 0.0
		actualRun CPUload 0.8 0.0
		# actualRun CPUload 0.9 0.0
		actualRun CPUload 1.0 0.0
		###
		mv GPUload.csv ${1}_GPUload_s${s}
		# # rm GPUload.csv
		mv CPUload.csv ${1}_CPUload_s${s}
	done
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 \
		BENCH=MEMCD CPU_STEAL_ONLY_GETS=1 >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		### DO CPU load
		actualRun CPUload_GETs 0.1 0.0
		actualRun CPUload_GETs 0.2 0.0
		actualRun CPUload_GETs 0.3 0.0
		actualRun CPUload_GETs 0.4 0.0
		# actualRun CPUload 0.5 0.0
		actualRun CPUload_GETs 0.6 0.0
		# actualRun CPUload 0.7 0.0
		actualRun CPUload_GETs 0.8 0.0
		# actualRun CPUload 0.9 0.0
		actualRun CPUload_GETs 1.0 0.0
		###
		mv CPUload_GETs.csv ${1}_CPUload_GETs_s${s}
	done
}

function doRun_GPUonly {
	# TODO: for some reason DISABLED is slower...
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 CPUEn=0 PROFILE=1 \
		PR_MAX_RWSET_SIZE=20 -j 14 BENCH=MEMCD >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		### DO GPU load
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT CPU_BACKOFF=200
		tail -N 0.2 Bank.csv > /tmp/BankLastLine.csv
		cat /tmp/BankLastLine.csv >> Bank.csv # duplicates last line
		mv Bank.csv ${1}_s${s}
	done
}

function doRun_CPUonly {
	make clean ; make CMP_TYPE=DISABLED BANK_PART=1 USE_TSX_IMPL=1 INST_CPU=0 GPUEn=0 \
		PROFILE=1 PR_MAX_RWSET_SIZE=20 -j 14 BENCH=MEMCD >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		### DO GPU load
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.2 -S 0 -G $GPU_INPUT -C $CPU_INPUT CPU_BACKOFF=200
		tail -N 0.2 Bank.csv > /tmp/BankLastLine.csv
		cat /tmp/BankLastLine.csv >> Bank.csv # duplicates last line
		mv Bank.csv ${1}_s${s}
	done
}
#
# # DATASET=$SMALL_DATASET #medium
# #
# # doRun_HeTM_SHARED_noConfl memcd_SHARED_noConfl_VERS VERS
# # doRun_HeTM_SHARED_noConfl memcd_SHARED_noConfl_BMAP BMAP
# #
# # doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS VERS
# # doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP BMAP
# #
# # doRun_HeTM memcd_VERS VERS
# # doRun_HeTM memcd_BMAP BMAP
# #
# # doRun_GPUonly memcd_GPUonly
# # doRun_CPUonly memcd_CPUonly

DATASET=$LARGE_DATASET

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"

# ### 64 k
#
# GPU_BLOCKS=128
# GPU_THREADS=128
# # GPU_THREADS=256
# REQUEST_GRANULARITY=280000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_64k VERS 0.38 0.62
# REQUEST_GRANULARITY=220000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_64k BMAP 0.49 0.51
#
# REQUEST_GRANULARITY=400000
# doRun_GPUonly memcd_GPUonly_LARGE_B_64k 1.0 0.0
# doRun_CPUonly memcd_CPUonly_LARGE_B_64k 0.0 1.0

### 128 k

GPU_BLOCKS=128
GPU_THREADS=256
# GPU_THREADS=256
REQUEST_GRANULARITY=320000
doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_128k VERS 0.41 0.59
REQUEST_GRANULARITY=270000
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_128k BMAP 0.49 0.51

REQUEST_GRANULARITY=400000
doRun_GPUonly memcd_GPUonly_LARGE_B_128k 1.0 0.0
doRun_CPUonly memcd_CPUonly_LARGE_B_128k 0.0 1.0

# ### 256 k
#
# GPU_BLOCKS=256
# GPU_THREADS=256
# # GPU_THREADS=256
# REQUEST_GRANULARITY=340000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_256k VERS 0.43 0.57
# REQUEST_GRANULARITY=300000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_256k BMAP 0.51 0.49
#
# REQUEST_GRANULARITY=400000
# doRun_GPUonly memcd_GPUonly_LARGE_B_256k 1.0 0.0
# doRun_CPUonly memcd_CPUonly_LARGE_B_256k 0.0 1.0

### 512 k

GPU_BLOCKS=512
GPU_THREADS=256
# GPU_THREADS=256
doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_512k VERS
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_512k BMAP

REQUEST_GRANULARITY=400000
doRun_GPUonly memcd_GPUonly_LARGE_B_512k
doRun_CPUonly memcd_CPUonly_LARGE_B_512k

# ### 1 M
#
# GPU_BLOCKS=512
# GPU_THREADS=512
# # GPU_THREADS=256
# REQUEST_GRANULARITY=370000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_1M VERS 0.42 0.58
# REQUEST_GRANULARITY=390000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_1M BMAP 0.47 0.53
#
# REQUEST_GRANULARITY=400000
# doRun_GPUonly memcd_GPUonly_LARGE_B_1M 1.0 0.0
# doRun_CPUonly memcd_CPUonly_LARGE_B_1M 0.0 1.0


# # GPU_THREADS=512
# # doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_Lk_Lb VERS
# # doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_Lk_Lb BMAP
# #
# # doRun_GPUonly memcd_GPUonly_LARGE_Lk_Lb
# # doRun_CPUonly memcd_CPUonly_LARGE_Lk_Lb
#
# # doRun_HeTM memcd_VERS_LARGE VERS
# # doRun_HeTM memcd_BMAP_LARGE BMAP
#
# #################################

DATASET=$VSMALL_DATASET
#GPU_BLOCKS=256
# GPU_BLOCKS=512

# doRun_HeTM_SHARED_noConfl memcd_SHARED_noConfl_VERS_SMALL VERS
# doRun_HeTM_SHARED_noConfl memcd_SHARED_noConfl_BMAP_SMALL BMAP

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"
#
# ### 64 k
#
# GPU_THREADS=128
# GPU_BLOCKS=128
# # GPU_THREADS=256
# REQUEST_GRANULARITY=310000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_64k VERS 0.41 0.59
# REQUEST_GRANULARITY=270000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_64k BMAP 0.55 0.45
#
# REQUEST_GRANULARITY=400000
# doRun_GPUonly memcd_GPUonly_SMALL_B_64k 1.0 0.0
# doRun_CPUonly memcd_CPUonly_SMALL_B_64k 0.0 1.0

### 128 k

GPU_THREADS=128
GPU_BLOCKS=256
# GPU_THREADS=256
REQUEST_GRANULARITY=340000
doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_128k VERS 0.42 0.58
REQUEST_GRANULARITY=340000
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_128k BMAP 0.54 0.46

REQUEST_GRANULARITY=400000
doRun_GPUonly memcd_GPUonly_SMALL_B_128k 1.0 0.0
doRun_CPUonly memcd_CPUonly_SMALL_B_128k 0.0 1.0

# ### 256 k
#
# GPU_THREADS=256
# GPU_BLOCKS=256
# # GPU_THREADS=256
# REQUEST_GRANULARITY=400000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_256k VERS 0.42 0.58
# REQUEST_GRANULARITY=370000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_256k BMAP 0.54 0.46
#
# REQUEST_GRANULARITY=400000
# doRun_GPUonly memcd_GPUonly_SMALL_B_256k 1.0 0.0
# doRun_CPUonly memcd_CPUonly_SMALL_B_256k 0.0 1.0

### 512 k

GPU_THREADS=512
GPU_BLOCKS=256
# GPU_THREADS=256
REQUEST_GRANULARITY=380000
doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_512k VERS 0.43 0.57
REQUEST_GRANULARITY=410000
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_512k BMAP 0.46 0.54

REQUEST_GRANULARITY=400000
doRun_GPUonly memcd_GPUonly_SMALL_B_512k 1.0 0.0
doRun_CPUonly memcd_CPUonly_SMALL_B_512k 0.0 1.0

# ### 1 M
#
# GPU_THREADS=512
# GPU_BLOCKS=256
# # GPU_THREADS=256
# REQUEST_GRANULARITY=380000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_1M VERS 0.43 0.57
# REQUEST_GRANULARITY=410000
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_1M BMAP 0.46 0.54
#
# REQUEST_GRANULARITY=400000
# doRun_GPUonly memcd_GPUonly_SMALL_B_1M 1.0 0.0
# doRun_CPUonly memcd_CPUonly_SMALL_B_1M 0.0 1.0
