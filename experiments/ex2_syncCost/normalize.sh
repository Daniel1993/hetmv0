#!/bin/bash

DATA_FOLDER=~/Documents/Data/HeTM_replicate_PACT/ex2_syncCost
FOLDER=$(ls -d $DATA_FOLDER/0proc*/ | tail -n 1)
SCRIPTS=../aux_files

if [[ $# -gt 0 ]] ; then
	SCRIPTS=$1
fi

$SCRIPTS/../aux_files/normalize_theo.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED_large_w100.avg \
  ${FOLDER}GPUonly_rand_sep_DISABLED_large_w100.avg \
  ${FOLDER}BMAP_rand_sep_large_w100.avg \
  "X.37.DURATION_BATCH" \
  ${FOLDER}NORM_BMAP_rand_sep_large_w100

$SCRIPTS/../aux_files/normalize_theo.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED_large_w100.avg \
  ${FOLDER}GPUonly_rand_sep_DISABLED_large_w100.avg \
  ${FOLDER}VERS_BLOC_rand_sep_large_w100.avg \
  "X.37.DURATION_BATCH" \
  ${FOLDER}NORM_VERS_BLOC_rand_sep_large_w100
