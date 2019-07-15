#!/bin/bash

FOLDER="../data_ex1/" # "`ls -d */ | tail -n 1`"

./normalize_CPU.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED.avg \
  ${FOLDER}VERS_rand_sep.avg \
  ${FOLDER}NORM_inst_rand_sep_TSX.avg

./normalize_CPU.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED_STM.avg \
  ${FOLDER}VERS_rand_sep_STM.avg \
  ${FOLDER}NORM_inst_rand_sep_STM.avg

./normalize_GPU.R \
  ${FOLDER}GPUonly_rand_sep_DISABLED.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP_NO_RS.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP_NO_WS.avg \
  ${FOLDER}NORM_inst_rand_sep_GPU.avg

./normalize_GPU.R \
  ${FOLDER}GPUonly_rand_sep_DISABLED_10b.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP_10b.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP_NO_RS_10b.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP_NO_WS_10b.avg \
  ${FOLDER}NORM_inst_rand_sep_GPU_10b.avg
