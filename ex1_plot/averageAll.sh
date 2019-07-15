#!/bin/bash

SCRIPTS=../scripts

FOLDER=../data_ex1/ # `ls -d */ | tail -n 1`

for i in `ls ${FOLDER}*_s1.tsv`
do
	$SCRIPTS/SCRIPT_compute_AVG_ERR.R `ls $i | sed -e 's/s1/s*/g'`
	mv avg.txt `ls $i | sed -e 's/_s1.tsv/.avg/g'`
done
