#!/bin/bash

SCRIPTS=../scripts

if [[ $# -gt 0 ]] ; then
	SCRIPTS=$1
fi

FOLDER=$(find . -maxdepth 1 -type d | sort | tail -n 1)

if [[ $# -gt 1 ]] ; then
	FOLDER=$2
fi


for i in $(ls ${FOLDER}/*_s1.tsv)
do
	$SCRIPTS/SCRIPT_compute_AVG_ERR.R $(ls $i | sed -e 's/s1/s*/g')
	mv avg.txt $(ls $i | sed -e 's/_s1.tsv/.avg/g')
done
