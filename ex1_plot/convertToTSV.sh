#!/bin/bash

FOLDER=../data_ex1/ # `ls -d */ | tail -n 1`

for f in ${FOLDER}*_s*
do
	sed "s/;/	/g" $f | sed 1d > $f.tsv
done
