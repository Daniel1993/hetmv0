#!/bin/bash

FOLDER=../data_ex1 #`ls -d */ | tail -n 1`

for f in $FOLDER/*
do
	paste -d ";" $f prec_write.txt > $f.tmp
	mv $f.tmp $f
done
