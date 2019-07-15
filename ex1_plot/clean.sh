#!/bin/bash

FOLDER=`ls -d */ | tail -n 1`

rm $FOLDER/*.tsv $FOLDER/*.avg 

