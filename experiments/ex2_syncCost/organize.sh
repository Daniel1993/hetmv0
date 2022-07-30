#!/bin/bash

FOLDER=$(ls -d */ | tail -n 1)
PROC_FOLDER=0proc$FOLDER

mkdir -p $PROC_FOLDER
mv $FOLDER/*.avg $PROC_FOLDER


