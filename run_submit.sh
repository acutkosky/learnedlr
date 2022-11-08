#!/bin/bash -l
# I am tired of cluttering up the directory with output files
# so, let's hide them in a runs/ directory.

mkdir -p runs/


RUNFILE=submit.sh.`date +%N`

cp submit.sh runs/$RUNFILE

cd runs

qsub $RUNFILE $@
