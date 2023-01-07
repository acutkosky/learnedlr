#!/bin/bash -l
# I am tired of cluttering up the directory with output files
# so, let's hide them in a runs/ directory.

mkdir -p runs/


RUNFILE=submit_jax.sh.`date +%s%N`

cp submit_jax.sh runs/$RUNFILE

echo "creating runs/$RUNFILE"

cd runs

qsub $RUNFILE $@
