#!/bin/bash -l

# example usage:
# qsub submit.sh --config config/default.yaml

# Request 4 CPUs
#$ -pe omp 8

# Request 1 GPU
#$ -l gpus=1
#$ -l gpu_c=3.7
#$ -l gpu_memory=13G

#specify a project (probably not necessary, so currently off)
##     $ -P aclab

#merge the error and output
#$ -j y

#send email at the end
#$ -m e



module load python3 pytorch tensorflow
source env/bin/activate

python trainer_c4.py $@