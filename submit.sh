#!/bin/bash -l

# Request 4 CPUs
#$ -pe omp 8

# Request 1 GPU
#$ -l gpus=1
#$ -l gpu_c=3.7

#specify a project
#$ -P aclab

#merge the error and output
#$ -j y

#send email at the end
#$ -m e



module load python3 pytorch tensorflow
source env/bin/activate

# -lr 0.01 -opt nigt -diag false --batch_size 1 --ministeps 1 --eps 0.001 -wd 0.0
python trainer_c4.py $@