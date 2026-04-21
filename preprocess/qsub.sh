#!/bin/bash
#PBS -N FlowSplit
#PBS -P col7560.course
#PBS -l select=1:ncpus=8:mem=400GB:centos=icelake
#PBS -l walltime=40:00:00
#PBS -o hmyjob.out
#PBS -e hmyjob.err

cd $PBS_O_WORKDIR

# initialize conda properly on compute node
eval "$(conda shell.bash hook)"
conda activate netex

python flows_split.py
