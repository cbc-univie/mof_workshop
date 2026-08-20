#!/bin/bash

#SBATCH -p ngpu,gpu
#SBATCH --gres=gpu

source /home/marion/miniconda3/bin/activate/ mlnew

python 03_equilibration.py

#sbatch -J "equi" -o out/equi.out 03_equilibration.sh