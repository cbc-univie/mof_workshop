#!/bin/bash

#SBATCH -p ngpu,gpu
#SBATCH --gres=gpu

source /home/marion/miniconda3/bin/activate/ mlnew

python 04_nvt.py

#sbatch -J "nvt" -o out/nvt.out 04_nvt.sh