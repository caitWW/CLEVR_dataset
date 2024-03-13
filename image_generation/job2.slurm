#!/bin/bash
#SBATCH --job-name=test #name for job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 #num of gpus per node
#SBATCH --partition=mig
#SBATCH --time=04:00:00 #total run time limit (HH:MM:SS)

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=qw3971@princeton.edu

module load anaconda3/2023.3
conda activate torch-env

python3 retina.py