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

CYCLES_CUDA_EXTRA_CFLAGS="-I/usr/local/cuda-11.7/include" /home/qw3971/Downloads/blender-2.79-linux-glibc219-x86_64/blender --background --python render_images.py -- --num_images 2000 --min_objects 1 --max_objects 1 --output_image_dir test2 --use_gpu 1