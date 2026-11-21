#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0-2
#SBATCH -G a30:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user="lhuan130@asu.edu"

cd /scratch/lhuan130/compare_gan_ldm_proj/Torch-StudioGAN

#job
../myenv/bin/python src/main.py -t -hdf5 -l -metrics is fid prdc -ref "train" -cfg src/configs/Tiny_ImageNet/ReACGAN.yaml -data /scratch/lhuan130/compare_gan_ldm_proj/gan/data -save /scratch/lhuan130/compare_gan_ldm_proj/gan/results/reacgan_tiny -mpc --pre_resizer "lanczos" --post_resizer "friendly" --eval_backbone "InceptionV3_tf"
