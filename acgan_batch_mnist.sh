#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-1
#SBATCH -G a30:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user="lhuan130@asu.edu"


cd /scratch/lhuan130/compare_gan_ldm_proj/

#job
myenv/bin/python gan/acgan_mnist.py

