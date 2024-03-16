#!/bin/bash

#SBATCH --job-name=test_dwt_large
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1     
#SBATCH --mail-type=end
#SBATCH --mail-user=xypeng9903@outlook.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load gcc cuda

./cudaTensorCoreGemm

bash quick_start/dwt_var/test.sh