#!/bin/bash

#SBATCH --job-name=wt_large
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4     
#SBATCH --mail-type=end
#SBATCH --mail-user=xypeng9903@outlook.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load gcc cuda

./cudaTensorCoreGemm

accelerate launch train.py \
--config configs/train_ffhq_wt_large.json \
--batch-size 8 \
--grad-accum-steps 4 \
--sample-n 1 \
--name runs/train/ffhq_dct_large