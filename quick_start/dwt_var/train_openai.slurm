#!/bin/bash

#SBATCH --job-name=openai_dwt
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:8     
#SBATCH --mail-type=end
#SBATCH --mail-user=xypeng9903@outlook.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load gcc cuda

./cudaTensorCoreGemm

srun python train_openai.py \
    --checkpoint runs/train_openai/version_5/checkpoints/epoch=43-step=27017.ckpt \
    --batch-size 12 \
    --accumulate-grad-batches 2 \
    --num-workers 32