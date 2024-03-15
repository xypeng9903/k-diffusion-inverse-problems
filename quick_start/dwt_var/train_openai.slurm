#!/bin/bash

#SBATCH --job-name=openai_dwt
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4     
#SBATCH --mail-type=end
#SBATCH --mail-user=xypeng9903@outlook.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load gcc cuda

./cudaTensorCoreGemm

srun python train_openai.py --openai-ckpt ../model_zoo/diffusion_ffhq_10m.pt --batch-size 12