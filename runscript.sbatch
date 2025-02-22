#!/bin/bash
#SBATCH --job-name=GRU4
#SBATCH --output=../logs/mb_array_%A_%a.out
#SBATCH --error=../logs/mb_array_%A_%a.err

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH 
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --nice=10000

 
python RNN.py