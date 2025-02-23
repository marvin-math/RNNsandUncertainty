#!/bin/bash
#SBATCH --job-name=GRU4
#SBATCH --output=../logs/RNNsimulation_%A_%a.out
#SBATCH --error=../logs/RNNsimulation_%A_%a.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH 
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --nice=10000

 
python RNN.py