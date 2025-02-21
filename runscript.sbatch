#!/bin/bash
#SBATCH --job-name=GRU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=my_job_output.txt
#SBATCH --error=my_job_error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

 
python RNN.py