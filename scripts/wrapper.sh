#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G      # memory; default unit is megabytes
#SBATCH --time=24:00:00           # time (DD-HH:MM)
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=a25mural@uwaterloo.ca
#SBATCH --mail-type=ALL

export CUDA_LAUNCH_BLOCKING=1; python3 /home/caesar/scratch/jobs/scripts/trainingdata.py
