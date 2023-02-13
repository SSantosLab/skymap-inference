#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH -J lightweight_injection
#SBATCH --mail-user=imcmahon@umich.edu
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00

module load python
conda activate bbh_inference

#run the application:
python3 lw_injection_full.py

