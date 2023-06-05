#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1              
#SBATCH --exclusive
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:4
#SBATCH -o ./hi.out         
#SBATCH -e ./hi.err      

echo $SBATCH_GRES
echo $SLURM_JOB_NODELIST
echo $SLURM_NTASKS_PER_GPU
echo $SLURM_NTASKS