#!/bin/bash
#SBATCH --partition=gpu2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=12
#SBATCH -o %j.sbatch.%N.out         
#SBATCH -e %j.sbatch.%N.err         

mkdir -p $SLURM_JOB_ID

srun --partition=gpu2 \
      --gres=gpu:a10:1 \
      --cpus-per-task=12 \
      -o %j/%N.%t.out \  # %j jobid, %N node id, %t task id
      -e %j/%N.%t.err \
      bash -c "nvidia-smi && sleep 100" 
