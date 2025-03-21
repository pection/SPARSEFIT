#!/bin/bash
#SBATCH --job-name=test_slurm_char_base      # Job name
#SBATCH --output=train_gpu_slurm_char_base.out    # Standard output and error log
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --time=72:00:00             # Time limit hh:mm:ss
#SBATCH --partition=batch            # Partition name
#SBATCH --gres=gpu:1              # Request 2 GPUs
#SBATCH --cpus-per-task=10

cd /home/slurmtest01/SILAFIT
source ~/SILAFIT/bin/activate
echo "The current time is: $(date)"
pip install --upgrade deepspeed
deactivate