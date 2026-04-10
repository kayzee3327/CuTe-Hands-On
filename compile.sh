#!/bin/bash

#SBATCH --job-name=compile      # Job name
#SBATCH --output=compile.out  # Standard output and error log (%j expands to jobID)
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=64G                   # Job memory request
#SBATCH --time=00:15:00             # Time limit hrs:min:sec

source /home/spack/spack/share/spack/setup-env.sh
spack load cmake@3.28.6
spack load cuda@12.9.0

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build


echo "Job finished."