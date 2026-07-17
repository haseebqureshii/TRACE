#!/bin/bash
#SBATCH --job-name=trace_stress_test
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=stress_test.out
#SBATCH --error=stress_test.err

echo "Initializing Project TRACE Stress Test..."

# 1. Load Environment
module load shpc/python/3.9.2-slim/module

# 2. Air-gap the network (RESTORED)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 3. HPC Anti-Deadlock Measures
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# 4. Execute
python run_stress_test.py

echo "Job Complete."