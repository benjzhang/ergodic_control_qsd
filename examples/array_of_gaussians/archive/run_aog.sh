#!/bin/bash
#SBATCH --job-name=aog
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/aog_%j.out
#SBATCH --error=logs/aog_%j.err

set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/array_of_gaussians
echo "Host: $(hostname)  Started: $(date)"
python -u run_aog.py
echo "Finished: $(date)"
