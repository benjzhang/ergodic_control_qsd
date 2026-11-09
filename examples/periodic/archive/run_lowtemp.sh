#!/bin/bash
#SBATCH --job-name=periodic_lowT
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/lowtemp_%j.out
#SBATCH --error=logs/lowtemp_%j.err

set -e

module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1

cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/periodic

echo "Host: $(hostname)   Started: $(date)"
# batch simulation only (no postprocess/plots); variance analysis done separately
python -u periodic_batch_runner.py
echo "Finished: $(date)"
