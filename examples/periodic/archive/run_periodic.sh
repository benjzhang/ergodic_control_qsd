#!/bin/bash
#SBATCH --job-name=periodic_fv
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/periodic_%j.out
#SBATCH --error=logs/periodic_%j.err

set -e

module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1

cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/periodic

echo "Host: $(hostname)   Started: $(date)"
python -u periodic_run_all.py
echo "Finished: $(date)"
