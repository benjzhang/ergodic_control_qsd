#!/bin/bash
#SBATCH --job-name=ec_cval
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ec_cval_%j.out
#SBATCH --error=logs/ec_cval_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname) Started: $(date)"
python -u control_validate_smooth.py
echo "Finished: $(date)"
