#!/bin/bash
#SBATCH --job-name=ec_wdsm
#SBATCH --partition=cpu
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/ec_wdsm_%j.out
#SBATCH --error=logs/ec_wdsm_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname) Started: $(date)"
python -u control_weighted_smooth.py --epochs 5000
echo "Finished: $(date)"
