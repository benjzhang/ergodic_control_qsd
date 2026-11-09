#!/bin/bash
#SBATCH --job-name=ec_sample
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ec_sample_%j.out
#SBATCH --error=logs/ec_sample_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname)  Started: $(date)  T=${T:-100} N=${N:-20}"
python -u sample_qsd.py --T "${T:-100}" --N "${N:-20}" --burn "${BURN:-0.2}" --nsamp "${NSAMP:-80000}"
echo "Finished: $(date)"
