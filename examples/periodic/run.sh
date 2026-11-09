#!/bin/bash
#SBATCH --job-name=periodic
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/periodic_%j.out
#SBATCH --error=logs/periodic_%j.err

# Submit with, e.g.:
#   sbatch --export=ALL,EPS=0.5,N=5,T=100,REPS=2 run.sh
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/periodic

echo "Host: $(hostname)  Started: $(date)  eps=${EPS} N=${N} T=${T:-100} reps=${REPS:-2}"
python -u run.py --epsilon "${EPS}" --N "${N}" --T "${T:-100}" --reps "${REPS:-2}"
echo "Finished: $(date)"
