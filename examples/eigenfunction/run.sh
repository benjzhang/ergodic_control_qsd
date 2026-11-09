#!/bin/bash
#SBATCH --job-name=eigenfunction
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/eigenfunction_%j.out
#SBATCH --error=logs/eigenfunction_%j.err

# Submit with, e.g.:
#   sbatch --export=ALL,EPS=0.25,N=50,T=100 run.sh
# (eps=0.25 -> a=eps=0.25, generator -(eps/2)D^2, lambda ~= 0.142)
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/eigenfunction

echo "Host: $(hostname)  Started: $(date)  eps=${EPS:-0.25} N=${N:-50} T=${T:-100} h0=${H0:-0.1} burn=${BURN:-10}"
python -u run.py --epsilon "${EPS:-0.25}" --N "${N:-50}" --T "${T:-100}" --h0 "${H0:-0.1}" --burn "${BURN:-10}"
echo "Finished: $(date)"
