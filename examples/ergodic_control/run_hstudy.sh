#!/bin/bash
#SBATCH --job-name=ec_hstud
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ec_hstud_%j.out
#SBATCH --error=logs/ec_hstud_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname) Started: $(date)  h0=${H0} T=${T:-50}"
python -u sample_qsd.py --T ${T:-50} --N 20 --h0 ${H0} --burn 0.3 --nsamp 20000 --seed 1
echo "Finished: $(date)"
