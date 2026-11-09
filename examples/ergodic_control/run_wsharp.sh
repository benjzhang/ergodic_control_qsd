#!/bin/bash
#SBATCH --job-name=ec_wshrp
#SBATCH --partition=cpu
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/ec_wshrp_%j.out
#SBATCH --error=logs/ec_wshrp_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname) Started: $(date)"
python -u control_weighted_sharp.py --T 100 --N 20 --burn 0.1 --epochs 5000
echo "control run done, rendering figure: $(date)"
python -u plot_howitworks_sharp.py
echo "Finished: $(date)"
