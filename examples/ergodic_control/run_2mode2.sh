#!/bin/bash
#SBATCH --job-name=ec2m2
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ec2m2_%j.out
#SBATCH --error=logs/ec2m2_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname) Started: $(date)"
python -u compare_2mode.py --killing confine --init wellA --a ${A} --kappa ${KAPPA:-2} --T 100 --N 20
echo "Finished: $(date)"
