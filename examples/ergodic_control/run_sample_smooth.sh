#!/bin/bash
#SBATCH --job-name=ec_smpl
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ec_smpl_%j.out
#SBATCH --error=logs/ec_smpl_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname) Started: $(date)"
python -u sample_smooth.py --T 50 --N 20 --h0 0.1 --seed 0 --nsamp 40000
echo "Finished: $(date)"
