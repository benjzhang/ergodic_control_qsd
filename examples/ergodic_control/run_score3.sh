#!/bin/bash
#SBATCH --job-name=ec_score3
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ec_score3_%j.out
#SBATCH --error=logs/ec_score3_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname) Started: $(date)"
python -u score_control_smooth3.py --epochs 8000 --sigma 0.08 --harmonics 5
echo "Finished: $(date)"
