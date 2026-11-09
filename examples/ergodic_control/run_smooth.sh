#!/bin/bash
#SBATCH --job-name=ec_smooth
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11
#SBATCH --output=logs/ec_smooth_%A_%a.out
#SBATCH --error=logs/ec_smooth_%A_%a.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control

# array index 0-5 -> plain FV seeds 0-5 ;  6-11 -> INS seeds 0-5
IDX=${SLURM_ARRAY_TASK_ID}
if [ "$IDX" -lt 6 ]; then S="fv";  SEED=$IDX; else S="ins"; SEED=$((IDX-6)); fi
echo "Host: $(hostname) Started: $(date)  sampler=$S seed=$SEED h0=${H0:-0.1} T=${T:-50}"
python -u run_smooth.py --sampler $S --seed $SEED --h0 ${H0:-0.1} --T ${T:-50} --N 20 --burn 0.3
echo "Finished: $(date)"
