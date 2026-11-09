#!/bin/bash
#SBATCH --job-name=ec_trace
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-7
#SBATCH --output=logs/ec_trace_%A_%a.out
#SBATCH --error=logs/ec_trace_%A_%a.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
IDX=${SLURM_ARRAY_TASK_ID}
if [ "$IDX" -lt 4 ]; then S="ins"; SEED=$IDX; else S="fv"; SEED=$((IDX-4)); fi
echo "Host: $(hostname) $(date) sampler=$S seed=$SEED"
python -u trace_smooth.py --sampler $S --seed $SEED --T 50 --N 20 --h0 0.1
echo "done $(date)"
