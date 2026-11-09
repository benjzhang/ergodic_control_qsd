#!/bin/bash
#SBATCH --job-name=ec_cmp
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ec_cmp_%j.out
#SBATCH --error=logs/ec_cmp_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname)  Started: $(date)"
python -u compare_fv.py --T ${T:-100} --N ${N:-20} --h0 ${H0:-0.1} --burn ${BURN:-0.2}
echo "Finished: $(date)"
