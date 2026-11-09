#!/bin/bash
#SBATCH --job-name=ec_control
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ec_control_%j.out
#SBATCH --error=logs/ec_control_%j.err
set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/ergodic_qsd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/examples/ergodic_control
echo "Host: $(hostname)  Started: $(date)"
python -u control_test.py --walkers ${WALKERS:-4000} --steps ${STEPS:-15000} --dt ${DT:-1e-3}
echo "Finished: $(date)"
