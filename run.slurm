#!/bin/bash
#SBATCH --job-name=glm_pf
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/js3785/Codes/GLM_Git/GLM_SPIKES/logs/%x_%j.out
#SBATCH --error=/home/js3785/Codes/GLM_Git/GLM_SPIKES/logs/%x_%j.err

set -euo pipefail

echo "========== Job info =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-N/A}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "=============================="

cd /home/js3785/Codes/GLM_Git/GLM_SPIKES
mkdir -p logs

source /home/js3785/anaconda3/etc/profile.d/conda.sh
conda activate behavior

export JOBLIB_TEMP_FOLDER=/tmp/$USER/joblib_$SLURM_JOB_ID
mkdir -p "$JOBLIB_TEMP_FOLDER"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

echo "Python: $(which python)"
python --version

srun --cpu-bind=cores python -u GLM_Poisson_Forward.py

echo "End time: $(date)"
