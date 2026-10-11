#!/bin/bash
#SBATCH --job-name=hybrid-cvip-resume
#SBATCH --partition=gpu_faculty
#SBATCH --account=faculty
#SBATCH --qos=normal
#SBATCH --nice=0
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --time-min=00:45:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/uformer-%j.out

set -euo pipefail
cd "${HYBRID_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec "$(pwd -P)/.venv/bin/python" scripts/run_faculty_pipeline.py
