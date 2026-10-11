#!/bin/bash
#SBATCH --job-name=hybrid-cvip-uformer
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --qos=normal
#SBATCH --nice=0
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --time-min=00:45:00
#SBATCH --output=logs/uformer-%j.out

set -euo pipefail
cd "${HYBRID_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
PYTHON="$(pwd -P)/.venv/bin/python"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Allocated GPU is not visible"; p=torch.cuda.get_device_properties(0); print(p); assert p.total_memory >= 35*1024**3, "A full 40 GB A100 is required"'
"$PYTHON" scripts/prepare_foe_protocol.py --config configs/uformer_gamma.json
"$PYTHON" run_uformer_gamma.py --config configs/uformer_gamma.json --looks 1 --device cuda
"$PYTHON" run_uformer_gamma.py --config configs/uformer_gamma.json --looks 10 --device cuda
"$PYTHON" evaluate_uformer_gamma.py --config configs/uformer_gamma.json --device cuda --replicates 5
"$PYTHON" profile_complexity.py --config configs/uformer_gamma.json --device cuda
"$PYTHON" scripts/export_camera_ready_results.py --config configs/uformer_gamma.json
