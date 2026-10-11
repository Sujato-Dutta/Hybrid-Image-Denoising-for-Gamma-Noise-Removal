#!/bin/bash
#SBATCH --job-name=hybrid-cvip-setup
#SBATCH --partition=cpu_faculty
#SBATCH --account=faculty
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-%j.out

set -euo pipefail
cd "${HYBRID_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$(pwd -P)/tmp/pip-cache"
mkdir -p tmp
python3 -m venv --system-site-packages .venv
PYTHON="$(pwd -P)/.venv/bin/python"
if ! "$PYTHON" -c 'import torch; assert torch.version.cuda is not None'; then
    "$PYTHON" -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
fi
"$PYTHON" -m pip install -r requirements-camera-ready.txt einops
"$PYTHON" -m pip freeze > docs/faculty_environment.lock.txt
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/verify_uformer.py
"$PYTHON" scripts/prepare_foe_protocol.py --config configs/uformer_gamma.json
echo 'Faculty environment and protocol verification complete.'
