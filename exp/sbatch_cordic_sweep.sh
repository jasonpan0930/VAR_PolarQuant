#!/bin/bash
#SBATCH -A MST112145
#SBATCH -p gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH -t 00:30:00
#SBATCH -J fhat_sanity
#SBATCH -o /home/jasonpan0930/var_research/VAR_polarQuant/logs/fhat_sanity_%j.out

set -euo pipefail
ROOT="/home/jasonpan0930/var_research/VAR_polarQuant"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/jasonpan0930/.conda/envs/var_env/bin/python}"
mkdir -p "${ROOT}/logs"

echo "=== f_hat golden sanity check ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

PYTHONUNBUFFERED=1 ${PYTHON} exp/fhat_sanity.py

echo ""
echo "=== Done: $(date) ==="
