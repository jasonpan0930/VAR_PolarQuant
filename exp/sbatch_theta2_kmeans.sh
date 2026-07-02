#!/bin/bash
# Submit per-level θ₂ K-means codebook fit (K+V in one pass).
#
# Usage:
#   sbatch exp/sbatch_theta2_kmeans.sh           # default: T1-T3 4-bit, T4-T5 2-bit
#   sbatch --export=ALL,REFIT=1,PER_LEVEL_K="16 16 16 16 16" exp/sbatch_theta2_kmeans.sh  # all 4-bit
#   sbatch --export=ALL,REFIT=1,PER_LEVEL_K="16 16 16 8 8" exp/sbatch_theta2_kmeans.sh    # T4/T5 3-bit
#
#   To save to a custom dir (e.g. to later copy to consumption paths):
#   sbatch --export=ALL,REFIT=1,PER_LEVEL_K="16 16 16 16 16",OUT_DIR=/path/to/my_cb_k16 ...
#
# Outputs (per OUT_DIR or default configs/codebooks/theta2_kmeans_d<depth>_T{N}):
#   T{1..5}/codebook.json      (K)
#   T{1..5}_v/codebook.json    (V)

#SBATCH -A MST112145
#SBATCH -p gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH -t 04:00:00
#SBATCH -J theta2_kmeans_perlevel
#SBATCH -o /home/jasonpan0930/var_research/VAR_polarQuant/logs/theta2_kmeans_perlevel_%j.out

set -euo pipefail
ROOT="/home/jasonpan0930/var_research/VAR_polarQuant"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/jasonpan0930/.conda/envs/var_env/bin/python}"
DEPTH="${MODEL_DEPTH:-30}"

CLASS_LABELS="${CLASS_LABELS:-22 45 123 437 701}"
REFIT="${REFIT:-0}"
PER_LEVEL_K="${PER_LEVEL_K:-16 16 16 4 4}"
OUT_DIR="${OUT_DIR:-}"

ARGS=(
    --model-depth "${DEPTH}"
    --per-level
    --per-level-k ${PER_LEVEL_K}
    --class-labels ${CLASS_LABELS}
)

if [[ -n "${OUT_DIR}" ]]; then
    ARGS+=(--out-dir "${OUT_DIR}")
fi

if (( REFIT )); then
    ARGS+=(--refit)
fi

echo "depth=${DEPTH} K+V per-level k=(${PER_LEVEL_K}) classes=[${CLASS_LABELS}] refit=${REFIT} out_dir=${OUT_DIR:-default}"

"${PYTHON}" "${ROOT}/exp/exp_theta2_kmeans.py" "${ARGS[@]}"
