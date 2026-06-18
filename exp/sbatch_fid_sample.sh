#!/bin/bash
# Example: 8-way class shard for FID sampling on TWCC.
# Usage:
#   cd VAR_polarQuant
#   sbatch --array=0-7 exp/sbatch_fid_sample.sh none
#   sbatch --array=0-7 exp/sbatch_fid_sample.sh uniform_int4
#   sbatch --array=0-7 exp/sbatch_fid_sample.sh int6_kmeans_int4
#
# Deeper model (var_d30.pth, ~2B params; needs longer walltime):
#   sbatch --time=08:00:00 --export=ALL,MODEL_DEPTH=30 --array=0-7 exp/sbatch_fid_sample.sh none
#   sbatch --time=08:00:00 --export=ALL,MODEL_DEPTH=30 --array=0-7 exp/sbatch_fid_sample.sh uniform_int4
#
# Adjust -A, -p, paths, and array size to match your account.

#SBATCH -A MST112145
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH -t 04:00:00
# gtest walltime cap is 30 min; re-submit the same command to continue (--skip-existing).
#SBATCH --array=0-7
#SBATCH -J var_fid
#SBATCH -o /home/jasonpan0930/VAR_research/VAR_PolarQuant/logs/fid_%x_%A_%a.out

set -euo pipefail
ROOT="/home/jasonpan0930/VAR_research/VAR_PolarQuant"
cd "${ROOT}"

# $1 = --polar-quant for exp/exp_fid_sample.py (θ₁ is always INT6 except baseline)
#
# Baseline (FP16 K cache, no polar quant):
#   none | baseline | fp16 | off
#
# Polar θ₂ configs (enable_polar_k_cache):
#   uniform_int4      INT6 θ₁ + uniform INT4 θ₂ (16 levels)
#   e2m1_fp4          INT6 θ₁ + OCP E2M1 FP4 θ₂
#   fp6_e3m2          INT6 θ₁ + FP6 E3M2 θ₂ (64 codes → 4-bit use in tree)
#   fp6_e2m3          INT6 θ₁ + FP6 E2M3 θ₂
#   int6_kmeans_int4  INT6 θ₁ + K-means INT4 θ₂ (needs depth-matched codebook; see below)
#
# K-means codebook (fit once per model depth, then FID):
#   srun ... python exp/exp_theta2_kmeans.py --model-depth 30 --refit
#   -> polar_quant_dumps/theta2_kmeans_d30/codebook.json
#
POLAR_QUANT="${1:-none}"
PYTHON="${PYTHON:-/home/jasonpan0930/.conda/envs/var_env/bin/python}"
DEPTH="${MODEL_DEPTH:-16}"

N_SHARDS=8
CLASSES_PER=$(( (1000 + N_SHARDS - 1) / N_SHARDS ))
CLASS_START=$(( SLURM_ARRAY_TASK_ID * CLASSES_PER ))
CLASS_END=$(( CLASS_START + CLASSES_PER - 1 ))
if (( CLASS_END >= 1000 )); then CLASS_END=999; fi

OUT_DIR="${ROOT}/fid_samples/${POLAR_QUANT}_d${DEPTH}"
mkdir -p "${ROOT}/logs" "${OUT_DIR}"

KMEANS_ARGS=()
if [[ "${POLAR_QUANT}" == "int6_kmeans_int4" ]]; then
  KMEANS_CB="${ROOT}/polar_quant_dumps/theta2_kmeans_d${DEPTH}/codebook.json"
  if [[ "${DEPTH}" == "16" && ! -f "${KMEANS_CB}" && -f "${ROOT}/polar_quant_dumps/theta2_kmeans/codebook.json" ]]; then
    KMEANS_CB="${ROOT}/polar_quant_dumps/theta2_kmeans/codebook.json"
  fi
  KMEANS_ARGS=(--kmeans-codebook "${KMEANS_CB}")
fi

SKIP="${SKIP_EXISTING:-1}"
SKIP_ARGS=()
if (( SKIP )); then
  SKIP_ARGS=(--skip-existing)
fi

echo "shard=${SLURM_ARRAY_TASK_ID} polar=${POLAR_QUANT} depth=${DEPTH} classes=[${CLASS_START},${CLASS_END}] skip=${SKIP} -> ${OUT_DIR}"

"${PYTHON}" "${ROOT}/exp/exp_fid_sample.py" \
  --model-depth "${DEPTH}" \
  --polar-quant "${POLAR_QUANT}" \
  --out-dir "${OUT_DIR}" \
  --class-start "${CLASS_START}" \
  --class-end "${CLASS_END}" \
  "${KMEANS_ARGS[@]}" \
  "${SKIP_ARGS[@]}"
