# Environment Setup Guide

This guide covers setting up the **VAR_polarQuant** project from scratch on a Linux machine with an NVIDIA GPU.

---

## Prerequisites

- NVIDIA GPU with CUDA support (V100 or newer)
- NVIDIA driver ≥ 535
- Python 3.10+
- Git
- ~30 GB disk space (checkpoints + experiment outputs)

---

## 1. Clone the Repo

```bash
git clone git@github.com:jasonpan0930/VAR_PolarQuant.git
cd VAR_PolarQuant
```

---

## 2. Python Environment

Choose **one** of the two options. Do **not** nest venv inside conda.

### Option A: Generic Linux (venv)

```bash
python3 -m venv venv
source venv/bin/activate
```

### Option B: NCHC Clusters — 晶創25 / TWCC / 台灣杉 (conda only)

These clusters provide `miniconda3` via `module load`. Use **pure conda** — no extra `venv` layer.

```bash
module load miniconda3
conda create -n var_env python=3.10 -y
conda activate var_env
```

In your SLURM job script, add the same two lines before running Python:
```bash
module load miniconda3
conda activate var_env
```

> **Storage**: If `/home` quota is tight, create the conda env under `/work/<user>`:
> ```bash
> conda create -p /work/$USER/var_env python=3.10 -y
> conda activate /work/$USER/var_env
> ```

---

## 3. Install PyTorch

| Cluster | GPU | Command |
|---------|-----|---------|
| 晶創25 | H100 / H200 (CUDA 12) | `pip install torch torchvision` |
| TWCC (台灣杉二號) | V100 (CUDA 11.8) | `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118` |
| Generic | Mixed | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |

---

## 4. Install Python Dependencies

Install **torch first** (Section 3), then the rest:

```bash
# Avoid requirements.txt downgrading torch — install deps manually or with --no-deps
pip install -r requirements.txt --no-deps

# Additional dependencies (not in requirements.txt)
pip install matplotlib seaborn scikit-learn tensorflow tqdm
```

> **Note**: `tensorflow` is only needed for FID evaluation (OpenAI evaluator uses TF 2.x). On NCHC clusters, add `module load cuda/11.7` before evaluator runs.

---

## 5. Download Model Checkpoints

Download from [HuggingFace FoundationVision/VAR](https://huggingface.co/FoundationVision/VAR):

```bash
# VAE (required)
wget https://huggingface.co/FoundationVision/VAR/resolve/main/vae_ch160v4096z32.pth

# VAR-d16 (310M params, FID ≈ 3.55)
wget https://huggingface.co/FoundationVision/VAR/resolve/main/var_d16.pth

# VAR-d30 (2B params, FID ≈ 1.97) — optional, ~7.5 GB
wget https://huggingface.co/FoundationVision/VAR/resolve/main/var_d30.pth
```

Place all `.pth` files in the project root (`VAR_polarQuant/`).

---

## 6. Set Up guided-diffusion (FID Evaluation)

The FID evaluator is OpenAI's `guided-diffusion`.

```bash
# Clone beside the project
cd ..
git clone https://github.com/openai/guided-diffusion.git
cd guided-diffusion/evaluations

# Download ImageNet reference batch (required for FID computation)
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

### FID evaluator script

```bash
cat > run_fid_eval.sh << 'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
SAMPLE_NPZ="${1:?usage: run_fid_eval.sh SAMPLE.npz}"
REF_NPZ="${2:-VIRTUAL_imagenet256_labeled.npz}"
PYTHON="path/to/your/python"   # venv/bin/python or conda python
cd "$(dirname "$0")"
"$PYTHON" evaluator.py "$REF_NPZ" "$SAMPLE_NPZ"
SCRIPT

chmod +x run_fid_eval.sh
```

> **NCHC clusters**: The evaluator may require `module load cuda/11.7` and specific `LD_LIBRARY_PATH` for pip-installed `nvidia/cudnn` and `nvidia/cublas`.

---

## 7. Verify Setup

```bash
# venv:  source venv/bin/activate
# conda: conda activate var_env

# Test model loading (CPU is fine for this check)
python -c "
import torch
from models import build_vae_var

vae_ckpt = 'vae_ch160v4096z32.pth'
var_ckpt = 'var_d16.pth'
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device='cpu', patch_nums=patch_nums,
    num_classes=1000, depth=16, shared_aln=False,
)
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
print('Checkpoints loaded successfully')
print(f'VAR depth={len(var.blocks)}, heads={var.num_heads}')
"
```

---

## 8. Test Inference (GPU required)

Generate a few images on GPU to verify everything works:

```bash
python exp/exp_fid_sample.py \
  --model-depth 16 \
  --polar-quant none \
  --out-dir fid_samples/test_d16 \
  --class-start 0 --class-end 0 \
  --samples-per-class 2
```

Expected output: `fid_samples/test_d16/0000_00.png` and `0000_01.png`.

> **NCHC clusters**: Do NOT run this on the login node. Use `srun` or `sbatch`.

---

## Directory Layout After Setup

```
VAR_PolarQuant/
├── vae_ch160v4096z32.pth          # VAE checkpoint
├── var_d16.pth                    # VAR-d16 checkpoint
├── var_d30.pth                    # VAR-d30 checkpoint (optional)
├── exp/                           # Experiment scripts
│   ├── exp_fid_sample.py
│   ├── exp_theta2_kmeans.py
│   ├── exp_polar_angle_dist.py
│   ├── exp_cross_block_cumulative_mse.py
│   ├── sbatch_fid_sample.sh
│   └── ...
├── utils/                         # Core libraries
│   ├── polar_kv_quant.py          # Polar encode/decode
│   ├── angle_quant.py             # Codebooks & configs
│   ├── theta2_kmeans.py           # K-means codebook
│   └── polar_angle_viz.py         # Error visualization
├── models/                        # Modified VAR model
│   ├── basic_var.py               # SelfAttention with polar cache
│   └── var.py                     # enable_polar_k_cache()
├── SETUP.md                       # This file
└── PROJECT.md                     # Project overview

../guided-diffusion/evaluations/   # OpenAI FID evaluator
```

---

## SLURM / HPC Notes

On NCHC cluster environments (晶創25, TWCC, 台灣杉):

- **Never** run CUDA inference on the login node — use `srun` or `sbatch`.
- Load conda in your job script:
  ```bash
  module load miniconda3
  conda activate var_env
  ```
- The `exp/sbatch_fid_sample.sh` script is pre-configured for 8-way class sharding.
- Override defaults with environment variables:
  ```bash
  sbatch --export=ALL,MODEL_DEPTH=30 --array=0-7 exp/sbatch_fid_sample.sh uniform_int4
  ```
- Monitor with `squeue -u $USER`.
- 晶創25 login: `ssh <user>@nano5.nchc.org.tw`
