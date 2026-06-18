"""
Plot global K-vector MSE propagation (k_error_global.png) per polar quant config,
plus VAR sample images (original fp16 K cache + one per polar config).

  cd VAR_polarQuant && python exp/exp_polar_angle_dist.py

Requires a pre-fitted K-means θ₂ codebook (does not re-run K-means here):
  polar_quant_dumps/theta2_kmeans/codebook.json
  (create via exp/exp_theta2_kmeans.py)

Outputs:
  polar_quant_dumps/angle_plots/sample_original.png
  polar_quant_dumps/angle_plots/<config>/k_error_global.png   (one inference pass each)
  polar_quant_dumps/angle_plots/<config>/sample_var.png
"""
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import PIL.Image as PImage
import torch
import torchvision

from models import build_vae_var
from utils.angle_quant import POLAR_QUANT_CONFIGS, register_theta2_kmeans_codebook
from utils.polar_angle_viz import PolarAngleStatsSession, render_all_plots
from utils.theta2_kmeans import load_theta2_codebook

OUT_BASE = ROOT / 'polar_quant_dumps' / 'angle_plots'
THETA2_KMEANS_CODEBOOK = ROOT / 'polar_quant_dumps' / 'theta2_kmeans' / 'codebook.json'
MODEL_DEPTH = 30
BATCH_SIZE = 1
CLASS_LABELS = (22, )
SEED, CFG, TOP_K, TOP_P = 0, 4, 900, 0.95
TARGET_BLOCKS = (15,)
POLAR_QUANT_CONFIGS_TO_RUN = (
    'uniform_int4',
    'e2m1_fp4',
    'fp6_e3m2',
    'fp6_e2m3',
    'int6_kmeans_int4',
)

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

vae_ckpt = ROOT / 'vae_ch160v4096z32.pth'
var_ckpt = ROOT / f'var_d{MODEL_DEPTH}.pth'
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')


def save_tensor_image(recon: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = torchvision.utils.make_grid(recon, nrow=1, padding=0, pad_value=1.0)
    grid = grid.permute(1, 2, 0).mul(255).cpu().numpy()
    PImage.fromarray(grid.astype(np.uint8)).save(path)
    print(f'saved sample -> {path}')


def set_infer_seeds() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def run_infer(var, label_B: torch.Tensor) -> torch.Tensor:
    set_infer_seeds()
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            return var.autoregressive_infer_cfg(
                B=BATCH_SIZE, label_B=label_B, cfg=CFG,
                top_k=TOP_K, top_p=TOP_P, g_seed=SEED,
            )


def register_kmeans_theta2_from_codebook(path: Path = THETA2_KMEANS_CODEBOOK) -> None:
    """Load saved K-means θ₂ centers and register ``int6_kmeans_int4`` (no refit)."""
    if not path.is_file():
        raise FileNotFoundError(
            f'K-means θ₂ codebook not found: {path}\n'
            'Fit and save it first: python exp/exp_theta2_kmeans.py'
        )
    centers, meta = load_theta2_codebook(path)
    register_theta2_kmeans_codebook(centers)
    n = meta.get('n_samples', '?')
    print(f'registered int6_kmeans_int4 from {path} ({len(centers)} levels, fit n_samples={n})')


def run_config(var, label_B: torch.Tensor, config_name: str, out_dir: Path) -> None:
    angle_stats = PolarAngleStatsSession(
        target_blocks=TARGET_BLOCKS,
        max_samples_per_stage=15000,
        max_structured_tokens=256,
        batch_index=0,
    )
    var.set_polar_quant(config_name)
    var.set_angle_stats(angle_stats)

    cfg = POLAR_QUANT_CONFIGS[config_name]
    print(f'--- {config_name}: {cfg.label} -> {out_dir} ---')
    print('running polar inference to collect K errors...')
    recon = run_infer(var, label_B)

    kerr = angle_stats.aggregate_k_errors()
    render_all_plots({}, out_dir, kerr=kerr)
    save_tensor_image(recon, out_dir / 'sample_var.png')
    print(f'done -> {out_dir}')


vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
)
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()

label_B = torch.tensor(CLASS_LABELS[:BATCH_SIZE], device=device)
OUT_BASE.mkdir(parents=True, exist_ok=True)
register_kmeans_theta2_from_codebook()

print('running baseline (fp16 K cache)...')
var.set_polar_quant(None)
save_tensor_image(run_infer(var, label_B), OUT_BASE / 'sample_original.png')

for name in POLAR_QUANT_CONFIGS_TO_RUN:
    if name not in POLAR_QUANT_CONFIGS:
        raise ValueError(f'unknown config {name!r}')
    run_config(var, label_B, name, OUT_BASE / name)

print(f'all configs done under {OUT_BASE}')
