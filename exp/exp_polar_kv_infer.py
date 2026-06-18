"""
Polar-quantized K cache inference + save quant tensors for research.

  cd VAR_polarQuant && python exp/exp_polar_kv_infer.py

Outputs:
  polar_quant_dumps/sample_baseline.png   (fp16 K cache, same seed/class)
  polar_quant_dumps/sample_polar.png
  polar_quant_dumps/sample_compare.png    (baseline | polar side-by-side)
  polar_quant_dumps/stageXX_pnY/blockZZ_k_polar.npz  (q1, q2, z per layer/stage)
  polar_quant_dumps/manifest.json
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
from utils.polar_kv_quant import roundtrip_error
from utils.polar_kv_store import PolarKVDumpSession

OUT_DIR = ROOT / 'polar_quant_dumps'
MODEL_DEPTH = 16
BATCH_SIZE = 1
CLASS_LABELS = (437,)
SEED, CFG, TOP_K, TOP_P = 0, 4, 900, 0.95
DUMP_BLOCKS = None  # None = all blocks; or e.g. (0, 15)
RUN_BASELINE_COMPARE = True
THETA2_QUANT = 'e2m1_fp4'  # 'uniform_int4' | 'e2m1_fp4'


def save_tensor_image(recon: torch.Tensor, path: Path) -> None:
    grid = torchvision.utils.make_grid(recon, nrow=1, padding=0, pad_value=1.0)
    grid = grid.permute(1, 2, 0).mul(255).cpu().numpy()
    PImage.fromarray(grid.astype(np.uint8)).save(path)


def run_infer(var, label_B: torch.Tensor, use_polar: bool, dump=None):
    if use_polar:
        var.set_polar_quant(THETA2_QUANT)
    else:
        var.set_polar_quant(None)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            return var.autoregressive_infer_cfg(
                B=BATCH_SIZE, label_B=label_B, cfg=CFG,
                top_k=TOP_K, top_p=TOP_P, g_seed=SEED,
            )

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

vae_ckpt = ROOT / 'vae_ch160v4096z32.pth'
var_ckpt = ROOT / f'var_d{MODEL_DEPTH}.pth'
hf = 'https://huggingface.co/FoundationVision/var/resolve/main'
for ckpt in (vae_ckpt, var_ckpt):
    if not ckpt.is_file():
        os.system(f'wget -q {hf}/{ckpt.name} -O {ckpt}')

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# quick round-trip sanity
_rt = roundtrip_error(torch.randn(4, 64))
print(f'polar K64 round-trip: mse={_rt["mse"]:.6f} max={_rt["max_abs"]:.4f}')

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
)
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()

OUT_DIR.mkdir(parents=True, exist_ok=True)
label_B = torch.tensor(CLASS_LABELS[:BATCH_SIZE], device=device)

if RUN_BASELINE_COMPARE:
    print('running baseline (fp16 K cache)...')
    baseline = run_infer(var, label_B, use_polar=False)
    save_tensor_image(baseline, OUT_DIR / 'sample_baseline.png')
    print(f'saved baseline -> {OUT_DIR / "sample_baseline.png"}')

print('running polar K cache...')
dump = PolarKVDumpSession(str(OUT_DIR), batch_index=0)
recon = run_infer(var, label_B, use_polar=True, dump=dump)
save_tensor_image(recon, OUT_DIR / 'sample_polar.png')
print(f'saved polar -> {OUT_DIR / "sample_polar.png"}')

if RUN_BASELINE_COMPARE:
    compare = torch.cat([baseline, recon], dim=0)
    save_tensor_image(compare, OUT_DIR / 'sample_compare.png')
    diff = (baseline - recon).abs()
    print(
        f'image diff vs baseline: mse={float((diff ** 2).mean()):.6f} '
        f'max={float(diff.max()):.4f} mean={float(diff.mean()):.4f}'
    )
    print(f'saved compare -> {OUT_DIR / "sample_compare.png"}  (left=baseline, right=polar)')
