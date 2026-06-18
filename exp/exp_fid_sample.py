"""
Generate ImageNet samples for FID / IS / Precision / Recall (OpenAI evaluator protocol).

Official VAR benchmarking (see README.md):
  cfg=1.5, top_k=900, top_p=0.96, more_smooth=False
  50,000 PNGs = 1000 classes x 50 images/class

Examples (run on GPU compute node via srun/sbatch, not login):

  # Baseline fp16 K cache, classes 0..124 (shard 0 of 8)
  python exp/exp_fid_sample.py --polar-quant none --out-dir fid_samples/baseline \\
      --class-start 0 --class-end 124

  # Polar quant config
  python exp/exp_fid_sample.py --polar-quant int6_kmeans_int4 \\
      --out-dir fid_samples/int6_kmeans_int4 --class-start 0 --class-end 999

  # After all shards merged into one folder with exactly 50_000 PNGs:
  python exp/exp_fid_sample.py --pack-npz-only --out-dir fid_samples/baseline

Then:
  python evaluator.py VIRTUAL_imagenet256_labeled.npz fid_samples/baseline.npz
"""
from __future__ import annotations

import argparse
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
from tqdm import tqdm

from models import build_vae_var
from utils.angle_quant import POLAR_QUANT_CONFIGS, register_theta2_kmeans_codebook
from utils.misc import create_npz_from_sample_folder
from utils.theta2_kmeans import load_theta2_codebook

NUM_CLASSES = 1000
SAMPLES_PER_CLASS_DEFAULT = 50
THETA2_KMEANS_CODEBOOK = ROOT / 'polar_quant_dumps' / 'theta2_kmeans' / 'codebook.json'

# VAR paper FID settings
FID_CFG = 1.5
FID_TOP_K = 900
FID_TOP_P = 0.96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='VAR / polar-K FID sample generation')
    p.add_argument('--model-depth', type=int, default=16, choices=(16, 20, 24, 30))
    p.add_argument(
        '--polar-quant', type=str, default='none',
        help='none|baseline = fp16 K cache; otherwise a polar config name '
             f'({", ".join(sorted(POLAR_QUANT_CONFIGS))}, int6_kmeans_int4)',
    )
    p.add_argument('--out-dir', type=Path, required=True, help='PNG output directory')
    p.add_argument('--class-start', type=int, default=0, help='first class index (inclusive)')
    p.add_argument('--class-end', type=int, default=999, help='last class index (inclusive)')
    p.add_argument('--samples-per-class', type=int, default=SAMPLES_PER_CLASS_DEFAULT)
    p.add_argument('--cfg', type=float, default=FID_CFG)
    p.add_argument('--top-k', type=int, default=FID_TOP_K)
    p.add_argument('--top-p', type=float, default=FID_TOP_P)
    p.add_argument('--seed-base', type=int, default=0, help='g_seed = seed_base + class*100 + sample_idx')
    p.add_argument('--kmeans-codebook', type=Path, default=THETA2_KMEANS_CODEBOOK)
    p.add_argument('--skip-existing', action='store_true', help='skip PNG if path exists')
    p.add_argument('--dry-run', action='store_true', help='print plan only')
    p.add_argument(
        '--pack-npz-only', action='store_true',
        help='only pack existing PNG folder to .npz (needs 50_000 files)',
    )
    return p.parse_args()


def png_path(out_dir: Path, class_idx: int, sample_idx: int) -> Path:
    return out_dir / f'{class_idx:04d}_{sample_idx:02d}.png'


def save_recon_png(recon: torch.Tensor, path: Path) -> None:
    """recon: (1, 3, H, W) in [0, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = recon[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    PImage.fromarray(img).save(path)


def setup_polar(var, polar_quant: str, kmeans_codebook: Path) -> str:
    name = polar_quant.lower().strip()
    if name in ('none', 'baseline', 'fp16', 'off'):
        var.set_polar_quant(None)
        return 'baseline_fp16_k'
    if name == 'int6_kmeans_int4':
        if not kmeans_codebook.is_file():
            raise FileNotFoundError(
                f'K-means codebook missing: {kmeans_codebook}\n'
                'Run: python exp/exp_theta2_kmeans.py'
            )
        centers, _ = load_theta2_codebook(kmeans_codebook)
        register_theta2_kmeans_codebook(centers)
    elif name not in POLAR_QUANT_CONFIGS:
        opts = ', '.join(['none'] + sorted(POLAR_QUANT_CONFIGS) + ['int6_kmeans_int4'])
        raise ValueError(f'unknown --polar-quant {polar_quant!r}; choose: {opts}')
    var.set_polar_quant(name)
    return name


def build_models(depth: int, device: str):
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae_ckpt = ROOT / 'vae_ch160v4096z32.pth'
    var_ckpt = ROOT / f'var_d{depth}.pth'
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=NUM_CLASSES, depth=depth, shared_aln=False,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    return vae, var


def count_planned(out_dir: Path, c0: int, c1: int, n_per: int, skip_existing: bool) -> int:
    n = 0
    for c in range(c0, c1 + 1):
        for s in range(n_per):
            if skip_existing and png_path(out_dir, c, s).is_file():
                continue
            n += 1
    return n


def generate(args: argparse.Namespace) -> None:
    c0 = max(0, args.class_start)
    c1 = min(NUM_CLASSES - 1, args.class_end)
    if c0 > c1:
        raise ValueError(f'invalid class range [{c0}, {c1}]')

    out_dir = args.out_dir.resolve()
    n_plan = count_planned(out_dir, c0, c1, args.samples_per_class, args.skip_existing)
    print(f'out_dir={out_dir}')
    print(f'classes [{c0}, {c1}] x {args.samples_per_class} = '
          f'{(c1 - c0 + 1) * args.samples_per_class} slots, to_generate={n_plan}')
    print(f'polar_quant={args.polar_quant} cfg={args.cfg} top_k={args.top_k} top_p={args.top_p} '
          f'more_smooth=False')
    if args.dry_run:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print('WARNING: CUDA not available; FID sampling will be extremely slow on CPU.')
    _, var = build_models(args.model_depth, device)
    run_tag = setup_polar(var, args.polar_quant, args.kmeans_codebook)
    print(f'polar mode: {run_tag} on {device}')

    torch.backends.cudnn.benchmark = True
    done, skipped = 0, 0
    with torch.inference_mode():
        for class_idx in tqdm(range(c0, c1 + 1), desc='classes'):
            for sample_idx in range(args.samples_per_class):
                path = png_path(out_dir, class_idx, sample_idx)
                if args.skip_existing and path.is_file():
                    skipped += 1
                    continue
                g_seed = args.seed_base + class_idx * 100 + sample_idx
                label_B = torch.tensor([class_idx], device=device, dtype=torch.long)
                with torch.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
                    recon = var.autoregressive_infer_cfg(
                        B=1,
                        label_B=label_B,
                        cfg=args.cfg,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        g_seed=g_seed,
                        more_smooth=False,
                    )
                save_recon_png(recon, path)
                done += 1
    print(f'finished: wrote {done} PNGs, skipped {skipped} existing under {out_dir}')


def main() -> None:
    args = parse_args()
    if args.pack_npz_only:
        create_npz_from_sample_folder(str(args.out_dir.resolve()))
        return
    generate(args)


if __name__ == '__main__':
    main()
