"""
Fit 16 θ₂ levels (4-bit) via MSE-weighted K-means on FP angles, then evaluate K MSE.

Codebook is model-specific: fit with the same depth you use at FID time.

  cd VAR_polarQuant
  python exp/exp_theta2_kmeans.py --model-depth 16
  python exp/exp_theta2_kmeans.py --model-depth 30 --refit \\
      --class-labels 22 45 123 437 701 --target-blocks 7 15 22 29

Outputs (per depth):
  polar_quant_dumps/theta2_kmeans_d<depth>/codebook.json
  polar_quant_dumps/theta2_kmeans_d<depth>/codebook_vs_uniform.png
  polar_quant_dumps/angle_plots/d<depth>/int6_kmeans_int4/k_error_global.png
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import build_vae_var
from utils.angle_quant import (
    POLAR_QUANT_CONFIGS,
    THETA2_UNIFORM_INT4,
    register_theta2_kmeans_codebook,
    register_theta2_kmeans_codebook_v,
)
from utils.polar_angle_viz import PolarAngleStatsSession, render_all_plots
from utils.theta2_kmeans import kmeans_theta2_codebook, load_theta2_codebook, save_theta2_codebook

COLLECT_CONFIG = 'uniform_int4'
BATCH_SIZE = 1
DEFAULT_CLASS_LABELS = (22, 45, 123, 437, 701)
SEED, TOP_K, TOP_P = 0, 900, 0.95
KMEANS_SEED = 0

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='MSE-weighted θ₂ K-means codebook fit')
    p.add_argument('--model-depth', type=int, default=16, choices=(16, 20, 24, 30))
    p.add_argument(
        '--refit', action='store_true',
        help='recompute codebook even if output file already exists',
    )
    p.add_argument(
        '--codebook-out', type=Path, default=None,
        help='default: polar_quant_dumps/theta2_kmeans_d<depth>/codebook.json',
    )
    p.add_argument(
        '--target-blocks', type=int, nargs='+', default=None,
        help='block indices for θ₂ collection (default: quartiles at depth/4, depth/2, 3*depth/4, depth-1)',
    )
    p.add_argument(
        '--class-labels', type=int, nargs='+', default=list(DEFAULT_CLASS_LABELS),
        help=f'ImageNet class indices, one inference each (default: {list(DEFAULT_CLASS_LABELS)})',
    )
    p.add_argument('--cfg', type=float, default=4.0, help='CFG during θ₂ collection')
    p.add_argument('--seed-base', type=int, default=SEED, help='g_seed = seed_base + class_idx')
    p.add_argument('--skip-k-error-plot', action='store_true')
    p.add_argument(
        '--collect-v', action='store_true',
        help='collect V θ₂ instead of K (for separate V codebook)',
    )
    return p.parse_args()


def default_codebook_path(depth: int) -> Path:
    legacy = ROOT / 'polar_quant_dumps' / 'theta2_kmeans' / 'codebook.json'
    if depth == 16 and legacy.is_file():
        return legacy
    return ROOT / 'polar_quant_dumps' / f'theta2_kmeans_d{depth}' / 'codebook.json'


def default_v_codebook_path(depth: int) -> Path:
    return ROOT / 'polar_quant_dumps' / f'theta2_kmeans_d{depth}_v' / 'codebook.json'


def default_target_blocks(depth: int) -> tuple[int, ...]:
    """Quartile blocks including the last layer."""
    blocks = (depth // 4, depth // 2, (3 * depth) // 4, depth - 1)
    return tuple(sorted(set(blocks)))


def collect_theta2_mse(
    var,
    label_B: torch.Tensor,
    target_blocks: tuple[int, ...],
    cfg: float,
    g_seed: int,
) -> PolarAngleStatsSession:
    angle_stats = PolarAngleStatsSession(
        target_blocks=target_blocks,
        max_samples_per_stage=15000,
        max_structured_tokens=256,
        batch_index=0,
    )
    var.set_polar_quant(COLLECT_CONFIG)
    var.set_angle_stats(angle_stats)
    torch.manual_seed(g_seed)
    random.seed(g_seed)
    np.random.seed(g_seed)
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            var.autoregressive_infer_cfg(
                B=BATCH_SIZE, label_B=label_B, cfg=cfg,
                top_k=TOP_K, top_p=TOP_P, g_seed=g_seed,
            )
    return angle_stats


def collect_multi_class_theta2(
    var,
    class_labels: Sequence[int],
    target_blocks: tuple[int, ...],
    cfg: float,
    seed_base: int,
    collect_v: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    theta2_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    per_class: list[dict] = []

    target = 'V' if collect_v else 'K'
    print(f'collecting FP θ₂ for {target} under {COLLECT_CONFIG} blocks={target_blocks} cfg={cfg}')
    for class_idx in class_labels:
        g_seed = seed_base + class_idx
        label_B = torch.tensor([class_idx], device=device)
        print(f'  class {class_idx} g_seed={g_seed} ...', flush=True)
        session = collect_theta2_mse(var, label_B, target_blocks, cfg, g_seed)
        if collect_v:
            theta2, weights = session.aggregate_theta2_mse_for_kmeans_v()
        else:
            theta2, weights = session.aggregate_theta2_mse_for_kmeans()
        theta2_parts.append(theta2)
        weight_parts.append(weights)
        per_class.append({
            'class_label': class_idx,
            'g_seed': g_seed,
            'n_theta2': int(theta2.size),
            'weight_sum': float(weights.sum()),
        })
        print(f'    θ₂ samples={theta2.size:,} weight_sum={weights.sum():.6f}')

    return np.concatenate(theta2_parts), np.concatenate(weight_parts), per_class


def plot_codebooks(kmeans_cb: np.ndarray, out_path: Path, depth: int, subtitle: str = '') -> None:
    uniform_cb = THETA2_UNIFORM_INT4.codebook_numpy()
    x = np.arange(16)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, uniform_cb, 'o-', label='uniform INT4', color='C1')
    ax.plot(x, kmeans_cb, 's-', label='K-means (MSE-weighted)', color='C0')
    ax.set_xlabel('level index (sorted)')
    ax.set_ylabel('θ₂ (rad)')
    title = f'θ₂ codebook (VAR-d{depth}): uniform vs K-means'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_k_error_plot(
    var,
    label_B: torch.Tensor,
    config_name: str,
    out_dir: Path,
    target_blocks: tuple[int, ...],
    cfg: float,
    g_seed: int,
) -> None:
    angle_stats = PolarAngleStatsSession(
        target_blocks=target_blocks,
        max_samples_per_stage=15000,
        max_structured_tokens=256,
        batch_index=0,
    )
    var.set_polar_quant(config_name)
    var.set_angle_stats(angle_stats)
    torch.manual_seed(g_seed)
    random.seed(g_seed)
    np.random.seed(g_seed)
    cfg_obj = POLAR_QUANT_CONFIGS.get(config_name, POLAR_QUANT_CONFIGS['uniform_int4'])
    print(f'--- {config_name}: {cfg_obj.label} -> {out_dir} ---')
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            var.autoregressive_infer_cfg(
                B=BATCH_SIZE, label_B=label_B, cfg=cfg,
                top_k=TOP_K, top_p=TOP_P, g_seed=g_seed,
            )
    kerr = angle_stats.aggregate_k_errors()
    render_all_plots({}, out_dir, kerr=kerr)


def main() -> None:
    args = parse_args()
    depth = args.model_depth
    target_blocks = tuple(args.target_blocks) if args.target_blocks else default_target_blocks(depth)
    class_labels = tuple(args.class_labels)
    codebook_path = args.codebook_out or (default_v_codebook_path(depth) if args.collect_v else default_codebook_path(depth))
    out_kmeans = codebook_path.parent
    config_name = 'int6_kmeans_int4_v' if args.collect_v else 'int6_kmeans_int4'
    target_label = 'V' if args.collect_v else 'K'

    print(f'device: {device}')
    print(f'model_depth={depth} target_blocks={target_blocks} class_labels={class_labels}')
    print(f'target={target_label} config={config_name} codebook_path={codebook_path}')

    vae_ckpt = ROOT / 'vae_ch160v4096z32.pth'
    var_ckpt = ROOT / f'var_d{depth}.pth'
    if not var_ckpt.is_file():
        raise FileNotFoundError(f'missing checkpoint: {var_ckpt}')

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=depth, shared_aln=False,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()

    if codebook_path.exists() and not args.refit:
        centers, meta = load_theta2_codebook(codebook_path)
        print(f'loaded existing codebook from {codebook_path} (use --refit to recompute)')
    else:
        theta2, weights, per_class = collect_multi_class_theta2(
            var, class_labels, target_blocks, args.cfg, args.seed_base,
            collect_v=args.collect_v,
        )
        print(f'total θ₂ samples: {theta2.size:,}, weight sum={weights.sum():.6f}')
        centers, meta = kmeans_theta2_codebook(theta2, weights, seed=KMEANS_SEED)
        meta.update({
            'model_depth': depth,
            'target_blocks': list(target_blocks),
            'class_labels': list(class_labels),
            'per_class': per_class,
            'cfg': args.cfg,
            'seed_base': args.seed_base,
            'collect_config': COLLECT_CONFIG,
            'collect_target': 'V' if args.collect_v else 'K',
            'l0_only': True,
        })
        save_theta2_codebook(codebook_path, centers, meta)
        print(f'saved codebook -> {codebook_path}')
        print(f'  backend={meta.get("backend")} weighted_angle_mse={meta.get("weighted_mse_angle"):.6e}')
        print(f'  uniform_angle_mse={meta.get("uniform_mse_angle"):.6e}')

    if args.collect_v:
        scheme = register_theta2_kmeans_codebook_v(centers, config_name=config_name)
    else:
        scheme = register_theta2_kmeans_codebook(centers, config_name=config_name)
    print(f'{target_label} θ₂ centroids (rad):')
    for i, c in enumerate(scheme.codebook):
        print(f'  [{i:2d}] {c:.8f}')

    n_cls, n_blk = len(class_labels), len(target_blocks)
    plot_codebooks(
        centers, out_kmeans / 'codebook_vs_uniform.png', depth,
        subtitle=f'{target_label}-K-means: {n_cls} classes × {n_blk} blocks (L0 only)',
    )
    print(f'codebook plot -> {out_kmeans / "codebook_vs_uniform.png"}')

    if not args.skip_k_error_plot and not args.collect_v:
        eval_class = class_labels[0]
        out_dir = ROOT / 'polar_quant_dumps' / 'angle_plots' / f'd{depth}' / config_name
        run_k_error_plot(
            var,
            torch.tensor([eval_class], device=device),
            config_name,
            out_dir,
            target_blocks,
            args.cfg,
            args.seed_base + eval_class,
        )
    print('done')


if __name__ == '__main__':
    main()
