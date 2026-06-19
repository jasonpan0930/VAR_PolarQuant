"""
Cross-block propagated NMSE (baseline vs polar quant), sampled every N depths.

Modes (--mode):
  forced (default): same token-path via forced sampler; compares last AR stage only
  free: free-running — same seed, no forced sampler; quant model generates its own tokens
  both: run both forced and free

Forced mode outputs:
  polar_quant_dumps/cross_block_mse/<prefix>nrmse_percent_at_depth_every4.png
  polar_quant_dumps/cross_block_mse/<prefix>abs_mse_at_depth_every4.png
  polar_quant_dumps/cross_block_mse/<prefix>cosine_distance_at_depth_every4.png
  polar_quant_dumps/cross_block_mse/<prefix>block_error_at_depth_every4.json

Free-running mode outputs:
  .../<prefix>heatmap_nrmse_{method}_{mode_suffix}.png         (scale × block heatmap)
  .../<prefix>heatmap_cos_{method}_{mode_suffix}.png           (scale × block cosine distance)
  .../<prefix>perscale_nrmse_{method}_{mode_suffix}.png        (per-scale line plot)
  .../<prefix>perscale_cos_{method}_{mode_suffix}.png          (per-scale cosine distance)
  .../<prefix>token_disagree_{mode_suffix}.png                 (token disagree bar)
  .../<prefix>fhat_nrmse_{mode_suffix}.png                     (f_hat NRMSE per scale)
  .../<prefix>fhat_cos_{mode_suffix}.png                       (f_hat cosine distance per scale)
  .../<prefix>fhat_spatial_cos_scale9_{method}_...png          (f_hat per-position cos, scale 9, 16×16)
  .../<prefix>fhat_spatial_cos_mean_{mode_suffix}.png          (mean per-position f_hat cos per scale)
  .../<prefix>free_running_metrics_{mode_suffix}.json

Usage:
  cd VAR_polarQuant
  python exp/exp_cross_block_cumulative_mse.py --class-labels 22 45 123 --out-prefix labels3
  python exp/exp_cross_block_cumulative_mse.py --class-labels 22 45 123 --mode free
  python exp/exp_cross_block_cumulative_mse.py --class-labels 22 45 123 --mode both

With --cumulative (additional, forced mode):
  .../<prefix>cumulative_nmse_every4.png
  .../<prefix>cumulative_nmse_percent_every4.png

Metric at block b on the **last AR stage only** (si = num_stages-1, longest KV context):
  NMSE_b = sum((x_quant - x_ref)^2) / sum(x_ref^2)
  NRMSE_b (%) = sqrt(NMSE_b) * 100
  (Earlier stages are skipped: pooling them made depth-30 look artificially low.)

At sampled depth d (block index d-1): plot NRMSE (%) and other metrics (no sum across blocks).

Also:
  abs_mse = mean((x_quant - x_ref)^2)   # not divided by ||x_ref||^2
  cosine_distance = 1 - cos(x_quant, x_ref)
"""
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch

import models.var as var_mod
from models import build_vae_var
from utils.angle_quant import POLAR_QUANT_CONFIGS, register_theta2_kmeans_codebook, register_theta2_kmeans_codebook_v, register_per_level_codebook
from utils.theta2_kmeans import load_theta2_codebook

OUT_DIR = ROOT / "polar_quant_dumps" / "cross_block_mse"
THETA2_KMEANS_CODEBOOK = ROOT / "polar_quant_dumps" / "theta2_kmeans_d30" / "codebook.json"
THETA2_KMEANS_CODEBOOK_V = ROOT / "polar_quant_dumps" / "theta2_kmeans_d30_v" / "codebook.json"
MODEL_DEPTH = 30
BATCH_SIZE = 1
CLASS_LABELS = (22, 45, 123, 437, 701)
SEED, CFG, TOP_K, TOP_P = 0, 4, 900, 0.95
METHODS = (
    "uniform_int4",
    "e2m1_fp4",
    "fp6_e3m2",
    "fp6_e2m3",
    "int6_kmeans_int4",
)
EVERY_N_DEPTH = 1


@dataclass
class BlockErrorAccumulator:
    sse: float = 0.0
    ref_energy: float = 0.0
    cur_energy: float = 0.0
    dot: float = 0.0
    n_elem: int = 0

    def add(self, ref: torch.Tensor, cur: torch.Tensor) -> None:
        diff = cur - ref
        self.sse += float((diff * diff).sum().item())
        self.ref_energy += float((ref * ref).sum().item())
        self.cur_energy += float((cur * cur).sum().item())
        self.dot += float((ref * cur).sum().item())
        self.n_elem += diff.numel()

    def nmse(self) -> float:
        return self.sse / max(self.ref_energy, 1e-12)

    def nrmse_percent(self) -> float:
        return 100.0 * math.sqrt(self.nmse())

    def abs_mse(self) -> float:
        """Mean squared error; denominator is element count, not ||x_ref||^2."""
        return self.sse / max(self.n_elem, 1)

    def cosine_distance(self) -> float:
        denom = math.sqrt(max(self.ref_energy, 0.0) * max(self.cur_energy, 0.0))
        if denom < 1e-12:
            return 0.0
        cos_sim = max(-1.0, min(1.0, self.dot / denom))
        return 1.0 - cos_sim


def set_infer_seeds() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def run_infer(var, label_B: torch.Tensor, device: str, forced_stage_indices: List[torch.Tensor] = None) -> None:
    set_infer_seeds()
    orig_sampler = var_mod.sample_with_top_k_top_p_
    counter = 0

    if forced_stage_indices is not None:
        def _forced_sampler(logits_BlV, top_k=0, top_p=0.0, rng=None, num_samples=1):
            nonlocal counter
            if counter >= len(forced_stage_indices):
                raise RuntimeError(
                    f"forced sampler out of range: counter={counter}, total={len(forced_stage_indices)}"
                )
            idx = forced_stage_indices[counter].to(logits_BlV.device)
            counter += 1
            return idx.unsqueeze(-1)

        var_mod.sample_with_top_k_top_p_ = _forced_sampler

    with torch.inference_mode():
        try:
            with torch.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                _ = var.autoregressive_infer_cfg(
                    B=BATCH_SIZE,
                    label_B=label_B,
                    cfg=CFG,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    g_seed=SEED,
                )
        finally:
            var_mod.sample_with_top_k_top_p_ = orig_sampler

    if forced_stage_indices is not None and counter != len(forced_stage_indices):
        raise RuntimeError(
            f"forced sampler not fully consumed: used={counter}, total={len(forced_stage_indices)}"
        )


def maybe_register_kmeans_codebook() -> None:
    if THETA2_KMEANS_CODEBOOK.is_file():
        centers, _ = load_theta2_codebook(THETA2_KMEANS_CODEBOOK)
        register_theta2_kmeans_codebook(centers)
    else:
        print(f"warning: {THETA2_KMEANS_CODEBOOK} not found, int6_kmeans_int4 may fail")
    if THETA2_KMEANS_CODEBOOK_V.is_file():
        v_centers, _ = load_theta2_codebook(THETA2_KMEANS_CODEBOOK_V)
        register_theta2_kmeans_codebook_v(v_centers)
        print(f"loaded V codebook from {THETA2_KMEANS_CODEBOOK_V}")
    else:
        print(f"warning: {THETA2_KMEANS_CODEBOOK_V} not found; V will fall back to K codebook")


def _num_ar_stages(var) -> int:
    return len(var.patch_nums)


def collect_baseline_block_outputs_and_indices(
    var, label_B: torch.Tensor, device: str
) -> tuple[Dict[int, List[torch.Tensor]], List[torch.Tensor]]:
    """Per block: one tensor per AR stage; we compare only the last stage (index num_stages-1)."""
    num_stages = _num_ar_stages(var)
    block_outputs: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(var.blocks))}
    stage_indices: List[torch.Tensor] = []
    hooks = []

    def make_hook(block_idx: int):
        def _hook(_module, _inputs, output):
            block_outputs[block_idx].append(output.detach().float().cpu())
            if len(block_outputs[block_idx]) > num_stages:
                raise RuntimeError(
                    f"block {block_idx}: more than {num_stages} forwards (unexpected AR loop)"
                )
        return _hook

    for i, block in enumerate(var.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    orig_sampler = var_mod.sample_with_top_k_top_p_

    def _record_sampler(logits_BlV, top_k=0, top_p=0.0, rng=None, num_samples=1):
        out = orig_sampler(logits_BlV, top_k=top_k, top_p=top_p, rng=rng, num_samples=num_samples)
        stage_indices.append(out[:, :, 0].detach().cpu())
        return out

    var_mod.sample_with_top_k_top_p_ = _record_sampler
    var.set_polar_quant(None)
    try:
        run_infer(var, label_B, device=device)
    finally:
        var_mod.sample_with_top_k_top_p_ = orig_sampler

    for h in hooks:
        h.remove()
    return block_outputs, stage_indices


def collect_free_run_outputs(
    var, label_B: torch.Tensor, device: str,
    method: str = None, quant_v: bool = True,
) -> tuple[Dict[int, List[torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
    """
    Free-running inference (no forced sampler). Returns:
      - block_outputs: {block_idx: [stage0_out, stage1_out, ..., stage9_out]}
      - stage_indices: list of tensors (one per scale), each (B, patch_num²)
      - f_hat_snapshots: list of tensors (one per scale), each (B, Cvae, HW, HW)
    """
    num_stages = _num_ar_stages(var)
    num_blocks = len(var.blocks)
    block_outputs: Dict[int, List[torch.Tensor]] = {bi: [] for bi in range(num_blocks)}
    stage_indices: List[torch.Tensor] = []
    f_hat_snapshots: List[torch.Tensor] = []
    hooks = []

    def make_block_hook(bi: int):
        def _hook(_module, _inputs, output):
            block_outputs[bi].append(output.detach().float().cpu())
            if len(block_outputs[bi]) > num_stages:
                raise RuntimeError(f"block {bi}: more than {num_stages} forwards (unexpected AR loop)")
        return _hook

    for bi, block in enumerate(var.blocks):
        hooks.append(block.register_forward_hook(make_block_hook(bi)))

    # Hook sampler to record stage indices
    orig_sampler = var_mod.sample_with_top_k_top_p_
    def _record_sampler(logits_BlV, top_k=0, top_p=0.0, rng=None, num_samples=1):
        out = orig_sampler(logits_BlV, top_k=top_k, top_p=top_p, rng=rng, num_samples=num_samples)
        stage_indices.append(out[:, :, 0].detach().cpu())
        return out
    var_mod.sample_with_top_k_top_p_ = _record_sampler

    # Hook f_hat after each scale's get_next_autoregressive_input
    vae_proxy = var.vae_quant_proxy[0]
    orig_get_next = vae_proxy.get_next_autoregressive_input
    def _patched_get_next(si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor):
        new_f_hat, next_token_map = orig_get_next(si, SN, f_hat, h_BChw)
        f_hat_snapshots.append(new_f_hat.detach().float().cpu().clone())
        return new_f_hat, next_token_map
    vae_proxy.get_next_autoregressive_input = _patched_get_next

    try:
        var.set_polar_quant(method, quant_v=quant_v) if method is not None else var.set_polar_quant(None)
        run_infer(var, label_B, device=device)
    finally:
        var_mod.sample_with_top_k_top_p_ = orig_sampler
        vae_proxy.get_next_autoregressive_input = orig_get_next
        for h in hooks:
            h.remove()

    for bi in range(num_blocks):
        if len(block_outputs[bi]) != num_stages:
            raise RuntimeError(f"block {bi}: expected {num_stages} stages, got {len(block_outputs[bi])}")
    if len(stage_indices) != num_stages:
        raise RuntimeError(f"sampler: expected {num_stages} stages, got {len(stage_indices)}")
    if len(f_hat_snapshots) != num_stages:
        raise RuntimeError(f"f_hat hook: expected {num_stages} stages, got {len(f_hat_snapshots)}")

    return block_outputs, stage_indices, f_hat_snapshots


def compute_method_block_metrics(
    var,
    label_B: torch.Tensor,
    device: str,
    method: str,
    baseline_outputs: Dict[int, List[torch.Tensor]],
    forced_stage_indices: List[torch.Tensor],
    quant_v: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (per_block_nmse, per_block_abs_mse, per_block_cosine_distance)."""
    num_stages = _num_ar_stages(var)
    last_stage = num_stages - 1
    accum = [BlockErrorAccumulator() for _ in range(len(var.blocks))]
    call_idx = [0 for _ in range(len(var.blocks))]
    hooks = []

    def make_hook(block_idx: int):
        def _hook(_module, _inputs, output):
            j = call_idx[block_idx]
            ref_list = baseline_outputs[block_idx]
            if j >= len(ref_list):
                raise RuntimeError(f"block {block_idx}: compare calls exceed baseline calls ({j} >= {len(ref_list)})")
            if j == last_stage:
                ref = ref_list[j]
                cur = output.detach().float().cpu()
                if cur.shape != ref.shape:
                    raise RuntimeError(
                        f"block {block_idx} stage {j}: shape mismatch {tuple(cur.shape)} vs {tuple(ref.shape)}"
                    )
                accum[block_idx].add(ref, cur)
            call_idx[block_idx] += 1
        return _hook

    for i, block in enumerate(var.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    var.set_polar_quant(method, quant_v=quant_v)
    run_infer(var, label_B, device=device, forced_stage_indices=forced_stage_indices)

    for h in hooks:
        h.remove()

    for i in range(len(var.blocks)):
        if call_idx[i] != num_stages:
            raise RuntimeError(
                f"block {i}: expected {num_stages} forwards, got {call_idx[i]}"
            )
        if len(baseline_outputs[i]) != num_stages:
            raise RuntimeError(
                f"block {i}: baseline has {len(baseline_outputs[i])} stages, expected {num_stages}"
            )

    nmse = np.array([a.nmse() for a in accum], dtype=np.float64)
    abs_mse = np.array([a.abs_mse() for a in accum], dtype=np.float64)
    cos_dist = np.array([a.cosine_distance() for a in accum], dtype=np.float64)
    return nmse, abs_mse, cos_dist


def depth_samples(depth: int, every: int) -> np.ndarray:
    """0-based block indices, every N depths, always include last block (depth-1)."""
    idx = list(range(every - 1, depth, every))
    if depth - 1 not in idx:
        idx.append(depth - 1)
    return np.array(idx, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-block NMSE vs baseline (forced token path); default = per-depth, not summed"
    )
    p.add_argument(
        "--class-labels",
        type=int,
        nargs="+",
        default=list(CLASS_LABELS),
        help="ImageNet class ids to average over.",
    )
    p.add_argument("--seed", type=int, default=SEED, help="Global seed.")
    p.add_argument("--out-prefix", type=str, default="", help="Output filename prefix.")
    p.add_argument(
        "--cumulative",
        action="store_true",
        help="Also plot/save sum_{b=1..d} NMSE_b (legacy; propagation is already in NMSE at depth d).",
    )
    p.add_argument(
        "--quant-v", action=argparse.BooleanOptionalAction, default=True,
        help="Quantize V cache (default: True); --no-quant-v for K-only comparison",
    )
    p.add_argument(
        "--per-level-kmeans", type=str, nargs="+", default=[],
        metavar=("NAME", "LEVELS"),
        help="Add per-level kmeans configs. Format: NAME LEVEL_ASSIGNMENT pairs.\n"
             "Example: --per-level-kmeans my_11144 1,1,1,4,4 my_12345 1,2,3,4,5\n"
             "Loads codebooks from polar_quant_dumps/theta2_kmeans_d30_T{N}/codebook.json",
    )
    p.add_argument(
        "--mode", type=str, default="forced", choices=["forced", "free", "both"],
        help="Comparison mode:\n"
             "  forced: current default — forced token path, last scale only\n"
             "  free:   free-running — same seed, no forced sampler, all scales\n"
             "  both:   run both forced and free",
    )
    return p.parse_args()


def _plot_lines(
    sample_depths: List[int],
    y_dict: Dict[str, List[float]],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for method, vals in y_dict.items():
        ax.plot(sample_depths, vals, marker="o", linewidth=2.0, label=method)
    ax.set_xlabel(f"Depth (sampled every {EVERY_N_DEPTH} blocks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(
    data: np.ndarray,
    ylabel: str,
    title: str,
    out_path: Path,
    num_stages: int,
    num_blocks: int,
    xlabel: str = "Block depth",
    cbar_label: str = "NRMSE (%)",
    cmap: str = "YlOrRd",
) -> None:
    """Plot a 2D heatmap: rows = scales, cols = blocks."""
    fig, ax = plt.subplots(figsize=(max(8, num_blocks * 0.28), max(4, num_stages * 0.45)))
    im = ax.imshow(data, aspect="auto", origin="upper", cmap=cmap, vmin=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(num_blocks))
    ax.set_xticklabels([f"{b+1}" for b in range(num_blocks)], fontsize=7, rotation=90)
    ax.set_yticks(range(num_stages))
    ax.set_yticklabels([f"scale {s}" for s in range(num_stages)], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_token_disagree(
    disagree_pct: Dict[str, List[float]],
    num_stages: int,
    out_path: Path,
    quant_mode: str = "KV",
) -> None:
    """Bar chart: token disagreement rate per scale."""
    fig, ax = plt.subplots(figsize=(max(8, num_stages * 0.7), 5.2))
    n_methods = len(disagree_pct)
    bar_width = 0.8 / max(n_methods, 1)
    x = np.arange(num_stages)
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    for mi, (method, vals) in enumerate(disagree_pct.items()):
        ax.bar(x + mi * bar_width - (n_methods - 1) * bar_width / 2, vals, bar_width,
               label=method, color=colors[mi % 10])
    ax.set_xlabel("Scale")
    ax.set_ylabel("Token disagreement rate")
    ax.set_title(f"Token disagreement per scale (free-running, {quant_mode} quant)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in range(num_stages)])
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fhat_metric(
    fhat_vals: Dict[str, List[float]],
    num_stages: int,
    out_path: Path,
    ylabel: str = "f_hat NRMSE (%)",
    quant_mode: str = "KV",
) -> None:
    """Line plot: f_hat metric per scale."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = list(range(num_stages))
    for method, vals in fhat_vals.items():
        ax.plot(x, vals, marker="o", linewidth=2.0, label=method)
    ax.set_xlabel("Scale")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (free-running, {quant_mode} quant)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_per_scale_lines(
    vals_by_scale: Dict[int, List[float]],
    num_blocks: int,
    out_path: Path,
    method_label: str,
    ylabel: str = "NRMSE (%)",
    quant_mode: str = "KV",
) -> None:
    """Per-scale line plot: metric vs block depth for each scale."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = list(range(1, num_blocks + 1))
    colors = plt.cm.viridis(np.linspace(0, 1, len(vals_by_scale)))
    for si in sorted(vals_by_scale.keys()):
        color = colors[si] if si < len(colors) else None
        ax.plot(x, vals_by_scale[si], marker=".", linewidth=1.2,
                label=f"scale {si}", color=color)
    ax.set_xlabel("Block depth")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-scale block {ylabel} ({method_label}, {quant_mode} quant)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    global SEED
    args = parse_args()
    SEED = int(args.seed)
    mode = args.mode

    if mode == "both":
        main_forced(args)
        print("\n" + "=" * 60 + "\n")
        main_free(args)
    elif mode == "free":
        main_free(args)
    else:
        main_forced(args)


def _build_model(device: str):
    """Build and load VAR model, return (vae, var)."""
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    vae_ckpt = ROOT / "vae_ch160v4096z32.pth"
    var_ckpt = ROOT / f"var_d{MODEL_DEPTH}.pth"
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums, num_classes=1000,
        depth=MODEL_DEPTH, shared_aln=False,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval()
    var.eval()
    return vae, var


def _register_per_level_configs(per_level_args: list, quant_v: bool) -> tuple:
    """Register per-level kmeans configs and return (methods_list, methods)."""
    methods_list = list(METHODS)
    pl = per_level_args
    if len(pl) % 2 != 0:
        raise ValueError("--per-level-kmeans requires even number of args (pairs of NAME LEVELS)")
    for i in range(0, len(pl), 2):
        config_name = pl[i]
        level_str = pl[i + 1]
        level_ids = [int(x.strip()) for x in level_str.split(',')]
        if len(level_ids) != 5:
            raise ValueError(f"--per-level-kmeans {config_name}: LEVELS must have 5 entries, got {len(level_ids)}")
        cb_cache: Dict[int, list] = {}
        for n in set(level_ids):
            cb_path = ROOT / 'polar_quant_dumps' / f'theta2_kmeans_d{MODEL_DEPTH}_T{n}' / 'codebook.json'
            if not cb_path.is_file():
                raise FileNotFoundError(f'per-level codebook missing: {cb_path}')
            centers, _ = load_theta2_codebook(cb_path)
            cb_cache[n] = list(float(c) for c in centers)
            print(f'  [per-level] loaded CB{n} ({len(cb_cache[n])}-entry) from {cb_path}')
        centers_per_level = [cb_cache[n] for n in level_ids]
        label = f'INT6 θ₁ + per-level K-means θ₂ ({level_str})'
        register_per_level_codebook(centers_per_level, config_name=config_name, label=label)
        methods_list.append(config_name)
        if quant_v:
            v_cb_cache: Dict[int, list] = {}
            all_v_found = True
            for n in set(level_ids):
                v_path = ROOT / 'polar_quant_dumps' / f'theta2_kmeans_d{MODEL_DEPTH}_v_T{n}' / 'codebook.json'
                if v_path.is_file():
                    v_centers, _ = load_theta2_codebook(v_path)
                    v_cb_cache[n] = list(float(c) for c in v_centers)
                else:
                    all_v_found = False
            if all_v_found:
                v_centers_per_level = [v_cb_cache[n] for n in level_ids]
                v_label = f'INT6 θ₁ + per-level K-means θ₂ (V, {level_str})'
                register_per_level_codebook(v_centers_per_level, config_name=f'{config_name}_v', label=v_label)
                print(f'  [per-level] registered V config: {config_name}_v')
            else:
                print(f'  [per-level] warning: V codebooks incomplete for {config_name}, V will fall back to K codebook')
        print(f'  [per-level] registered config: {config_name} ({level_str})')
    return methods_list, tuple(methods_list)


def main_forced(args) -> None:
    """Current forced-sampler mode: compare last scale only."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device != "cuda":
        print("warning: CUDA unavailable; this run will be slow.")
    maybe_register_kmeans_codebook()
    quant_v = args.quant_v
    methods_list, methods = _register_per_level_configs(args.per_level_kmeans, quant_v)

    vae, var = _build_model(device)

    depth = len(var.blocks)
    sample_idx = depth_samples(depth, every=EVERY_N_DEPTH)
    sample_depths = (sample_idx + 1).tolist()
    num_stages = _num_ar_stages(var)

    label_list = [int(x) for x in args.class_labels]
    if not label_list:
        raise ValueError("no class labels provided")
    quant_mode = "KV" if quant_v else "K-only"
    print(f"running labels={label_list}, seed={SEED}, cumulative={args.cumulative}, quant_v={quant_v} ({quant_mode}), mode=forced")

    method_to_block_nmse_sum: Dict[str, np.ndarray] = {m: np.zeros(depth, dtype=np.float64) for m in methods}
    method_to_block_nrmse_pct_sum: Dict[str, np.ndarray] = {m: np.zeros(depth, dtype=np.float64) for m in methods}
    method_to_block_abs_mse_sum: Dict[str, np.ndarray] = {m: np.zeros(depth, dtype=np.float64) for m in methods}
    method_to_block_cos_sum: Dict[str, np.ndarray] = {m: np.zeros(depth, dtype=np.float64) for m in methods}
    method_to_nrmse_at_depth_sum: Dict[str, np.ndarray] = {m: np.zeros(len(sample_idx), dtype=np.float64) for m in methods}
    method_to_abs_at_depth_sum: Dict[str, np.ndarray] = {m: np.zeros(len(sample_idx), dtype=np.float64) for m in methods}
    method_to_cos_at_depth_sum: Dict[str, np.ndarray] = {m: np.zeros(len(sample_idx), dtype=np.float64) for m in methods}
    method_to_cum_at_depth_sum: Dict[str, np.ndarray] = {m: np.zeros(len(sample_idx), dtype=np.float64) for m in methods}

    for label in label_list:
        label_B = torch.tensor([label], device=device)
        print(f"[label={label}] collecting baseline block outputs and sampled token path...")
        baseline_outputs, baseline_stage_indices = collect_baseline_block_outputs_and_indices(var, label_B, device=device)

        for method in methods:
            if method not in POLAR_QUANT_CONFIGS and method != "int6_kmeans_int4":
                raise ValueError(f"unknown method: {method}")
            print(f"[label={label}] computing method={method}...")
            per_block_nmse, per_block_abs_mse, per_block_cos = compute_method_block_metrics(
                var, label_B, device=device, method=method,
                baseline_outputs=baseline_outputs,
                forced_stage_indices=baseline_stage_indices,
                quant_v=quant_v,
            )
            per_block_nrmse_pct = 100.0 * np.sqrt(per_block_nmse)
            method_to_block_nmse_sum[method] += per_block_nmse
            method_to_block_nrmse_pct_sum[method] += per_block_nrmse_pct
            method_to_block_abs_mse_sum[method] += per_block_abs_mse
            method_to_block_cos_sum[method] += per_block_cos
            method_to_nrmse_at_depth_sum[method] += per_block_nrmse_pct[sample_idx]
            method_to_abs_at_depth_sum[method] += per_block_abs_mse[sample_idx]
            method_to_cos_at_depth_sum[method] += per_block_cos[sample_idx]
            if args.cumulative:
                method_to_cum_at_depth_sum[method] += np.cumsum(per_block_nmse)[sample_idx]

    n_labels = float(len(label_list))
    method_to_block_nmse: Dict[str, List[float]] = {m: (method_to_block_nmse_sum[m] / n_labels).tolist() for m in methods}
    method_to_block_nrmse_percent: Dict[str, List[float]] = {m: (method_to_block_nrmse_pct_sum[m] / n_labels).tolist() for m in methods}
    method_to_nrmse_percent_at_depth: Dict[str, List[float]] = {m: (method_to_nrmse_at_depth_sum[m] / n_labels).tolist() for m in methods}
    method_to_block_abs_mse: Dict[str, List[float]] = {m: (method_to_block_abs_mse_sum[m] / n_labels).tolist() for m in methods}
    method_to_abs_mse_at_depth: Dict[str, List[float]] = {m: (method_to_abs_at_depth_sum[m] / n_labels).tolist() for m in methods}
    method_to_block_cosine_distance: Dict[str, List[float]] = {m: (method_to_block_cos_sum[m] / n_labels).tolist() for m in methods}
    method_to_cosine_distance_at_depth: Dict[str, List[float]] = {m: (method_to_cos_at_depth_sum[m] / n_labels).tolist() for m in methods}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.out_prefix}_" if args.out_prefix else ""
    mode_suffix = "KV" if quant_v else "Konly"

    nrmse_fig_path = OUT_DIR / f"{prefix}nrmse_percent_at_depth_{mode_suffix}_every4.png"
    _plot_lines(sample_depths, method_to_nrmse_percent_at_depth,
                ylabel="NRMSE at depth d (%)",
                title=f"Propagated block NRMSE vs baseline ({quant_mode} quant, forced token path)",
                out_path=nrmse_fig_path)
    abs_fig_path = OUT_DIR / f"{prefix}abs_mse_at_depth_{mode_suffix}_every4.png"
    cos_fig_path = OUT_DIR / f"{prefix}cosine_distance_at_depth_{mode_suffix}_every4.png"
    _plot_lines(sample_depths, method_to_abs_mse_at_depth,
                ylabel="Mean squared error at depth d",
                title=f"Absolute MSE (not / ||h_ref||^2) vs baseline ({quant_mode})",
                out_path=abs_fig_path)
    _plot_lines(sample_depths, method_to_cosine_distance_at_depth,
                ylabel="Cosine distance at depth d",
                title=f"1 - cos(h_quant, h_ref) at block output ({quant_mode})",
                out_path=cos_fig_path)

    json_path = OUT_DIR / f"{prefix}block_error_at_depth_{mode_suffix}_every4.json"
    payload = {
        "model_depth": MODEL_DEPTH, "seed": SEED, "cfg": CFG, "top_k": TOP_K, "top_p": TOP_P,
        "forced_sampling_path": True, "quant_v": quant_v, "quant_mode": quant_mode,
        "class_labels": label_list, "num_class_labels": len(label_list),
        "every_n_depth": EVERY_N_DEPTH, "sample_depths": sample_depths,
        "metric": {
            "nmse_at_block_b": "sum((x_q-x_r)^2) / sum(x_r^2), last AR stage only",
            "nrmse_percent_at_block_b": "sqrt(nmse_at_block_b) * 100",
            "abs_mse_at_block_b": "mean((x_q-x_r)^2), not divided by sum(x_r^2)",
            "cosine_distance_at_block_b": "1 - dot(x_q,x_r)/(||x_q||*||x_r||)",
            "num_ar_stages": num_stages,
            "at_depth_d": "block index b = d-1; includes propagation from earlier blocks",
            "x_ref": "baseline block output (enable_polar_k_cache=False)",
            "x_quant": "polar-quant block output (same forced token path as baseline)",
        },
        "per_block_nmse": method_to_block_nmse,
        "per_block_nrmse_percent": method_to_block_nrmse_percent,
        "per_block_abs_mse": method_to_block_abs_mse,
        "per_block_cosine_distance": method_to_block_cosine_distance,
        "nrmse_percent_at_depth_every_n": method_to_nrmse_percent_at_depth,
        "abs_mse_at_depth_every_n": method_to_abs_mse_at_depth,
        "cosine_distance_at_depth_every_n": method_to_cosine_distance_at_depth,
    }
    if args.cumulative:
        method_to_cum_at_depth: Dict[str, List[float]] = {m: (method_to_cum_at_depth_sum[m] / n_labels).tolist() for m in methods}
        method_to_cum_at_depth_pct = {m: [v * 100.0 for v in vals] for m, vals in method_to_cum_at_depth.items()}
        cum_fig = OUT_DIR / f"{prefix}cumulative_nmse_{mode_suffix}_every4.png"
        cum_fig_pct = OUT_DIR / f"{prefix}cumulative_nmse_percent_{mode_suffix}_every4.png"
        _plot_lines(sample_depths, method_to_cum_at_depth, ylabel="Cumulative NMSE (sum over blocks 1..d)",
                    title="Legacy: summed per-block NMSE vs baseline", out_path=cum_fig)
        _plot_lines(sample_depths, method_to_cum_at_depth_pct, ylabel="Cumulative NMSE (%)",
                    title="Legacy: summed per-block NMSE vs baseline", out_path=cum_fig_pct)
        payload["metric"]["cumulative_nmse_at_depth_d"] = "sum_{b=1..d} nmse_at_block_b (optional, can double-count propagation)"
        payload["cumulative_nmse_every_n"] = method_to_cum_at_depth
        payload["cumulative_nmse_percent_every_n"] = method_to_cum_at_depth_pct
        print(f"saved legacy cumulative plot -> {cum_fig}")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"saved plot -> {nrmse_fig_path}")
    print(f"saved plot -> {abs_fig_path}")
    print(f"saved plot -> {cos_fig_path}")
    print(f"saved data -> {json_path}")
    final_depth = sample_depths[-1]
    print(f"at depth={final_depth} ({quant_mode}, avg over labels):")
    for method in methods:
        nrmse_pct = method_to_nrmse_percent_at_depth[method][-1]
        abs_m = method_to_abs_mse_at_depth[method][-1]
        cos_d = method_to_cosine_distance_at_depth[method][-1]
        print(f"  {method:20s}  NRMSE={nrmse_pct:.4f}%  abs_mse={abs_m:.6f}  cos_dist={cos_d:.6f}")


def main_free(args) -> None:
    """Free-running mode: same seed, no forced sampler, compare all scales.
    Produces:
      - heatmap: NRMSE % (scale × block)
      - token disagreement bar per scale
      - f_hat NRMSE per scale
      - per-scale line plot: NRMSE vs block depth
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device != "cuda":
        print("warning: CUDA unavailable; this run will be slow.")
    maybe_register_kmeans_codebook()
    quant_v = args.quant_v
    methods_list, methods = _register_per_level_configs(args.per_level_kmeans, quant_v)

    vae, var = _build_model(device)

    num_blocks = len(var.blocks)
    num_stages = _num_ar_stages(var)
    label_list = [int(x) for x in args.class_labels]
    if not label_list:
        raise ValueError("no class labels provided")
    quant_mode = "KV" if quant_v else "K-only"
    print(f"running labels={label_list}, seed={SEED}, quant_v={quant_v} ({quant_mode}), mode=free")

    # ── accumulators per method ──
    # NRMSE heatmap: (num_stages, num_blocks) nrmse_pct sum
    method_heatmap_sum: Dict[str, np.ndarray] = {m: np.zeros((num_stages, num_blocks), dtype=np.float64) for m in methods}
    # Cosine distance heatmap: (num_stages, num_blocks) cosine distance sum
    method_cos_heatmap_sum: Dict[str, np.ndarray] = {m: np.zeros((num_stages, num_blocks), dtype=np.float64) for m in methods}
    # heatmap sum counts
    method_heatmap_count: Dict[str, int] = {m: 0 for m in methods}
    # token disagreement per scale: (num_stages,) sum
    method_tok_disagree_sum: Dict[str, np.ndarray] = {m: np.zeros(num_stages, dtype=np.float64) for m in methods}
    # f_hat NRMSE per scale: (num_stages,) sum
    method_fhat_sum: Dict[str, np.ndarray] = {m: np.zeros(num_stages, dtype=np.float64) for m in methods}
    # f_hat cosine distance per scale: (num_stages,) sum
    method_fhat_cos_sum: Dict[str, np.ndarray] = {m: np.zeros(num_stages, dtype=np.float64) for m in methods}
    # f_hat per-spatial-position cosine distance: (num_stages, H, W) sum
    # H, W are determined from first f_hat snapshot (all f_hat snapshots are same spatial size)
    method_fhat_spatial_cos_sum: Dict[str, np.ndarray] = {m: None for m in methods}
    method_fhat_spatial_count: Dict[str, int] = {m: 0 for m in methods}

    for label in label_list:
        label_B = torch.tensor([label], device=device)
        print(f"[label={label}] collecting baseline free-running outputs...")
        set_infer_seeds()
        base_outs, base_tokens, base_fhat = collect_free_run_outputs(
            var, label_B, device=device, method=None,
        )

        for method in methods:
            print(f"[label={label}] collecting quant free-running method={method}...")
            set_infer_seeds()  # reset to same seed for fair comparison
            quant_outs, quant_tokens, quant_fhat = collect_free_run_outputs(
                var, label_B, device=device, method=method, quant_v=quant_v,
            )

            # ── per-scale block NRMSE & cosine distance (heatmaps) ──
            for si in range(num_stages):
                for bi in range(num_blocks):
                    ref = base_outs[bi][si]
                    cur = quant_outs[bi][si]
                    if cur.shape != ref.shape:
                        raise RuntimeError(
                            f"scale {si} block {bi}: shape mismatch {tuple(cur.shape)} vs {tuple(ref.shape)}"
                        )
                    # NRMSE
                    sse = float((cur - ref).pow(2).sum().item())
                    ref_norm = float(ref.pow(2).sum().item())
                    nmse = sse / max(ref_norm, 1e-12)
                    nrmse_pct = 100.0 * math.sqrt(nmse)
                    method_heatmap_sum[method][si, bi] += nrmse_pct
                    # cosine distance
                    dot = float((ref * cur).sum().item())
                    cur_norm = float(cur.pow(2).sum().item())
                    denom = math.sqrt(max(ref_norm, 0.0) * max(cur_norm, 0.0))
                    cos_sim = max(-1.0, min(1.0, dot / max(denom, 1e-12)))
                    method_cos_heatmap_sum[method][si, bi] += (1.0 - cos_sim)
            method_heatmap_count[method] += 1

            # ── token disagreement per scale ──
            for si in range(num_stages):
                base_tok = base_tokens[si].flatten()
                quant_tok = quant_tokens[si].flatten()
                n_total = base_tok.numel()
                n_same = (base_tok == quant_tok).sum().item()
                disagree_rate = 1.0 - n_same / max(n_total, 1)
                method_tok_disagree_sum[method][si] += disagree_rate

            # ── f_hat NRMSE & cosine distance per scale ──
            for si in range(num_stages):
                bf = base_fhat[si]
                qf = quant_fhat[si]
                if bf is None or qf is None:
                    fhat_nrmse = 0.0
                    fhat_cos = 0.0
                else:
                    sse = float((qf - bf).pow(2).sum().item())
                    ref_norm = float(bf.pow(2).sum().item())
                    fhat_nrmse = 100.0 * math.sqrt(sse / max(ref_norm, 1e-12))
                    # cosine distance
                    dot = float((bf * qf).sum().item())
                    cur_norm = float(qf.pow(2).sum().item())
                    denom = math.sqrt(max(ref_norm, 0.0) * max(cur_norm, 0.0))
                    cos_sim = max(-1.0, min(1.0, dot / max(denom, 1e-12)))
                    fhat_cos = 1.0 - cos_sim
                method_fhat_sum[method][si] += fhat_nrmse
                method_fhat_cos_sum[method][si] += fhat_cos

            # ── f_hat per-spatial-position cosine distance ──
            for si in range(num_stages):
                bf = base_fhat[si]
                qf = quant_fhat[si]
                if bf is not None and qf is not None:
                    B, C, H, W = bf.shape
                    # per-position cosine distance: (H, W)
                    bf_flat = bf[0].reshape(C, -1)  # (C, H*W)
                    qf_flat = qf[0].reshape(C, -1)
                    b_norms = bf_flat.norm(dim=0)     # (H*W,)
                    q_norms = qf_flat.norm(dim=0)     # (H*W,)
                    dots = (bf_flat * qf_flat).sum(dim=0)  # (H*W,)
                    denom = (b_norms * q_norms).clamp(min=1e-12)
                    cos_sim = (dots / denom).clamp(-1.0, 1.0)
                    spatial_cos_dist = (1.0 - cos_sim).reshape(H, W).numpy().astype(np.float64)
                    if method_fhat_spatial_cos_sum[method] is None:
                        method_fhat_spatial_cos_sum[method] = np.zeros((num_stages, H, W), dtype=np.float64)
                    method_fhat_spatial_cos_sum[method][si] += spatial_cos_dist
            method_fhat_spatial_count[method] += 1

    # ── average over labels ──
    n_labels = float(len(label_list))
    method_heatmap: Dict[str, np.ndarray] = {
        m: method_heatmap_sum[m] / max(method_heatmap_count[m], 1) for m in methods
    }
    method_cos_heatmap: Dict[str, np.ndarray] = {
        m: method_cos_heatmap_sum[m] / max(method_heatmap_count[m], 1) for m in methods
    }
    method_tok_disagree: Dict[str, List[float]] = {
        m: (method_tok_disagree_sum[m] / n_labels).tolist() for m in methods
    }
    method_fhat_nrmse: Dict[str, List[float]] = {
        m: (method_fhat_sum[m] / n_labels).tolist() for m in methods
    }
    method_fhat_cos: Dict[str, List[float]] = {
        m: (method_fhat_cos_sum[m] / n_labels).tolist() for m in methods
    }
    # per-spatial-position f_hat cosine: (num_stages, H, W) avg
    method_fhat_spatial: Dict[str, np.ndarray] = {}
    method_fhat_spatial_mean_per_scale: Dict[str, List[float]] = {}  # per-scale mean cos dist
    for m in methods:
        if method_fhat_spatial_cos_sum[m] is not None:
            cnt = max(method_fhat_spatial_count[m], 1)
            method_fhat_spatial[m] = method_fhat_spatial_cos_sum[m] / cnt
            method_fhat_spatial_mean_per_scale[m] = method_fhat_spatial[m].mean(axis=(1, 2)).tolist()
        else:
            method_fhat_spatial[m] = np.zeros((num_stages, 1, 1))
            method_fhat_spatial_mean_per_scale[m] = [0.0] * num_stages

    # ── save outputs ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.out_prefix}_" if args.out_prefix else ""
    mode_suffix = "KV_free" if quant_v else "Konly_free"

    # NRMSE heatmaps
    for method in methods:
        hm_path = OUT_DIR / f"{prefix}heatmap_nrmse_{method}_{mode_suffix}.png"
        _plot_heatmap(
            method_heatmap[method],
            ylabel="Scale",
            title=f"Free-running block NRMSE % ({method}, {quant_mode})",
            out_path=hm_path,
            num_stages=num_stages,
            num_blocks=num_blocks,
            cbar_label="NRMSE (%)",
        )
        print(f"saved NRMSE heatmap -> {hm_path}")

    # Cosine distance heatmaps
    for method in methods:
        cos_hm_path = OUT_DIR / f"{prefix}heatmap_cos_{method}_{mode_suffix}.png"
        _plot_heatmap(
            method_cos_heatmap[method],
            ylabel="Scale",
            title=f"Free-running block cosine distance ({method}, {quant_mode})",
            out_path=cos_hm_path,
            num_stages=num_stages,
            num_blocks=num_blocks,
            cbar_label="cosine distance",
        )
        print(f"saved cosine heatmap -> {cos_hm_path}")

    # Per-scale NRMSE line plots (one per method)
    for method in methods:
        per_scale_nrmse: Dict[int, List[float]] = {}
        for si in range(num_stages):
            per_scale_nrmse[si] = method_heatmap[method][si, :].tolist()
        line_path = OUT_DIR / f"{prefix}perscale_nrmse_{method}_{mode_suffix}.png"
        _plot_per_scale_lines(
            per_scale_nrmse, num_blocks, line_path,
            method_label=method, ylabel="NRMSE (%)", quant_mode=quant_mode,
        )
        print(f"saved per-scale NRMSE lines -> {line_path}")

    # Per-scale cosine distance line plots (one per method)
    for method in methods:
        per_scale_cos: Dict[int, List[float]] = {}
        for si in range(num_stages):
            per_scale_cos[si] = method_cos_heatmap[method][si, :].tolist()
        line_path = OUT_DIR / f"{prefix}perscale_cos_{method}_{mode_suffix}.png"
        _plot_per_scale_lines(
            per_scale_cos, num_blocks, line_path,
            method_label=method, ylabel="cosine distance", quant_mode=quant_mode,
        )
        print(f"saved per-scale cosine lines -> {line_path}")

    # Token disagreement
    tok_path = OUT_DIR / f"{prefix}token_disagree_{mode_suffix}.png"
    _plot_token_disagree(method_tok_disagree, num_stages, tok_path, quant_mode=quant_mode)
    print(f"saved token disagreement -> {tok_path}")

    # f_hat NRMSE
    fhat_path = OUT_DIR / f"{prefix}fhat_nrmse_{mode_suffix}.png"
    _plot_fhat_metric(method_fhat_nrmse, num_stages, fhat_path, ylabel="f_hat NRMSE (%)", quant_mode=quant_mode)
    print(f"saved f_hat NRMSE -> {fhat_path}")

    # f_hat cosine distance
    fhat_cos_path = OUT_DIR / f"{prefix}fhat_cos_{mode_suffix}.png"
    _plot_fhat_metric(method_fhat_cos, num_stages, fhat_cos_path, ylabel="f_hat cosine distance", quant_mode=quant_mode)
    print(f"saved f_hat cosine distance -> {fhat_cos_path}")

    # f_hat per-spatial-position cosine distance — spatial heatmaps (last scale, one per method)
    H_spat, W_spat = method_fhat_spatial[methods[0]].shape[1], method_fhat_spatial[methods[0]].shape[2]
    for method in methods:
        spat_hm_path = OUT_DIR / f"{prefix}fhat_spatial_cos_scale9_{method}_{mode_suffix}.png"
        _plot_heatmap(
            method_fhat_spatial[method][-1],  # last scale
            ylabel="spatial row",
            title=f"f_hat per-position cos distance scale 9 ({method}, {quant_mode})",
            out_path=spat_hm_path,
            num_stages=H_spat,
            num_blocks=W_spat,
            xlabel="spatial column",
            cbar_label="cosine distance",
            cmap="YlOrRd",
        )
        print(f"saved f_hat spatial cos heatmap (scale 9) -> {spat_hm_path}")

    # f_hat per-spatial-position mean cosine distance per scale (line plot, all methods)
    spat_mean_path = OUT_DIR / f"{prefix}fhat_spatial_cos_mean_{mode_suffix}.png"
    _plot_fhat_metric(
        method_fhat_spatial_mean_per_scale, num_stages, spat_mean_path,
        ylabel="mean per-position cos distance",
        quant_mode=quant_mode,
    )
    print(f"saved f_hat spatial cos mean per scale -> {spat_mean_path}")

    # JSON
    json_path = OUT_DIR / f"{prefix}free_running_metrics_{mode_suffix}.json"
    payload = {
        "model_depth": MODEL_DEPTH, "seed": SEED, "cfg": CFG, "top_k": TOP_K, "top_p": TOP_P,
        "mode": "free", "quant_v": quant_v, "quant_mode": quant_mode,
        "class_labels": label_list, "num_class_labels": len(label_list),
        "num_ar_stages": num_stages,
        "num_blocks": num_blocks,
        "metric_descriptions": {
            "heatmap_nrmse": "per-scale per-block NRMSE (%), rows=scales, cols=blocks",
            "heatmap_cosine_distance": "per-scale per-block cosine distance (1 - cos_sim), rows=scales, cols=blocks",
            "token_disagree_per_scale": "fraction of tokens that differ from baseline at each scale",
            "fhat_nrmse_per_scale": "NRMSE (%) of the accumulated f_hat latent map after each scale",
            "fhat_cosine_distance_per_scale": "cosine distance of f_hat latent map at each scale",
            "fhat_spatial_cos_per_scale": "per-spatial-position f_hat cosine distance (num_stages, H, W), rows=scales, then H×W spatial",
            "fhat_spatial_cos_mean_per_scale": "mean per-position f_hat cosine distance per scale",
        },
        "heatmap_nrmse": {m: method_heatmap[m].tolist() for m in methods},
        "heatmap_cosine_distance": {m: method_cos_heatmap[m].tolist() for m in methods},
        "token_disagree_per_scale": method_tok_disagree,
        "fhat_nrmse_per_scale": method_fhat_nrmse,
        "fhat_cosine_distance_per_scale": method_fhat_cos,
        "fhat_spatial_cos_per_scale": {m: method_fhat_spatial[m].tolist() for m in methods},
        "fhat_spatial_cos_mean_per_scale": method_fhat_spatial_mean_per_scale,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"saved data -> {json_path}")

    # Summary to console
    print(f"\nToken disagreement per scale ({quant_mode}, free-running):")
    header = f"{'scale':>6s}  {'tokens':>6s}  " + "  ".join(f"{m:>20s}" for m in methods)
    print(header)
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    for si in range(num_stages):
        n_tokens = patch_nums[si] ** 2
        vals = "  ".join(f"{method_tok_disagree[m][si]*100:>19.2f}%" for m in methods)
        print(f"  {si:>4d}  {n_tokens:>6d}  {vals}")

    print(f"\nf_hat NRMSE per scale ({quant_mode}, free-running):")
    header = f"{'scale':>6s}  " + "  ".join(f"{m:>20s}" for m in methods)
    print(header)
    for si in range(num_stages):
        vals = "  ".join(f"{method_fhat_nrmse[m][si]:>19.4f}%" for m in methods)
        print(f"  {si:>4d}  {vals}")

    print(f"\nf_hat cosine distance per scale ({quant_mode}, free-running):")
    header = f"{'scale':>6s}  " + "  ".join(f"{m:>20s}" for m in methods)
    print(header)
    for si in range(num_stages):
        vals = "  ".join(f"{method_fhat_cos[m][si]:>19.6f}" for m in methods)
        print(f"  {si:>4d}  {vals}")

    print(f"\nPer-scale block NRMSE summary ({quant_mode}, free-running):")
    for method in methods:
        hm = method_heatmap[method]
        print(f"  {method:20s}  mean={hm.mean():.4f}%  max={hm.max():.4f}%  "
              f"last_scale_mean={hm[-1,:].mean():.4f}%")

    print(f"\nPer-scale block cosine distance summary ({quant_mode}, free-running):")
    for method in methods:
        chm = method_cos_heatmap[method]
        print(f"  {method:20s}  mean={chm.mean():.6f}  max={chm.max():.6f}  "
              f"last_scale_mean={chm[-1,:].mean():.6f}")

    print(f"\nf_hat per-spatial-position cos distance mean per scale ({quant_mode}, free-running):")
    header = f"{'scale':>6s}  " + "  ".join(f"{m:>20s}" for m in methods)
    print(header)
    for si in range(num_stages):
        vals = "  ".join(f"{method_fhat_spatial_mean_per_scale[m][si]:>19.6f}" for m in methods)
        print(f"  {si:>4d}  {vals}")

    print(f"\nf_hat spatial cos distance summary (scale 9, {quant_mode}, free-running):")
    for method in methods:
        smap = method_fhat_spatial[method][-1]  # last scale, (H, W)
        print(f"  {method:20s}  mean={smap.mean():.6f}  max={smap.max():.6f}  "
              f"min={smap.min():.6f}  H={smap.shape[0]} W={smap.shape[1]}")


if __name__ == "__main__":
    main()
