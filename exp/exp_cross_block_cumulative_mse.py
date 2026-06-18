"""
Cross-block propagated NMSE (baseline vs polar quant), sampled every N depths.

Default plot uses NMSE at depth d only (propagation already included in block-d output).
Optional --cumulative adds sum_{b=1..d} NMSE_b (legacy, can double-count propagation).

Usage:
  cd VAR_polarQuant
  python exp/exp_cross_block_cumulative_mse.py --class-labels 22 45 123 --out-prefix labels3

Outputs (default):
  polar_quant_dumps/cross_block_mse/<prefix>nrmse_percent_at_depth_every4.png
  polar_quant_dumps/cross_block_mse/<prefix>abs_mse_at_depth_every4.png
  polar_quant_dumps/cross_block_mse/<prefix>cosine_distance_at_depth_every4.png
  polar_quant_dumps/cross_block_mse/<prefix>block_error_at_depth_every4.json

With --cumulative (additional):
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
from utils.angle_quant import POLAR_QUANT_CONFIGS, register_theta2_kmeans_codebook
from utils.theta2_kmeans import load_theta2_codebook

OUT_DIR = ROOT / "polar_quant_dumps" / "cross_block_mse"
THETA2_KMEANS_CODEBOOK = ROOT / "polar_quant_dumps" / "theta2_kmeans" / "codebook.json"
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
    var.enable_polar_k_cache(False)
    try:
        run_infer(var, label_B, device=device)
    finally:
        var_mod.sample_with_top_k_top_p_ = orig_sampler

    for h in hooks:
        h.remove()
    return block_outputs, stage_indices


def compute_method_block_metrics(
    var,
    label_B: torch.Tensor,
    device: str,
    method: str,
    baseline_outputs: Dict[int, List[torch.Tensor]],
    forced_stage_indices: List[torch.Tensor],
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

    var.enable_polar_k_cache(True, dump_session=None, polar_quant=method)
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


def main() -> None:
    global SEED
    args = parse_args()
    SEED = int(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device != "cuda":
        print("warning: CUDA unavailable; this run will be slow.")

    maybe_register_kmeans_codebook()

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    vae_ckpt = ROOT / "vae_ch160v4096z32.pth"
    var_ckpt = ROOT / f"var_d{MODEL_DEPTH}.pth"
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=patch_nums,
        num_classes=1000,
        depth=MODEL_DEPTH,
        shared_aln=False,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval()
    var.eval()

    depth = len(var.blocks)
    sample_idx = depth_samples(depth, every=EVERY_N_DEPTH)
    sample_depths = (sample_idx + 1).tolist()

    label_list = [int(x) for x in args.class_labels]
    if not label_list:
        raise ValueError("no class labels provided")
    print(f"running labels={label_list}, seed={SEED}, cumulative={args.cumulative}")

    method_to_block_nmse_sum: Dict[str, np.ndarray] = {m: np.zeros(depth, dtype=np.float64) for m in METHODS}
    method_to_block_nrmse_pct_sum: Dict[str, np.ndarray] = {m: np.zeros(depth, dtype=np.float64) for m in METHODS}
    method_to_block_abs_mse_sum: Dict[str, np.ndarray] = {m: np.zeros(depth, dtype=np.float64) for m in METHODS}
    method_to_block_cos_sum: Dict[str, np.ndarray] = {m: np.zeros(depth, dtype=np.float64) for m in METHODS}
    method_to_nrmse_at_depth_sum: Dict[str, np.ndarray] = {m: np.zeros(len(sample_idx), dtype=np.float64) for m in METHODS}
    method_to_abs_at_depth_sum: Dict[str, np.ndarray] = {m: np.zeros(len(sample_idx), dtype=np.float64) for m in METHODS}
    method_to_cos_at_depth_sum: Dict[str, np.ndarray] = {m: np.zeros(len(sample_idx), dtype=np.float64) for m in METHODS}
    method_to_cum_at_depth_sum: Dict[str, np.ndarray] = {m: np.zeros(len(sample_idx), dtype=np.float64) for m in METHODS}

    for label in label_list:
        label_B = torch.tensor([label], device=device)
        print(f"[label={label}] collecting baseline block outputs and sampled token path...")
        baseline_outputs, baseline_stage_indices = collect_baseline_block_outputs_and_indices(var, label_B, device=device)

        for method in METHODS:
            if method not in POLAR_QUANT_CONFIGS and method != "int6_kmeans_int4":
                raise ValueError(f"unknown method: {method}")
            print(f"[label={label}] computing method={method}...")
            per_block_nmse, per_block_abs_mse, per_block_cos = compute_method_block_metrics(
                var,
                label_B,
                device=device,
                method=method,
                baseline_outputs=baseline_outputs,
                forced_stage_indices=baseline_stage_indices,
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
    method_to_block_nmse: Dict[str, List[float]] = {
        m: (method_to_block_nmse_sum[m] / n_labels).tolist() for m in METHODS
    }
    method_to_block_nrmse_percent: Dict[str, List[float]] = {
        m: (method_to_block_nrmse_pct_sum[m] / n_labels).tolist() for m in METHODS
    }
    method_to_nrmse_percent_at_depth: Dict[str, List[float]] = {
        m: (method_to_nrmse_at_depth_sum[m] / n_labels).tolist() for m in METHODS
    }
    method_to_block_abs_mse: Dict[str, List[float]] = {
        m: (method_to_block_abs_mse_sum[m] / n_labels).tolist() for m in METHODS
    }
    method_to_abs_mse_at_depth: Dict[str, List[float]] = {
        m: (method_to_abs_at_depth_sum[m] / n_labels).tolist() for m in METHODS
    }
    method_to_block_cosine_distance: Dict[str, List[float]] = {
        m: (method_to_block_cos_sum[m] / n_labels).tolist() for m in METHODS
    }
    method_to_cosine_distance_at_depth: Dict[str, List[float]] = {
        m: (method_to_cos_at_depth_sum[m] / n_labels).tolist() for m in METHODS
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.out_prefix}_" if args.out_prefix else ""

    nrmse_fig_path = OUT_DIR / f"{prefix}nrmse_percent_at_depth_every4.png"
    _plot_lines(
        sample_depths,
        method_to_nrmse_percent_at_depth,
        ylabel="NRMSE at depth d (%)",
        title="Propagated block NRMSE vs baseline (FP16 K)",
        out_path=nrmse_fig_path,
    )
    abs_fig_path = OUT_DIR / f"{prefix}abs_mse_at_depth_every4.png"
    cos_fig_path = OUT_DIR / f"{prefix}cosine_distance_at_depth_every4.png"
    _plot_lines(
        sample_depths,
        method_to_abs_mse_at_depth,
        ylabel="Mean squared error at depth d",
        title="Absolute MSE (not / ||h_ref||^2) vs baseline",
        out_path=abs_fig_path,
    )
    _plot_lines(
        sample_depths,
        method_to_cosine_distance_at_depth,
        ylabel="Cosine distance at depth d",
        title="1 - cos(h_quant, h_ref) at block output",
        out_path=cos_fig_path,
    )

    json_path = OUT_DIR / f"{prefix}block_error_at_depth_every4.json"
    payload = {
        "model_depth": MODEL_DEPTH,
        "seed": SEED,
        "cfg": CFG,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "forced_sampling_path": True,
        "class_labels": label_list,
        "num_class_labels": len(label_list),
        "every_n_depth": EVERY_N_DEPTH,
        "sample_depths": sample_depths,
        "metric": {
            "nmse_at_block_b": "sum((x_q-x_r)^2) / sum(x_r^2), last AR stage only",
            "nrmse_percent_at_block_b": "sqrt(nmse_at_block_b) * 100",
            "abs_mse_at_block_b": "mean((x_q-x_r)^2), not divided by sum(x_r^2)",
            "cosine_distance_at_block_b": "1 - dot(x_q,x_r)/(||x_q||*||x_r||)",
            "num_ar_stages": _num_ar_stages(var),
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
        method_to_cum_at_depth: Dict[str, List[float]] = {
            m: (method_to_cum_at_depth_sum[m] / n_labels).tolist() for m in METHODS
        }
        method_to_cum_at_depth_pct = {m: [v * 100.0 for v in vals] for m, vals in method_to_cum_at_depth.items()}
        cum_fig = OUT_DIR / f"{prefix}cumulative_nmse_every4.png"
        cum_fig_pct = OUT_DIR / f"{prefix}cumulative_nmse_percent_every4.png"
        _plot_lines(
            sample_depths,
            method_to_cum_at_depth,
            ylabel="Cumulative NMSE (sum over blocks 1..d)",
            title="Legacy: summed per-block NMSE vs baseline",
            out_path=cum_fig,
        )
        _plot_lines(
            sample_depths,
            method_to_cum_at_depth_pct,
            ylabel="Cumulative NMSE (%)",
            title="Legacy: summed per-block NMSE vs baseline",
            out_path=cum_fig_pct,
        )
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
    print(f"at depth={final_depth} (avg over labels):")
    for method in METHODS:
        nrmse_pct = method_to_nrmse_percent_at_depth[method][-1]
        abs_m = method_to_abs_mse_at_depth[method][-1]
        cos_d = method_to_cosine_distance_at_depth[method][-1]
        print(f"  {method:20s}  NRMSE={nrmse_pct:.4f}%  abs_mse={abs_m:.6f}  cos_dist={cos_d:.6f}")


if __name__ == "__main__":
    main()
