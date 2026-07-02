"""
Attention-level drift diagnostics for polar KV quantization.

This script answers the first high-value questions in the drift checklist:
  - Does K quantization move attention scores / softmax distributions?
  - Is the local attention output error mostly from K, V, or their interaction?
  - Which layer/head/scale is most sensitive?

It runs a baseline pass to record the sampled token path, then runs quantized
passes on the same forced token path. During the quantized pass, SelfAttention
is wrapped to compute diagnostics from the same Q and the same FP K/V cache:

  score_fp = Q K_fp^T
  score_kq = Q K_q^T
  attn_fp = softmax(score_fp)
  attn_kq = softmax(score_kq)
  out_fp  = attn_fp V_fp
  out_K   = attn_kq V_fp
  out_V   = attn_fp V_q
  out_KV  = attn_kq V_q

The model's original forward still runs normally; the wrapper only records
side-channel metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import models.var as var_mod
from models import build_vae_var
from models.basic_var import SelfAttention
from utils.angle_quant import (
    POLAR_QUANT_CONFIGS,
    PolarQuantConfig,
    register_per_level_codebook,
    register_theta2_kmeans_codebook,
    register_theta2_kmeans_codebook_v,
)
from utils.polar_kv_quant import PolarK64Batch
from utils.theta2_kmeans import load_theta2_codebook

NUM_CLASSES = 1000
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
DEFAULT_CLASSES = (45, 135, 246)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention drift diagnostics for polar KV quantization")
    p.add_argument("--model-depth", type=int, default=30, choices=(16, 20, 24, 30))
    p.add_argument("--methods", type=str, nargs="+", default=["uniform_int4", "fp6_e2m3", "int6_kmeans_int4"])
    p.add_argument("--class-labels", type=int, nargs="+", default=list(DEFAULT_CLASSES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--top-k", type=int, default=900)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--quant-v", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-json", type=Path, default=ROOT / "polar_quant_dumps" / "attention_drift_diag.json")
    p.add_argument("--kmeans-codebook", type=Path, default=None)
    p.add_argument("--theta2-levels", type=str, default=None,
                   help="per-level codebook assignment for int6_kmeans_int4, e.g. 1,1,3,3,3")
    p.add_argument("--topk", type=int, nargs="+", default=[1, 4, 8, 16],
                   help="attention top-k overlap values")
    return p.parse_args()


def set_infer_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_models(depth: int, device: str):
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    vae_ckpt = ROOT / "vae_ch160v4096z32.pth"
    var_ckpt = ROOT / f"var_d{depth}.pth"
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=PATCH_NUMS,
        num_classes=NUM_CLASSES, depth=depth, shared_aln=False,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    return vae, var


def register_method_codebooks(method: str, args: argparse.Namespace) -> str:
    method = method.lower().strip()
    if method != "int6_kmeans_int4":
        if method not in POLAR_QUANT_CONFIGS:
            raise ValueError(f"unknown method {method!r}; known={sorted(POLAR_QUANT_CONFIGS)}")
        return method

    if args.theta2_levels:
        level_ids = [int(x.strip()) for x in args.theta2_levels.split(",")]
        if len(level_ids) != 5:
            raise ValueError("--theta2-levels must have 5 comma-separated entries")
        cb_cache: Dict[int, list] = {}
        for n in set(level_ids):
            cb_path = ROOT / "polar_quant_dumps" / f"theta2_kmeans_d{args.model_depth}_T{n}" / "codebook.json"
            centers, _ = load_theta2_codebook(cb_path)
            cb_cache[n] = [float(c) for c in centers]
        config_name = f"int6_kmeans_int4_L{''.join(str(n) for n in level_ids)}"
        register_per_level_codebook(
            [cb_cache[n] for n in level_ids],
            config_name=config_name,
            label=f"INT6 theta1 + per-level kmeans theta2 ({args.theta2_levels})",
        )
        if args.quant_v:
            v_cache: Dict[int, list] = {}
            all_v = True
            for n in set(level_ids):
                v_path = ROOT / "polar_quant_dumps" / f"theta2_kmeans_d{args.model_depth}_v_T{n}" / "codebook.json"
                if not v_path.is_file():
                    all_v = False
                    break
                centers, _ = load_theta2_codebook(v_path)
                v_cache[n] = [float(c) for c in centers]
            if all_v:
                register_per_level_codebook(
                    [v_cache[n] for n in level_ids],
                    config_name=f"{config_name}_v",
                    label=f"INT6 theta1 + per-level kmeans theta2 V ({args.theta2_levels})",
                )
        return config_name

    cb = args.kmeans_codebook
    if cb is None:
        cb = ROOT / "polar_quant_dumps" / f"theta2_kmeans_d{args.model_depth}" / "codebook.json"
        if args.model_depth == 16 and not cb.is_file():
            cb = ROOT / "polar_quant_dumps" / "theta2_kmeans" / "codebook.json"
    centers, _ = load_theta2_codebook(cb)
    register_theta2_kmeans_codebook(centers)
    v_cb = cb.parent.parent / f"{cb.parent.name}_v" / "codebook.json"
    if args.quant_v and v_cb.is_file():
        v_centers, _ = load_theta2_codebook(v_cb)
        register_theta2_kmeans_codebook_v(v_centers)
    return "int6_kmeans_int4"


def run_infer(
    var,
    label_B: torch.Tensor,
    args: argparse.Namespace,
    forced_stage_indices: Optional[List[torch.Tensor]] = None,
) -> None:
    set_infer_seeds(args.seed)
    orig_sampler = var_mod.sample_with_top_k_top_p_
    counter = 0

    if forced_stage_indices is not None:
        def _forced_sampler(logits_BlV, top_k=0, top_p=0.0, rng=None, num_samples=1):
            nonlocal counter
            idx = forced_stage_indices[counter].to(logits_BlV.device)
            counter += 1
            return idx.unsqueeze(-1)
        var_mod.sample_with_top_k_top_p_ = _forced_sampler

    with torch.inference_mode():
        try:
            with torch.autocast("cuda", enabled=(label_B.device.type == "cuda"), dtype=torch.float16):
                _ = var.autoregressive_infer_cfg(
                    B=1,
                    label_B=label_B,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    g_seed=args.seed,
                    more_smooth=False,
                )
        finally:
            var_mod.sample_with_top_k_top_p_ = orig_sampler

    if forced_stage_indices is not None and counter != len(forced_stage_indices):
        raise RuntimeError(f"forced sampler consumed {counter}, expected {len(forced_stage_indices)}")


def collect_baseline_tokens(var, label_B: torch.Tensor, args: argparse.Namespace) -> List[torch.Tensor]:
    stage_indices: List[torch.Tensor] = []
    orig_sampler = var_mod.sample_with_top_k_top_p_

    def _record_sampler(logits_BlV, top_k=0, top_p=0.0, rng=None, num_samples=1):
        out = orig_sampler(logits_BlV, top_k=top_k, top_p=top_p, rng=rng, num_samples=num_samples)
        stage_indices.append(out[:, :, 0].detach().cpu())
        return out

    var_mod.sample_with_top_k_top_p_ = _record_sampler
    var.set_polar_quant(None)
    try:
        run_infer(var, label_B, args)
    finally:
        var_mod.sample_with_top_k_top_p_ = orig_sampler
    return stage_indices


def _bhll_scores(q_blhc: torch.Tensor, k_blhc: torch.Tensor, scale: float) -> torch.Tensor:
    q = q_blhc.permute(0, 2, 1, 3).float()
    k = k_blhc.permute(0, 2, 1, 3).float()
    return torch.matmul(q * scale, k.transpose(-2, -1))


def _attn_entropy(attn: torch.Tensor) -> torch.Tensor:
    return -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)


def _cos_dist(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a.float(), b.float(), dim=dim, eps=1e-12)


def _rel_l2(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (a.float() - b.float()).norm(dim=dim) / a.float().norm(dim=dim).clamp_min(1e-12)


class MeanAccumulator:
    def __init__(self):
        self.sum: Dict[str, float] = defaultdict(float)
        self.count: Dict[str, int] = defaultdict(int)

    def add(self, key: str, value: float) -> None:
        if math.isfinite(value):
            self.sum[key] += float(value)
            self.count[key] += 1

    def mean(self, key: str) -> float:
        return self.sum[key] / max(self.count[key], 1)

    def keys(self) -> Iterable[str]:
        return self.sum.keys()


class AttentionDriftRecorder:
    def __init__(self, config: PolarQuantConfig, quant_v: bool, topk: List[int]):
        self.config = config
        self.quant_v = quant_v
        self.topk = sorted(set(int(k) for k in topk if k > 0))
        self.acc = MeanAccumulator()
        self.rows: List[dict] = []
        self._handles = []
        self._orig_forward = None
        self._call_idx: Dict[int, int] = defaultdict(int)
        self._fp_k_cache: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._fp_v_cache: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def reset_run_state(self) -> None:
        """Reset side caches for one autoregressive inference run."""
        self._call_idx = defaultdict(int)
        self._fp_k_cache = defaultdict(list)
        self._fp_v_cache = defaultdict(list)

    def install(self):
        if self._orig_forward is not None:
            raise RuntimeError("recorder already installed")
        self._orig_forward = SelfAttention.forward
        recorder = self

        def _wrapped_forward(attn_self: SelfAttention, x, attn_bias):
            recorder.record_before_forward(attn_self, x, attn_bias)
            return recorder._orig_forward(attn_self, x, attn_bias)

        SelfAttention.forward = _wrapped_forward

    def remove(self):
        if self._orig_forward is not None:
            SelfAttention.forward = self._orig_forward
            self._orig_forward = None

    def _maybe_v_config(self) -> PolarQuantConfig:
        v_key = f"{self.config.name}_v"
        if self.quant_v and v_key in POLAR_QUANT_CONFIGS:
            return POLAR_QUANT_CONFIGS[v_key]
        return self.config

    def record_before_forward(self, attn_self: SelfAttention, x: torch.Tensor, attn_bias) -> None:
        if not attn_self.caching:
            return

        B, L, C = x.shape
        qkv = F.linear(
            input=x,
            weight=attn_self.mat_qkv.weight,
            bias=torch.cat((attn_self.q_bias, attn_self.zero_k_bias, attn_self.v_bias)),
        ).view(B, L, 3, attn_self.num_heads, attn_self.head_dim)
        main_type = qkv.dtype

        using_flash = attn_self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or attn_self.using_xform:
            q, k, v = qkv.unbind(dim=2)  # BLHc
            q_blhc, k_blhc, v_blhc = q, k, v
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc
            q_blhc = q.permute(0, 2, 1, 3)
            k_blhc = k.permute(0, 2, 1, 3)
            v_blhc = v.permute(0, 2, 1, 3)

        if attn_self.attn_l2_norm:
            scale_mul = attn_self.scale_mul_1H11.clamp_max(attn_self.max_scale_mul).exp()
            scale_mul_blhc = scale_mul.transpose(1, 2)
            q_blhc = F.normalize(q_blhc, dim=-1).mul(scale_mul_blhc)
            k_blhc = F.normalize(k_blhc, dim=-1)

        bi = int(attn_self.block_idx)
        si = self._call_idx[bi]
        self._call_idx[bi] += 1

        self._fp_k_cache[bi].append(k_blhc.detach())
        self._fp_v_cache[bi].append(v_blhc.detach())
        k_fp = torch.cat(self._fp_k_cache[bi], dim=1)
        v_fp = torch.cat(self._fp_v_cache[bi], dim=1)

        k_q = PolarK64Batch.from_k(k_fp, config=self.config).decode(config=self.config, dtype=main_type)
        if self.quant_v:
            v_config = self._maybe_v_config()
            v_q = PolarK64Batch.from_k(v_fp, config=v_config).decode(config=v_config, dtype=main_type)
        else:
            v_q = v_fp

        with torch.no_grad():
            score_fp = _bhll_scores(q_blhc, k_fp, attn_self.scale)
            score_kq = _bhll_scores(q_blhc, k_q, attn_self.scale)
            attn_fp = score_fp.softmax(dim=-1)
            attn_kq = score_kq.softmax(dim=-1)

            out_fp = torch.matmul(attn_fp, v_fp.permute(0, 2, 1, 3).float())
            out_k = torch.matmul(attn_kq, v_fp.permute(0, 2, 1, 3).float())
            out_v = torch.matmul(attn_fp, v_q.permute(0, 2, 1, 3).float())
            out_kv = torch.matmul(attn_kq, v_q.permute(0, 2, 1, 3).float())

            delta = score_kq - score_fp
            score_rmse = delta.pow(2).mean(dim=(-2, -1)).sqrt()
            score_ref = score_fp.pow(2).mean(dim=(-2, -1)).sqrt().clamp_min(1e-12)
            score_rel_rmse = score_rmse / score_ref
            score_cos = 1.0 - _cos_dist(score_fp.flatten(-2), score_kq.flatten(-2), dim=-1)

            k_cos = _cos_dist(k_fp, k_q, dim=-1).mean(dim=1)
            k_rel = _rel_l2(k_fp, k_q, dim=-1).mean(dim=1)
            v_cos = _cos_dist(v_fp, v_q, dim=-1).mean(dim=1)
            v_rel = _rel_l2(v_fp, v_q, dim=-1).mean(dim=1)

            kl = (attn_fp.clamp_min(1e-12) * (attn_fp.clamp_min(1e-12).log() - attn_kq.clamp_min(1e-12).log())).sum(dim=-1)
            attn_l1 = (attn_fp - attn_kq).abs().sum(dim=-1)
            entropy_fp = _attn_entropy(attn_fp)
            entropy_kq = _attn_entropy(attn_kq)

            out_k_cos = _cos_dist(out_fp, out_k, dim=-1)
            out_v_cos = _cos_dist(out_fp, out_v, dim=-1)
            out_kv_cos = _cos_dist(out_fp, out_kv, dim=-1)
            out_k_rel = _rel_l2(out_fp, out_k, dim=-1)
            out_v_rel = _rel_l2(out_fp, out_v, dim=-1)
            out_kv_rel = _rel_l2(out_fp, out_kv, dim=-1)

            H = attn_self.num_heads
            for h in range(H):
                prefix = f"s{si:02d}.b{bi:02d}.h{h:02d}"
                vals = {
                    "score_delta_mean": delta[:, h].mean().item(),
                    "score_delta_std": delta[:, h].std().item(),
                    "score_delta_max_abs": delta[:, h].abs().max().item(),
                    "score_rel_rmse": score_rel_rmse[:, h].mean().item(),
                    "score_corr": score_cos[:, h].mean().item(),
                    "k_cos_dist": k_cos[:, h].mean().item(),
                    "k_rel_l2": k_rel[:, h].mean().item(),
                    "v_cos_dist": v_cos[:, h].mean().item(),
                    "v_rel_l2": v_rel[:, h].mean().item(),
                    "attn_kl": kl[:, h].mean().item(),
                    "attn_l1": attn_l1[:, h].mean().item(),
                    "attn_entropy_fp": entropy_fp[:, h].mean().item(),
                    "attn_entropy_q": entropy_kq[:, h].mean().item(),
                    "out_konly_cos_dist": out_k_cos[:, h].mean().item(),
                    "out_vonly_cos_dist": out_v_cos[:, h].mean().item(),
                    "out_kv_cos_dist": out_kv_cos[:, h].mean().item(),
                    "out_konly_rel_l2": out_k_rel[:, h].mean().item(),
                    "out_vonly_rel_l2": out_v_rel[:, h].mean().item(),
                    "out_kv_rel_l2": out_kv_rel[:, h].mean().item(),
                }
                for k_top in self.topk:
                    kk = min(k_top, attn_fp.shape[-1])
                    top_fp = attn_fp[:, h].topk(kk, dim=-1).indices
                    top_q = attn_kq[:, h].topk(kk, dim=-1).indices
                    if kk == 1:
                        vals["attn_top1_match"] = (top_fp == top_q).float().mean().item()
                    overlap = (top_fp.unsqueeze(-1) == top_q.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / kk
                    vals[f"attn_top{kk}_overlap"] = overlap.mean().item()
                for name, val in vals.items():
                    self.acc.add(f"{prefix}.{name}", val)
                    self.acc.add(f"s{si:02d}.b{bi:02d}.{name}", val)
                    self.acc.add(f"b{bi:02d}.h{h:02d}.{name}", val)
                    self.acc.add(name, val)

    def summary(self) -> dict:
        flat = {k: self.acc.mean(k) for k in sorted(self.acc.keys())}
        global_keys = [
            "score_delta_mean", "score_delta_std", "score_delta_max_abs", "score_rel_rmse", "score_corr",
            "k_cos_dist", "k_rel_l2", "v_cos_dist", "v_rel_l2",
            "attn_kl", "attn_l1", "attn_entropy_fp", "attn_entropy_q", "attn_top1_match",
            "attn_top4_overlap", "attn_top8_overlap", "attn_top16_overlap",
            "out_konly_cos_dist", "out_vonly_cos_dist", "out_kv_cos_dist",
            "out_konly_rel_l2", "out_vonly_rel_l2", "out_kv_rel_l2",
        ]
        return {
            "global": {k: flat[k] for k in global_keys if k in flat},
            "flat": flat,
        }


def run_method(var, method: str, args: argparse.Namespace, label_list: List[int]) -> dict:
    config_name = register_method_codebooks(method, args)
    config = POLAR_QUANT_CONFIGS[config_name]
    recorder = AttentionDriftRecorder(config=config, quant_v=args.quant_v, topk=args.topk)
    for label in label_list:
        label_B = torch.tensor([label], device=next(var.parameters()).device, dtype=torch.long)
        baseline_tokens = collect_baseline_tokens(var, label_B, args)
        var.set_polar_quant(config_name, quant_v=args.quant_v)
        recorder.reset_run_state()
        recorder.install()
        try:
            run_infer(var, label_B, args, forced_stage_indices=baseline_tokens)
        finally:
            recorder.remove()
            var.set_polar_quant(None)
    return recorder.summary()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA unavailable; this diagnostic will be very slow on CPU.")
    _, var = build_models(args.model_depth, device)
    label_list = [int(x) for x in args.class_labels]
    payload = {
        "model_depth": args.model_depth,
        "class_labels": label_list,
        "seed": args.seed,
        "cfg": args.cfg,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "quant_v": args.quant_v,
        "mode": "forced_token_path_attention_local",
        "metric_notes": {
            "score_corr": "cosine similarity between flattened score_fp and score_q per head",
            "attn_kl": "mean KL(attn_fp || attn_q) per query",
            "out_konly": "attn_q @ V_fp vs attn_fp @ V_fp",
            "out_vonly": "attn_fp @ V_q vs attn_fp @ V_fp",
            "out_kv": "attn_q @ V_q vs attn_fp @ V_fp",
        },
        "methods": {},
    }
    for method in args.methods:
        print(f"[diag] method={method}")
        payload["methods"][method] = run_method(var, method, args, label_list)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(f"saved -> {args.out_json}")
    for method, data in payload["methods"].items():
        g = data["global"]
        print(
            f"{method:20s} "
            f"score_rel_rmse={g.get('score_rel_rmse', float('nan')):.4g} "
            f"attn_kl={g.get('attn_kl', float('nan')):.4g} "
            f"top1={g.get('attn_top1_match', float('nan')):.3f} "
            f"Kout={g.get('out_konly_rel_l2', float('nan')):.4g} "
            f"Vout={g.get('out_vonly_rel_l2', float('nan')):.4g} "
            f"KVout={g.get('out_kv_rel_l2', float('nan')):.4g}"
        )


if __name__ == "__main__":
    main()
