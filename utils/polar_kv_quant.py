"""Hierarchical polar quantization for 64-dim K vectors.

Every encode/decode function receives config: PolarQuantConfig explicitly.
No global state reads — all quantization parameters flow through config.

Pipeline:
    encode:  K in R64 -> polar decompose -> quantize -> (q1, q2, z)
    cache:   store (q1: uint8x32, q2: uint8x31, z: fp16) in PolarKVCache
    decode:  (q1, q2, z) -> dequantize -> reconstruct K_hat -> matmul with Q
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from utils.angle_quant import DEFAULT_CONFIG, PolarQuantConfig, NUM_TREE_LEVELS
from utils.cordic import cordic_vectoring

HEAD_DIM = 64
NUM_Q1 = 32
NUM_Q2 = 31
PI = 3.141592653589793


# =====================================================================
#  Core encode: 64-dim K -> (q1, q2, z)  with explicit config
# =====================================================================

def encode_polar_k64(
    k: torch.Tensor,
    config: PolarQuantConfig = DEFAULT_CONFIG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode (..., 64) K tensor -> (q1: int64x32, q2: int64x31, z: float32).

    Stages:
      1. Pairwise polar: 32x (x0,x1) -> atan2 -> 32x theta1 -> quantize -> q1
      2. Tree merge:     32 lengths -> 16->8->4->2->1 -> 31x theta2 -> quantize -> q2
      3. z = final magnitude (root of merge tree)
    """
    assert k.shape[-1] == HEAD_DIM, f'expected last dim {HEAD_DIM}, got {k.shape[-1]}'

    # Stage 1: pairwise polar decomposition
    x0, x1 = k[..., 0::2], k[..., 1::2]
    y, theta1 = cordic_vectoring(x0, x1, config.cordic_iters)
    q1 = config.theta1.quantize(theta1)

    # Stage 2: merge tree (5 layers: 32->16->8->4->2->1)
    nodes = y
    q2_parts = []
    for li in range(NUM_TREE_LEVELS):
        a, b = nodes[..., 0::2], nodes[..., 1::2]
        nodes, theta2 = cordic_vectoring(a, b, config.cordic_iters)
        q2_parts.append(config.theta2_per_level[li].quantize(theta2))

    q2 = torch.cat(q2_parts, dim=-1)          # (..., 31)
    z = nodes[..., 0]                          # (...) root magnitude
    return q1, q2, z


# =====================================================================
#  Core decode: (q1, q2, z) -> reconstructed K_hat
# =====================================================================

def decode_polar_k64(
    q1: torch.Tensor, q2: torch.Tensor, z: torch.Tensor,
    config: PolarQuantConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Decode (q1: int64x32, q2: int64x31, z: float32) -> K_hat (..., 64)."""
    # Reverse merge tree: z -> 32 lengths y
    # q2 layout: [T1(16), T2(8), T3(4), T4(2), T5(1)], decoded right-to-left
    nodes = z.unsqueeze(-1)
    offset = NUM_Q2
    for layer in range(NUM_TREE_LEVELS):
        n = 2 ** layer
        offset -= n
        q2_slice = q2[..., offset:offset + n].to(torch.int64)
        # decode order is T5→T1 (reverse of encode T1→T5)
        theta = config.theta2_per_level[NUM_TREE_LEVELS - 1 - layer].dequantize(q2_slice)
        a = nodes * torch.cos(theta)
        b = nodes * torch.sin(theta)
        nodes = torch.stack([a, b], dim=-1).reshape(*nodes.shape[:-1], n * 2)

    # Reverse pairwise polar: y + theta1 -> (x0, x1) -> K_hat
    y = nodes                                     # (..., 32)
    theta1 = config.theta1.dequantize(q1)         # (..., 32)
    x0 = y * torch.cos(theta1)
    x1 = y * torch.sin(theta1)
    return torch.stack([x0, x1], dim=-1).reshape(*q1.shape[:-1], HEAD_DIM)


# =====================================================================
#  Tree helper (used by error analysis scripts)
# =====================================================================

def decode_y_from_polar_tree(
    q2: torch.Tensor, z_root: torch.Tensor,
    config: PolarQuantConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Decode merge tree -> 32 lengths y (..., 32)."""
    nodes = z_root.unsqueeze(-1)
    offset = NUM_Q2
    for layer in range(NUM_TREE_LEVELS):
        n = 2 ** layer
        offset -= n
        q2_slice = q2[..., offset:offset + n].to(torch.int64)
        # decode order is T5→T1 (reverse of encode T1→T5)
        theta = config.theta2_per_level[NUM_TREE_LEVELS - 1 - layer].dequantize(q2_slice)
        a = nodes * torch.cos(theta)
        b = nodes * torch.sin(theta)
        nodes = torch.stack([a, b], dim=-1).reshape(*nodes.shape[:-1], n * 2)
    return nodes


def reconstruct_k_from_y_theta1(y: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
    """y (..., 32), theta1 (..., 32) -> K (..., 64)."""
    x0 = y * torch.cos(theta1)
    x1 = y * torch.sin(theta1)
    return torch.stack([x0, x1], dim=-1).reshape(*y.shape[:-1], HEAD_DIM)


# =====================================================================
#  Internal: full encode with intermediate values (for error analysis)
# =====================================================================

def _encode_core(k: torch.Tensor, config: PolarQuantConfig):
    """Encode + return theta1/theta2 before quantization (for analysis)."""
    x0, x1 = k[..., 0::2], k[..., 1::2]
    y, theta1 = cordic_vectoring(x0, x1, config.cordic_iters)
    q1 = config.theta1.quantize(theta1)

    nodes = y
    q2_parts, theta2_parts = [], []
    for li in range(NUM_TREE_LEVELS):
        a, b = nodes[..., 0::2], nodes[..., 1::2]
        nodes, theta2 = cordic_vectoring(a, b, config.cordic_iters)
        q2_parts.append(config.theta2_per_level[li].quantize(theta2))
        theta2_parts.append(theta2)

    q2 = torch.cat(q2_parts, dim=-1)
    theta2 = torch.cat(theta2_parts, dim=-1)
    z = nodes[..., 0]
    return q1, q2, z, theta1, theta2


# =====================================================================
#  Error breakdown (optional diagnostics / codebook sanity checks)
# =====================================================================

def _vec_error(k_orig: torch.Tensor, k_hat: torch.Tensor):
    err = k_orig.float() - k_hat.float()
    mse = (err ** 2).mean(dim=-1)
    max_abs = err.abs().max(dim=-1).values
    mean_abs = err.abs().mean(dim=-1)
    return mse, max_abs, mean_abs


def polar_k64_error_breakdown(
    k: torch.Tensor,
    config: PolarQuantConfig = DEFAULT_CONFIG,
) -> Dict[str, torch.Tensor]:
    """Per-vector error at each quantization stage.

    Returns dict with: mse/max/mean at each stage, k_norm, cosine_sim.
    """
    kf = k.float()
    x0, x1 = kf[..., 0::2], kf[..., 1::2]
    y_exact = torch.sqrt(x0 * x0 + x1 * x1 + 1e-12)

    q1, q2, z, theta1, _ = _encode_core(kf, config=config)

    theta1_hat = config.theta1.dequantize(q1)
    k_after_theta1 = reconstruct_k_from_y_theta1(y_exact, theta1_hat)

    y_hat = decode_y_from_polar_tree(q2, z, config=config)
    k_after_theta2 = reconstruct_k_from_y_theta1(y_hat, theta1)

    k_after_full = decode_polar_k64(q1, q2, z, config=config)

    mse1, max1, mean1 = _vec_error(kf, k_after_theta1)
    mse2, max2, mean2 = _vec_error(kf, k_after_theta2)
    msef, maxf, meanf = _vec_error(kf, k_after_full)

    k_norm = kf.norm(dim=-1)
    k_hat_norm = k_after_full.norm(dim=-1).clamp_min(1e-12)
    cos_sim = (kf * k_after_full).sum(dim=-1) / (k_norm.clamp_min(1e-12) * k_hat_norm)

    return {
        'mse_after_theta1': mse1, 'mse_after_theta2': mse2, 'mse_after_full': msef,
        'max_after_theta1': max1, 'max_after_theta2': max2, 'max_after_full': maxf,
        'mean_after_theta1': mean1, 'mean_after_theta2': mean2, 'mean_after_full': meanf,
        'k_norm': k_norm, 'cosine_sim_full': cos_sim,
    }


# =====================================================================
#  PolarK64Batch — one step of quantized K  (B, L, H, 64)
# =====================================================================

@dataclass
class PolarK64Batch:
    q1: torch.Tensor   # (B, L, H, 32) uint8
    q2: torch.Tensor   # (B, L, H, 31) uint8
    z: torch.Tensor    # (B, L, H)   float16

    @classmethod
    def from_k(cls, k: torch.Tensor, config: PolarQuantConfig = DEFAULT_CONFIG) -> 'PolarK64Batch':
        """Encode (B, L, H, 64) K tensor -> PolarK64Batch."""
        q1, q2, z = encode_polar_k64(k.float(), config=config)
        return cls(q1=q1.to(torch.uint8), q2=q2.to(torch.uint8), z=z.to(torch.float16))

    def decode(self, config: PolarQuantConfig = DEFAULT_CONFIG, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Decode back to (B, L, H, 64) K_hat."""
        return decode_polar_k64(
            self.q1.to(torch.int64), self.q2.to(torch.int64), self.z.float(),
            config=config,
        ).to(dtype)

    def __len__(self) -> int:
        return self.q1.shape[1]

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        return {'q1': self.q1.cpu().numpy(), 'q2': self.q2.cpu().numpy(), 'z': self.z.cpu().numpy()}

    def save_npz(self, path: Union[str, Path], **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_numpy_dict()
        payload['_meta_json'] = np.array(json.dumps({**meta}))
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> Tuple['PolarK64Batch', Dict[str, Any]]:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data['_meta_json'])) if '_meta_json' in data else {}
        return cls(
            q1=torch.from_numpy(data['q1'].astype(np.uint8)),
            q2=torch.from_numpy(data['q2'].astype(np.uint8)),
            z=torch.from_numpy(data['z'].astype(np.float16)),
        ), meta


# =====================================================================
#  PolarKVCache — cross-step accumulated polar K cache (append-only)
# =====================================================================

class PolarKVCache:
    """Stores quantized K across autoregressive steps. V stays FP16 elsewhere."""

    def __init__(self, config: PolarQuantConfig = DEFAULT_CONFIG):
        self.config = config
        self.q1: Optional[torch.Tensor] = None
        self.q2: Optional[torch.Tensor] = None
        self.z: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return 0 if self.q1 is None else self.q1.shape[1]

    def append(self, batch: PolarK64Batch) -> None:
        if self.q1 is None:
            self.q1, self.q2, self.z = batch.q1, batch.q2, batch.z
        else:
            self.q1 = torch.cat([self.q1, batch.q1], dim=1)
            self.q2 = torch.cat([self.q2, batch.q2], dim=1)
            self.z = torch.cat([self.z, batch.z], dim=1)

    def as_batch(self) -> PolarK64Batch:
        assert self.q1 is not None, 'cache is empty'
        return PolarK64Batch(self.q1, self.q2, self.z)

    def decode_k(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Decode entire cached sequence -> K_hat (B, total_L, H, 64)."""
        return self.as_batch().decode(config=self.config, dtype=dtype)

    def clear(self) -> None:
        self.q1 = self.q2 = self.z = None

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        return self.as_batch().to_numpy_dict()

    def save_npz(self, path: Union[str, Path], **meta) -> None:
        self.as_batch().save_npz(path, **meta)


# =====================================================================
#  Roundtrip error utility
# =====================================================================

def roundtrip_error(k: torch.Tensor, config: PolarQuantConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    """Encode -> decode -> measure MSE/Max/Mean error."""
    q1, q2, z = encode_polar_k64(k.float(), config=config)
    k_hat = decode_polar_k64(q1, q2, z, config=config)
    err = (k.float() - k_hat).abs()
    return {'mse': float((err ** 2).mean()), 'max_abs': float(err.max()), 'mean_abs': float(err.mean())}


def encode_polar_k64_angles(k: torch.Tensor, config: PolarQuantConfig = DEFAULT_CONFIG) -> Dict[str, torch.Tensor]:
    """Encode + return theta before/after quant. theta2_hat is from roundtrip re-encode."""
    q1, q2, z, theta1, theta2 = _encode_core(k.float(), config=config)
    # re-encode the decoded K to get theta2 after full roundtrip
    k_hat = decode_polar_k64(q1, q2, z, config=config)
    _, _, _, _, theta2_roundtrip = _encode_core(k_hat, config=config)
    return {
        'q1': q1, 'q2': q2, 'z': z,
        'theta1': theta1, 'theta1_hat': config.theta1.dequantize(q1),
        'theta2': theta2, 'theta2_hat': theta2_roundtrip,
    }
