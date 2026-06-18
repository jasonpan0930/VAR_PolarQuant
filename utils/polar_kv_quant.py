"""Hierarchical polar quantization for 64-dim K vectors (configurable θ₁/θ₂ + FP16 z)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from utils.angle_quant import (
    get_polar_quant_config,
    get_theta1_scheme,
    get_theta2_scheme,
    set_polar_quant_config,
)

HEAD_DIM = 64
NUM_Q1 = 32
NUM_Q2 = 31
PI = math.pi


def quantize_theta1(theta: torch.Tensor) -> torch.Tensor:
    return get_theta1_scheme().quantize(theta)


def dequantize_theta1(q1: torch.Tensor) -> torch.Tensor:
    return get_theta1_scheme().dequantize(q1)


def quantize_theta2(theta: torch.Tensor) -> torch.Tensor:
    return get_theta2_scheme().quantize(theta)


def dequantize_theta2(q2: torch.Tensor) -> torch.Tensor:
    return get_theta2_scheme().dequantize(q2)


def _encode_polar_k64_core(k: torch.Tensor):
    assert k.shape[-1] == HEAD_DIM, f'expected last dim {HEAD_DIM}, got {k.shape[-1]}'
    x0 = k[..., 0::2]
    x1 = k[..., 1::2]
    y = torch.sqrt(x0 * x0 + x1 * x1 + 1e-12)
    theta1 = torch.atan2(x1, x0)
    q1 = quantize_theta1(theta1)
    theta1_hat = dequantize_theta1(q1)

    nodes = y
    q2_parts, theta2_parts, theta2_hat_parts = [], [], []
    for _ in range(5):
        a = nodes[..., 0::2]
        b = nodes[..., 1::2]
        z = torch.sqrt(a * a + b * b + 1e-12)
        theta2 = torch.atan2(b, a)
        q2 = quantize_theta2(theta2)
        q2_parts.append(q2)
        theta2_parts.append(theta2)
        theta2_hat_parts.append(dequantize_theta2(q2))
        nodes = z
    q2 = torch.cat(q2_parts, dim=-1)
    theta2 = torch.cat(theta2_parts, dim=-1)
    theta2_hat = torch.cat(theta2_hat_parts, dim=-1)
    z_root = nodes[..., 0]
    return q1, q2, z_root, theta1, theta1_hat, theta2, theta2_hat


def encode_polar_k64(k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    k: (..., 64) fp16/fp32
    returns q1 (..., 32) int64, q2 (..., 31) int64, z (...,) float32
    """
    q1, q2, z_root, _, _, _, _ = _encode_polar_k64_core(k)
    return q1, q2, z_root


def encode_polar_k64_angles(k: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return θ before quant and θ₂ re-extracted after full encode→decode round-trip."""
    q1, q2, z_root, theta1, theta1_hat, theta2, _ = _encode_polar_k64_core(k)
    k_hat = decode_polar_k64(q1, q2, z_root)
    _, _, _, _, _, theta2_roundtrip, _ = _encode_polar_k64_core(k_hat)
    return {
        'q1': q1,
        'q2': q2,
        'z': z_root,
        'theta1': theta1,
        'theta1_hat': theta1_hat,
        'theta2': theta2,
        'theta2_hat': theta2_roundtrip,
    }

def decode_y_from_polar_tree(q2: torch.Tensor, z_root: torch.Tensor) -> torch.Tensor:
    """Decode merge tree to 32 lengths y (..., 32); same q2 layout as decode_polar_k64."""
    theta2_all = dequantize_theta2(q2)
    nodes = z_root.unsqueeze(-1)
    offset = NUM_Q2
    for layer in range(5):
        n = 2 ** layer
        offset -= n
        theta = theta2_all[..., offset:offset + n]
        a = nodes * torch.cos(theta)
        b = nodes * torch.sin(theta)
        nodes = torch.stack([a, b], dim=-1).reshape(*nodes.shape[:-1], n * 2)
    return nodes


def reconstruct_k_from_y_theta1(y: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
    """y (..., 32), theta1 (..., 32) -> k (..., 64)."""
    x0 = y * torch.cos(theta1)
    x1 = y * torch.sin(theta1)
    return torch.stack([x0, x1], dim=-1).reshape(*y.shape[:-1], HEAD_DIM)


def _vec_error_stats(k_orig: torch.Tensor, k_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    err = k_orig.float() - k_hat.float()
    mse = (err ** 2).mean(dim=-1)
    max_abs = err.abs().max(dim=-1).values
    mean_abs = err.abs().mean(dim=-1)
    return mse, max_abs, mean_abs


def polar_k64_error_breakdown(k: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Per 64-dim K vector error at each hierarchical stage (global round-trip view).

    Stages (cumulative pipeline):
      fp           — original K (baseline)
      after_theta1 — quant θ₁ only; exact lengths y from FP
      after_theta2 — quant θ₂ tree + z; exact θ₁ from FP
      after_full   — quant θ₁ + θ₂ + z (what inference uses)
    """
    q1, q2, z, theta1, theta1_hat, _, _ = _encode_polar_k64_core(k.float())
    x0, x1 = k[..., 0::2], k[..., 1::2]
    y_exact = torch.sqrt(x0 * x0 + x1 * x1 + 1e-12)

    k_after_theta1 = reconstruct_k_from_y_theta1(y_exact, theta1_hat)
    y_hat = decode_y_from_polar_tree(q2, z)
    k_after_theta2 = reconstruct_k_from_y_theta1(y_hat, theta1)
    k_after_full = decode_polar_k64(q1, q2, z)

    mse1, max1, mean1 = _vec_error_stats(k, k_after_theta1)
    mse2, max2, mean2 = _vec_error_stats(k, k_after_theta2)
    msef, maxf, meanf = _vec_error_stats(k, k_after_full)
    dim_abs = (k.float() - k_after_full).abs().mean(dim=tuple(range(k.dim() - 1)))

    k_f = k.float()
    k_norm = k_f.norm(dim=-1)
    k_hat_f = k_after_full.float()
    k_hat_norm = k_hat_f.norm(dim=-1).clamp_min(1e-12)
    cos_sim = (k_f * k_hat_f).sum(dim=-1) / (k_norm.clamp_min(1e-12) * k_hat_norm)

    return {
        'mse_after_theta1': mse1,
        'mse_after_theta2': mse2,
        'mse_after_full': msef,
        'max_after_theta1': max1,
        'max_after_theta2': max2,
        'max_after_full': maxf,
        'mean_after_theta1': mean1,
        'mean_after_theta2': mean2,
        'mean_after_full': meanf,
        'dim_mean_abs_full': dim_abs,
        'k_norm': k_norm,
        'cosine_sim_full': cos_sim,
    }


def decode_polar_k64(
    q1: torch.Tensor, q2: torch.Tensor, z_root: torch.Tensor,
) -> torch.Tensor:
    """
    q1: (..., 32), q2: (..., 31), z_root: (...)
    returns k_hat (..., 64)
    """
    # q2 layout matches encode: [16, 8, 4, 2, 1] thetas per merge layer (32→1 tree).
    # Decode walks root→leaves, consuming q2 from the last segment backward.
    theta2_all = dequantize_theta2(q2)
    nodes = z_root.unsqueeze(-1)
    offset = NUM_Q2
    for layer in range(5):
        n = 2 ** layer
        offset -= n
        theta = theta2_all[..., offset:offset + n]
        a = nodes * torch.cos(theta)
        b = nodes * torch.sin(theta)
        nodes = torch.stack([a, b], dim=-1).reshape(*nodes.shape[:-1], n * 2)

    y = nodes
    theta1 = dequantize_theta1(q1)
    x0 = y * torch.cos(theta1)
    x1 = y * torch.sin(theta1)
    k_hat = torch.stack([x0, x1], dim=-1).reshape(*q1.shape[:-1], HEAD_DIM)
    return k_hat


@dataclass
class PolarK64Batch:
    """Quantized K for shape (B, L, H, 64)."""
    q1: torch.Tensor   # (B, L, H, 32) uint8
    q2: torch.Tensor   # (B, L, H, 31) uint8
    z: torch.Tensor    # (B, L, H) float16

    @classmethod
    def from_k(cls, k: torch.Tensor) -> 'PolarK64Batch':
        """k: (B, L, H, 64)"""
        q1, q2, z = encode_polar_k64(k.float())
        return cls(
            q1=q1.to(torch.uint8),
            q2=q2.to(torch.uint8),
            z=z.to(torch.float16),
        )

    def decode(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        k = decode_polar_k64(
            self.q1.to(torch.int64),
            self.q2.to(torch.int64),
            self.z.float(),
        )
        return k.to(dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.q1.shape[:-1]

    def __len__(self) -> int:
        return self.q1.shape[1]

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        return {
            'q1': self.q1.cpu().numpy(),
            'q2': self.q2.cpu().numpy(),
            'z': self.z.cpu().numpy(),
        }

    def save_npz(self, path: Union[str, Path], **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_numpy_dict()
        meta = dict(meta)
        meta.setdefault('polar_quant', get_polar_quant_config().name)
        meta.setdefault('theta2_quant', get_polar_quant_config().name)
        payload['_meta_json'] = np.array(json.dumps(meta))
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> Tuple['PolarK64Batch', Dict[str, Any]]:
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        meta = {}
        if '_meta_json' in data:
            meta = json.loads(str(data['_meta_json']))
        return cls(
            q1=torch.from_numpy(data['q1'].astype(np.uint8)),
            q2=torch.from_numpy(data['q2'].astype(np.uint8)),
            z=torch.from_numpy(data['z'].astype(np.float16)),
        ), meta


class PolarKVCache:
    """Append-only polar K cache; V stays fp16 in SelfAttention."""

    def __init__(self):
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
        assert self.q1 is not None
        return PolarK64Batch(self.q1, self.q2, self.z)

    def decode_k(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return self.as_batch().decode(dtype=dtype)

    def clear(self) -> None:
        self.q1 = self.q2 = self.z = None

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        return self.as_batch().to_numpy_dict()

    def save_npz(self, path: Union[str, Path], **meta) -> None:
        self.as_batch().save_npz(path, **meta)


def roundtrip_error(k: torch.Tensor) -> Dict[str, float]:
    q1, q2, z = encode_polar_k64(k.float())
    k_hat = decode_polar_k64(q1, q2, z)
    err = (k.float() - k_hat).abs()
    return {
        'mse': float((err ** 2).mean()),
        'max_abs': float(err.max()),
        'mean_abs': float(err.mean()),
    }
