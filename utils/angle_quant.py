"""Polar angle quantizers (θ₁ / θ₂) — scheme definitions + name→config lookup.

All quantize/dequantize operations are methods on AngleQuantScheme.
Config is always passed explicitly — NO global mutable state.

Usage:
    from utils.angle_quant import POLAR_QUANT_CONFIGS, DEFAULT_CONFIG
    config = POLAR_QUANT_CONFIGS['uniform_int4']
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

PI = math.pi


# ═══════════════════════════════════════════════════════════════════
#  FP4 / FP6 硬體浮點數格式解碼（只保留 scheme 定義需要的部分）
# ═══════════════════════════════════════════════════════════════════

E2M1_FP4_VALUES: Tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _decode_ocp_fp(bits: int, exp_bits: int, mant_bits: int, bias: int) -> float:
    s = (bits >> (exp_bits + mant_bits)) & 1
    e = (bits >> mant_bits) & ((1 << exp_bits) - 1)
    m = bits & ((1 << mant_bits) - 1)
    sign = -1.0 if s else 1.0
    if e == 0:
        return sign * (2.0 ** (1 - bias)) * (m / (2 ** mant_bits))
    return sign * (2.0 ** (e - bias)) * (1.0 + m / (2 ** mant_bits))


def _fp6_lut(exp_bits: int, mant_bits: int, bias: int) -> Tuple[float, ...]:
    return tuple(_decode_ocp_fp(i, exp_bits, mant_bits, bias) for i in range(64))


def _linearly_map(values: Tuple[float, ...], lo: float, hi: float) -> Tuple[float, ...]:
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    if span <= 0:
        return tuple(0.5 * (lo + hi) for _ in values)
    return tuple(lo + (v - vmin) / span * (hi - lo) for v in values)


def _uniform_codebook(lo: float, hi: float, n: int) -> Tuple[float, ...]:
    w = (hi - lo) / n
    return tuple(lo + (i + 0.5) * w for i in range(n))


# ═══════════════════════════════════════════════════════════════════
#  AngleQuantScheme — 一個量化方案的完整封裝（codebook + quantize/dequantize）
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AngleQuantScheme:
    name: str
    label: str
    codebook: Tuple[float, ...]

    @property
    def num_bins(self) -> int:
        return len(self.codebook)

    def codebook_numpy(self) -> np.ndarray:
        return np.asarray(self.codebook, dtype=np.float64)

    def codebook_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(self.codebook, device=device, dtype=dtype)

    def quantize(self, theta: torch.Tensor) -> torch.Tensor:
        """θ → nearest codebook index (nearest-neighbor)."""
        cb = self.codebook_tensor(theta.device, theta.dtype)
        return (theta.unsqueeze(-1) - cb).abs().argmin(dim=-1).to(torch.int64)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        """Codebook index → reconstructed θ."""
        return self.codebook_tensor(q.device)[q.to(torch.int64).clamp(0, self.num_bins - 1)]

    def typical_step(self) -> float:
        cb = np.sort(self.codebook_numpy())
        gaps = np.diff(cb)
        gaps = gaps[gaps > 0]
        if gaps.size == 0:
            return (cb[-1] - cb[0]) / max(self.num_bins, 1)
        return float(np.median(gaps))


# ═══════════════════════════════════════════════════════════════════
#  θ₁ schemes（θ₁ ∈ [−π, π]）
# ═══════════════════════════════════════════════════════════════════

THETA1_INT6_UNIFORM = AngleQuantScheme(
    name='int6_uniform', label='uniform INT6',
    codebook=_uniform_codebook(-PI, PI, 64),
)


# ═══════════════════════════════════════════════════════════════════
#  θ₂ schemes（θ₂ ∈ [0, π/2]）— 5 種方案
# ═══════════════════════════════════════════════════════════════════

THETA2_UNIFORM_INT4 = AngleQuantScheme(
    name='uniform_int4', label='uniform INT4',
    codebook=_uniform_codebook(0.0, PI / 2, 16),
)

THETA2_E2M1_FP4 = AngleQuantScheme(
    name='e2m1_fp4', label='E2M1 FP4',
    codebook=_linearly_map(E2M1_FP4_VALUES, 0.0, PI / 2),
)

THETA2_FP6_E3M2 = AngleQuantScheme(
    name='fp6_e3m2', label='FP6 E3M2',
    codebook=_linearly_map(_fp6_lut(3, 2, 3), 0.0, PI / 2),
)

THETA2_FP6_E2M3 = AngleQuantScheme(
    name='fp6_e2m3', label='FP6 E2M3',
    codebook=_linearly_map(_fp6_lut(2, 3, 1), 0.0, PI / 2),
)

# k-means scheme 由 register_theta2_kmeans_codebook() 動態植入
THETA2_KMEANS_INT4: Optional[AngleQuantScheme] = None
THETA2_KMEANS_INT4_V: Optional[AngleQuantScheme] = None


# ═══════════════════════════════════════════════════════════════════
#  PolarQuantConfig — θ₁ + θ₂ 配對（一個完整的 polar K 量化方案）
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PolarQuantConfig:
    name: str
    label: str
    theta1: AngleQuantScheme
    theta2: AngleQuantScheme


# ═══════════════════════════════════════════════════════════════════
#  命名方案登錄表 — 純查詢，沒有全域可變狀態
# ═══════════════════════════════════════════════════════════════════

POLAR_QUANT_CONFIGS: Dict[str, PolarQuantConfig] = {
    'uniform_int4': PolarQuantConfig(
        'uniform_int4', 'INT6 θ₁ + uniform INT4 θ₂',
        THETA1_INT6_UNIFORM, THETA2_UNIFORM_INT4,
    ),
    'e2m1_fp4': PolarQuantConfig(
        'e2m1_fp4', 'INT6 θ₁ + E2M1 FP4 θ₂',
        THETA1_INT6_UNIFORM, THETA2_E2M1_FP4,
    ),
    'fp6_e3m2': PolarQuantConfig(
        'fp6_e3m2', 'INT6 θ₁ + FP6 E3M2 θ₂',
        THETA1_INT6_UNIFORM, THETA2_FP6_E3M2,
    ),
    'fp6_e2m3': PolarQuantConfig(
        'fp6_e2m3', 'INT6 θ₁ + FP6 E2M3 θ₂',
        THETA1_INT6_UNIFORM, THETA2_FP6_E2M3,
    ),
}

DEFAULT_CONFIG: PolarQuantConfig = POLAR_QUANT_CONFIGS['uniform_int4']


def resolve_config(config: PolarQuantConfig | str | None = None) -> PolarQuantConfig:
    """字串名稱 or Config 物件 → PolarQuantConfig；None → DEFAULT_CONFIG."""
    if config is None:
        return DEFAULT_CONFIG
    if isinstance(config, PolarQuantConfig):
        return config
    key = config.lower().strip()
    if key in POLAR_QUANT_CONFIGS:
        return POLAR_QUANT_CONFIGS[key]
    raise ValueError(f'Unknown config {config!r}; options: {list(POLAR_QUANT_CONFIGS)}')


# ═══════════════════════════════════════════════════════════════════
#  K-means codebook 註冊（跑完 exp_theta2_kmeans.py 後動態植入）
# ═══════════════════════════════════════════════════════════════════

def register_theta2_kmeans_codebook(
    centers: Sequence[float],
    config_name: str = 'int6_kmeans_int4',
    label: str = 'INT6 θ₁ + K-means INT4 θ₂ (MSE-weighted)',
) -> AngleQuantScheme:
    if len(centers) != 16:
        raise ValueError(f'K-means θ₂ needs 16 centers, got {len(centers)}')
    scheme = AngleQuantScheme(
        name='kmeans_int4', label='K-means INT4 (MSE-weighted)',
        codebook=tuple(float(c) for c in sorted(centers)),
    )
    global THETA2_KMEANS_INT4
    THETA2_KMEANS_INT4 = scheme
    POLAR_QUANT_CONFIGS[config_name] = PolarQuantConfig(
        config_name, label, THETA1_INT6_UNIFORM, scheme,
    )
    return scheme


def register_theta2_kmeans_codebook_v(
    centers: Sequence[float],
    config_name: str = 'int6_kmeans_int4_v',
    label: str = 'INT6 θ₁ + K-means INT4 θ₂ (V, MSE-weighted)',
) -> AngleQuantScheme:
    if len(centers) != 16:
        raise ValueError(f'K-means θ₂ needs 16 centers, got {len(centers)}')
    scheme = AngleQuantScheme(
        name='kmeans_int4_v', label='K-means INT4 (V, MSE-weighted)',
        codebook=tuple(float(c) for c in sorted(centers)),
    )
    global THETA2_KMEANS_INT4_V
    THETA2_KMEANS_INT4_V = scheme
    POLAR_QUANT_CONFIGS[config_name] = PolarQuantConfig(
        config_name, label, THETA1_INT6_UNIFORM, scheme,
    )
    return scheme


# ═══════════════════════════════════════════════════════════════════
#  numpy helper（用於 experiment scripts）
# ═══════════════════════════════════════════════════════════════════

def dequantize_theta2_numpy(q2: np.ndarray, scheme: AngleQuantScheme) -> np.ndarray:
    q2 = np.asarray(q2, dtype=np.int64)
    return scheme.codebook_numpy()[q2.clip(0, scheme.num_bins - 1)]
