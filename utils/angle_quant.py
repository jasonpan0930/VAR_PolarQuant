"""Polar angle quantizers (θ₁ / θ₂) and named full-pipeline configs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

PI = math.pi

# OCP MX FP4 E2M1 (bias=1), codes 0..15.
E2M1_FP4_VALUES: Tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _decode_ocp_fp(bits: int, exp_bits: int, mant_bits: int, bias: int) -> float:
    """OCP MX FP6/FP4 subnormal + normal decode (Table 4/5, no Inf/NaN)."""
    s = (bits >> (exp_bits + mant_bits)) & 1
    e_mask = (1 << exp_bits) - 1
    m_mask = (1 << mant_bits) - 1
    e = (bits >> mant_bits) & e_mask
    m = bits & m_mask
    sign = -1.0 if s else 1.0
    if e == 0:
        # Subnormal: 2^(1-bias) × 0.mantissa  (for bias=1 this is 2^0 × m/2^m)
        return sign * (2.0 ** (1 - bias)) * (m / (2 ** mant_bits))
    return sign * (2.0 ** (e - bias)) * (1.0 + m / (2 ** mant_bits))


def _fp6_lut(exp_bits: int, mant_bits: int, bias: int) -> Tuple[float, ...]:
    return tuple(_decode_ocp_fp(i, exp_bits, mant_bits, bias) for i in range(64))


def _linear_map_to_interval(values: Tuple[float, ...], lo: float, hi: float) -> Tuple[float, ...]:
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    if span <= 0:
        mid = 0.5 * (lo + hi)
        return tuple(mid for _ in values)
    return tuple(lo + (v - vmin) / span * (hi - lo) for v in values)


def _uniform_codebook(lo: float, hi: float, n: int) -> Tuple[float, ...]:
    w = (hi - lo) / n
    return tuple(lo + (i + 0.5) * w for i in range(n))


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

    def typical_step(self) -> float:
        cb = np.sort(self.codebook_numpy())
        gaps = np.diff(cb)
        gaps = gaps[gaps > 0]
        if gaps.size == 0:
            return (cb[-1] - cb[0]) / max(self.num_bins, 1)
        return float(np.median(gaps))

    def quantize(self, theta: torch.Tensor) -> torch.Tensor:
        cb = self.codebook_tensor(theta.device, theta.dtype)
        return (theta.unsqueeze(-1) - cb).abs().argmin(dim=-1).to(torch.int64)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return self.codebook_tensor(q.device)[q.to(torch.int64).clamp(0, self.num_bins - 1)]


# --- θ₁ schemes ---
THETA1_INT6_UNIFORM = AngleQuantScheme(
    name='int6_uniform',
    label='uniform INT6',
    codebook=_uniform_codebook(-PI, PI, 64),
)
THETA1_INT4_UNIFORM = AngleQuantScheme(
    name='int4_uniform',
    label='uniform INT4',
    codebook=_uniform_codebook(-PI, PI, 16),
)

# --- θ₂ schemes (4-bit) ---
THETA2_UNIFORM_INT4 = AngleQuantScheme(
    name='uniform_int4',
    label='uniform INT4',
    codebook=_uniform_codebook(0.0, PI / 2, 16),
)
THETA2_E2M1_FP4 = AngleQuantScheme(
    name='e2m1_fp4',
    label='E2M1 FP4',
    codebook=_linear_map_to_interval(E2M1_FP4_VALUES, 0.0, PI / 2),
)

def _fp6_theta2_codebook(exp_bits: int, mant_bits: int, bias: int) -> Tuple[float, ...]:
    """
    θ₂ ∈ [0, π/2]: FP6 value range [−Vmax, +Vmax] linearly mapped to [0, π/2].
    Zero → π/4; ±0 both map to π/4 (1 duplicate out of 64, same as FP4).
    """
    values = _fp6_lut(exp_bits, mant_bits, bias)
    return _linear_map_to_interval(values, 0.0, PI / 2)


THETA2_FP6_E3M2 = AngleQuantScheme(
    name='fp6_e3m2',
    label='FP6 E3M2',
    codebook=_fp6_theta2_codebook(exp_bits=3, mant_bits=2, bias=3),
)
THETA2_FP6_E2M3 = AngleQuantScheme(
    name='fp6_e2m3',
    label='FP6 E2M3',
    codebook=_fp6_theta2_codebook(exp_bits=2, mant_bits=3, bias=1),
)


@dataclass(frozen=True)
class PolarQuantConfig:
    """θ₁ + θ₂ quant pairing for one polar-K experiment."""
    name: str
    label: str
    theta1: AngleQuantScheme
    theta2: AngleQuantScheme


# Filled by register_theta2_kmeans_codebook() after MSE-weighted K-means fit.
THETA2_KMEANS_INT4: Optional[AngleQuantScheme] = None


def register_theta2_kmeans_codebook(
    centers: Sequence[float],
    config_name: str = 'int6_kmeans_int4',
    label: str = 'INT6 θ₁ + K-means INT4 θ₂ (MSE-weighted)',
) -> AngleQuantScheme:
    """Register / replace θ₂ codebook from K-means centroids (16 levels, sorted)."""
    global THETA2_KMEANS_INT4
    if len(centers) != 16:
        raise ValueError(f'K-means θ₂ codebook must have 16 entries, got {len(centers)}')
    scheme = AngleQuantScheme(
        name='kmeans_int4',
        label='K-means INT4 (MSE-weighted)',
        codebook=tuple(float(c) for c in sorted(centers)),
    )
    THETA2_KMEANS_INT4 = scheme
    POLAR_QUANT_CONFIGS[config_name] = PolarQuantConfig(
        config_name, label, THETA1_INT6_UNIFORM, scheme,
    )
    return scheme


POLAR_QUANT_CONFIGS: Dict[str, PolarQuantConfig] = {
  # Legacy names (θ₁ stays INT6)
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

_active: PolarQuantConfig = POLAR_QUANT_CONFIGS['uniform_int4']


def get_polar_quant_config() -> PolarQuantConfig:
    return _active


def set_polar_quant_config(name: str) -> PolarQuantConfig:
    global _active
    key = name.lower().strip()
    if key not in POLAR_QUANT_CONFIGS:
        opts = ', '.join(sorted(POLAR_QUANT_CONFIGS))
        raise ValueError(f'unknown polar quant config {name!r}; choose one of: {opts}')
    _active = POLAR_QUANT_CONFIGS[key]
    return _active


def get_theta1_scheme() -> AngleQuantScheme:
    return _active.theta1


def get_theta2_scheme() -> AngleQuantScheme:
    return _active.theta2


# Backward-compatible aliases (θ₂-only naming from older scripts)
def get_theta2_quant_scheme() -> AngleQuantScheme:
    return get_theta2_scheme()


def set_theta2_quant_scheme(name: str) -> AngleQuantScheme:
    set_polar_quant_config(name)
    return get_theta2_scheme()


def dequantize_theta2_numpy(q2: np.ndarray, scheme: AngleQuantScheme | None = None) -> np.ndarray:
    scheme = scheme or get_theta2_scheme()
    q2 = np.asarray(q2, dtype=np.int64)
    return scheme.codebook_numpy()[q2.clip(0, scheme.num_bins - 1)]


# Legacy constants
Q2_BINS = 16
