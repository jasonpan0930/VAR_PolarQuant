"""CORDIC vectoring mode: replace atan2 + sqrt with shift-add iterations."""

import math
from typing import Tuple

import torch


# ─── Precomputed tables ────────────────────────────────────────────────

def _build_atan_table(max_iters: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """atan(2^(-i)) for i = 0..max_iters-1, shape (max_iters,)."""
    return torch.tensor(
        [math.atan(2.0 ** -i) for i in range(max_iters)],
        device=device, dtype=dtype,
    )


def _build_k_table(max_iters: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """CORDIC gain factor K[n] = prod_{i=0}^{n-1} 1/sqrt(1+2^(-2i)).

    Returns shape (max_iters + 1,). K[0]=1.0, K[n] for n>=1.
    """
    vals = [1.0]
    prod = 1.0
    for i in range(max_iters):
        prod *= 1.0 / math.sqrt(1.0 + 2.0 ** (-2 * i))
        vals.append(prod)
    return torch.tensor(vals, device=device, dtype=dtype)


# ─── Core iteration ────────────────────────────────────────────────────

def cordic_vectoring(
    x: torch.Tensor,
    y: torch.Tensor,
    num_iters: int,
    atan_table: torch.Tensor = None,
    k_table: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CORDIC vectoring mode: (x, y) → (r, theta).

    Standard CORDIC converges only for vectors in Q1/Q4 (x ≥ 0).
    For Q2/Q3 (x < 0) the vector is pre-rotated by π / -π and the
    result angle is adjusted after convergence.

    Returns:
      r     = K_n * sqrt(x₀² + y₀²)
      theta = atan2(y₀, x₀)  in (-π, π]

    Args:
      x: arbitrary shape (float, any sign)
      y: same shape as x
      num_iters: number of CORDIC iterations
      atan_table: precomputed atan(2^(-i)) shape (num_iters,)
      k_table: precomputed K[n] shape (>= num_iters+1,)
    """
    if num_iters < 1:
        raise ValueError(f"num_iters must be >= 1, got {num_iters}")

    if atan_table is None or atan_table.shape[0] < num_iters:
        atan_table = _build_atan_table(num_iters, device=x.device, dtype=x.dtype)

    if k_table is None or k_table.shape[0] <= num_iters:
        k_table = _build_k_table(num_iters, device=x.device, dtype=x.dtype)

    # ── Quadrant correction: map x < 0 → x ≥ 0 via π-rotation ──
    neg_x = x < 0
    pos_y = (y >= 0) & neg_x   # Q2: x<0, y>=0 → rotate +π
    neg_y = (y < 0) & neg_x    # Q3: x<0, y<0  → rotate -π

    x_cordic = x.clone()
    y_cordic = y.clone()
    angle_offset = torch.zeros_like(x)

    # Q2: (x, y) → (-x, -y) puts vector in Q4; actual = z + π
    x_cordic = torch.where(pos_y, -x_cordic, x_cordic)
    y_cordic = torch.where(pos_y, -y_cordic, y_cordic)
    angle_offset = torch.where(pos_y, torch.full_like(angle_offset, math.pi), angle_offset)

    # Q3: (x, y) → (-x, -y) puts vector in Q1; actual = z - π
    x_cordic = torch.where(neg_y, -x_cordic, x_cordic)
    y_cordic = torch.where(neg_y, -y_cordic, y_cordic)
    angle_offset = torch.where(neg_y, torch.full_like(angle_offset, -math.pi), angle_offset)

    # ── CORDIC loop on quadrant-corrected vector (x_cordic ≥ 0) ──
    x_i, y_i = x_cordic, y_cordic
    z_i = torch.zeros_like(x_cordic)

    for i in range(num_iters):
        # d = -sign(y_i): -1 when y >= 0 (clockwise), +1 when y < 0 (ccw)
        d = torch.where(y_i >= 0, -torch.ones_like(y_i), torch.ones_like(y_i))
        shift = 2.0 ** -i
        x_next = x_i - d * y_i * shift
        y_next = y_i + d * x_i * shift
        z_next = z_i - d * atan_table[i]
        x_i, y_i, z_i = x_next, y_next, z_next

    # Scale by CORDIC gain to recover true magnitude
    r = x_i * k_table[num_iters]
    theta = z_i + angle_offset

    return r, theta
