"""
Complete audit: INT6 θ₁ + FP6 E3M2 θ₂  (OCP MX Table 4)
  cd VAR_polarQuant && python exp/audit_fp6_e3m2.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
from math import pi

from utils.angle_quant import (
    _decode_ocp_fp, THETA2_FP6_E3M2, THETA1_INT6_UNIFORM,
    set_polar_quant_config, POLAR_QUANT_CONFIGS,
)
from utils.polar_kv_quant import encode_polar_k64, decode_polar_k64, roundtrip_error

# ─── 1. OCP table spot checks (FP6 E3M2 bias=3) ───
E3M2_CASES = [
    # (label, bits, expected from OCP Table 4)
    ('+0.0',      0b000000, 0.0),
    ('+0.0625',   0b000001, 0.0625),   # min subnorm
    ('+0.125',    0b000010, 0.125),
    ('+0.1875',   0b000011, 0.1875),   # max subnorm
    ('+0.25',     0b000100, 0.25),     # min normal E=001 M=00
    ('-0.25',     0b100100, -0.25),
    ('+0.3125',   0b000101, 0.3125),
    ('+0.375',    0b000110, 0.375),
    ('+0.4375',   0b000111, 0.4375),
    ('+1.0',      0b001100, 1.0),      # E=011 M=00
    ('+1.75',     0b001111, 1.75),
    ('+2.0',      0b010000, 2.0),
    ('+7.0',      0b010111, 7.0),
    ('+8.0',      0b011000, 8.0),
    ('+14.0',     0b011011, 14.0),
    ('+16.0',     0b011100, 16.0),
    ('+28.0',     0b011111, 28.0),     # max normal
    ('-28.0',     0b111111, -28.0),
]

print('=== 1. OCP E3M2 decode (bias=3) ===')
all_ok = True
for label, bits, expected in E3M2_CASES:
    got = _decode_ocp_fp(bits, exp_bits=3, mant_bits=2, bias=3)
    ok = abs(got - expected) < 1e-12
    if not ok:
        print(f'  FAIL {label:10s} bits=0b{bits:06b} got={got:+.6f} expected={expected:+.6f}')
        all_ok = False
if all_ok:
    print('  ALL OK')
else:
    print('  *** DECODE FAILURE ***')
    raise SystemExit(1)

# ─── 2. Codebook statistics (64 entries) ───
cb = THETA2_FP6_E3M2.codebook_numpy()
cb_rounded = np.round(cb, 12)
unique_angles = len(np.unique(cb_rounded))
print(f'\n=== 2. θ₂ E3M2 codebook ({len(cb)} entries) ===')
print(f'  unique angles: {unique_angles}/64')
print(f'  range: [{min(cb):.6f}, {max(cb):.6f}]  (expected [0, {pi/2:.4f}])')
print(f'  code  0 (+0):       {cb[0]:.8f}')
print(f'  code  1 (+0.0625):  {cb[1]:.8f}')
print(f'  code  4 (+0.25):    {cb[4]:.8f}')
print(f'  code  7 (+0.4375):  {cb[7]:.8f}')
print(f'  code 12 (+1.0):     {cb[12]:.8f}')
print(f'  code 15 (+1.75):    {cb[15]:.8f}')
print(f'  code 31 (+28.0):    {cb[31]:.8f}  expect {pi/2:.8f}')
print(f'  code 32 (-0):       {cb[32]:.8f}')
print(f'  code 33 (-0.0625):  {cb[33]:.8f}')
print(f'  code 63 (-28.0):    {cb[63]:.8f}')

if unique_angles != 32:
    print(f'  WARNING: expected 32 unique angles (sign folding), got {unique_angles}')

# ─── 3. Config wiring ───
cfg = POLAR_QUANT_CONFIGS['fp6_e3m2']
print(f'\n=== 3. Config {cfg.name} ===')
print(f'  label: {cfg.label}')
print(f'  θ₁: {cfg.theta1.name} ({cfg.theta1.label}) bins={cfg.theta1.num_bins}')
print(f'  θ₂: {cfg.theta2.name} ({cfg.theta2.label}) bins={cfg.theta2.num_bins}')
assert cfg.theta1.num_bins == 64, 'θ₁ must be 64-level INT6'
assert cfg.theta2.num_bins == 64, 'θ₂ must be 64-level FP6'

# ─── 4. Quantize / dequantize sanity for θ₂ ───
set_polar_quant_config('fp6_e3m2')
from utils.polar_kv_quant import quantize_theta2, dequantize_theta2
theta_test = torch.linspace(0, pi/2, 1001)
q2 = quantize_theta2(theta_test)
dq2 = dequantize_theta2(q2)
err_max = float((theta_test - dq2).abs().max())
print(f'\n=== 4. θ₂ roundtrip on linspace [0,π/2] ===')
print(f'  max recon error: {err_max:.6f} rad')
print(f'  q2 used unique codes: {len(torch.unique(q2))} / 64')

# ─── 5. Full polar roundtrip ───
k = torch.randn(16, 64)
q1, q2, z = encode_polar_k64(k)
k_hat = decode_polar_k64(q1, q2, z)
rt = roundtrip_error(k[0])
print(f'\n=== 5. Full polar roundtrip (random K) ===')
print(f'  q1 range: [0, {int(q1.max())}]  q2 range: [0, {int(q2.max())}]')
print(f'  K MSE: {rt["mse"]:.8f}  max|err|: {rt["max_abs"]:.6f}  mean|err|: {rt["mean_abs"]:.6f}')

# Compare with uniform_int4
set_polar_quant_config('uniform_int4')
rt4 = roundtrip_error(k[0])
print(f'\n=== 6. Compare uniform_int4 (same K[0]) ===')
print(f'  INT6+INT4  MSE: {rt4["mse"]:.8f}')
print(f'  INT6+E3M2  MSE: {rt["mse"]:.8f}')
print(f'  ratio E3M2 / INT4: {rt["mse"] / rt4["mse"]:.2f}x')

# ─── 7. Stage breakdown ───
set_polar_quant_config('fp6_e3m2')
from utils.polar_kv_quant import polar_k64_error_breakdown
bd = polar_k64_error_breakdown(k[:4])
for stage in ['after_theta1', 'after_theta2', 'after_full']:
    m = bd[f'mse_{stage}'].mean().item()
    print(f'  {stage:20s} mean MSE = {m:.3e}')

print('\n=== SUMMARY ===')
print(f'  OCP decode: {"OK" if all_ok else "FAIL"}')
print(f'  Config wiring: INT6 θ₁ + E2M3 θ₂ ✓')
print(f'  Unique angles: {unique_angles} (expected ≤32 for sign-folded FP6)')
print(f'  E3M2 full MSE / INT4 full MSE ≈ {rt["mse"]/rt4["mse"]:.2f}x')
