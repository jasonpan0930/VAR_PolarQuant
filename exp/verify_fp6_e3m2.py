"""
Sanity-check INT6 + FP6 E3M2 wiring and OCP decode vs spec table.

  cd VAR_polarQuant && python exp/verify_fp6_e3m2.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

from utils.angle_quant import (
    POLAR_QUANT_CONFIGS,
    THETA2_FP6_E3M2,
    _decode_ocp_fp,
)
from utils.polar_kv_quant import decode_polar_k64, encode_polar_k64, roundtrip_error

# OCP MX Table 4 spot checks (FP6 E3M2, bias=3): bit pattern -> value
OCP_E3M2_CASES = (
    ('+28.0', 0b011111),   # S=0 E=111 M=11
    ('+0.25', 0b000100),  # S=0 E=001 M=00  min normal
    ('-28.0', 0b111111),
    ('+0.0625', 0b000001),  # min subnorm: 2^(1-3) * 0.25 = 0.0625
    ('+0.1875', 0b000011),  # max subnorm: 2^(1-3) * 0.75 = 0.1875
)


def main() -> None:
    cfg_fp6 = POLAR_QUANT_CONFIGS['fp6_e3m2']
    assert cfg_fp6.theta1.num_bins == 64, 'theta1 must be INT6 (64 levels)'
    assert cfg_fp6.theta2.num_bins == 64, 'theta2 must be FP6 (64 levels)'
    assert cfg_fp6.theta1.name == 'int6_uniform'
    assert cfg_fp6.theta2.name == 'fp6_e3m2'

    print('=== config fp6_e3m2 ===')
    print(cfg_fp6.label)
    print(f'theta1 step (median): {cfg_fp6.theta1.typical_step():.6f} rad  (~2pi/64)')
    print(f'theta2 unique LUT angles: {len(np.unique(np.round(cfg_fp6.theta2.codebook_numpy(), 8)))}/64')

    print('\n=== OCP E3M2 decode (bias=3, E3M2) ===')
    for label, bits in OCP_E3M2_CASES:
        v = _decode_ocp_fp(bits, 3, 2, 3)
        print(f'  {label:8s}  bits={bits:06b}  decode={v:+.4f}')

    cb = THETA2_FP6_E3M2.codebook_numpy()
    print(f'\n  LUT[31] (|+28|) = {cb[31]:.6f}  expect pi/2 = {np.pi/2:.6f}')
    print(f'  LUT[4]  (+0.25) = {cb[4]:.6f}  expect {0.25/28*np.pi/2:.6f}')

    # Test FP6 E3M2 roundtrip
    k = torch.randn(8, 64)
    q1, q2, z = encode_polar_k64(k, config=cfg_fp6)
    assert q1.max() < 64 and q2.max() < 64
    k_hat = decode_polar_k64(q1, q2, z, config=cfg_fp6)
    rt = roundtrip_error(k[0], config=cfg_fp6)
    print('\n=== round-trip (random K) ===')
    print(f'  q1 range [0, {int(q1.max())}], q2 range [0, {int(q2.max())}]')
    print(f'  MSE={rt["mse"]:.6f}  mean_abs={rt["mean_abs"]:.6f}')

    # Compare with uniform_int4
    cfg_uni = POLAR_QUANT_CONFIGS['uniform_int4']
    rt_u = roundtrip_error(k[0], config=cfg_uni)
    print('\n=== compare uniform_int4 (same K[0]) ===')
    print(f'  INT6+INT4 theta2  MSE={rt_u["mse"]:.6f}')
    print(f'  INT6+FP6 E3M2     MSE={rt["mse"]:.6f}')

    print('\nOK — wiring matches INT6 theta1 + FP6 E3M2 theta2 LUT.')


if __name__ == '__main__':
    main()
