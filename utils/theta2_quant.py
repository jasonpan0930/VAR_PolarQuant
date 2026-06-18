"""Backward-compatible re-exports; use utils.angle_quant for new code."""
from utils.angle_quant import (  # noqa: F401
    Q2_BINS,
    dequantize_theta2_numpy,
    get_theta2_quant_scheme,
    set_polar_quant_config,
    set_theta2_quant_scheme,
)
