# VAR_polarQuant: FPGA-Oriented Polar KV Quantization

This branch keeps the core pieces needed to move VAR KV-cache quantization
toward FPGA implementation. Historical analysis scripts, plots, logs, and
diagnostic notebooks were removed from the tracked repo. The full pre-cleanup
workspace is still available in `../VAR_polarQuant_backup/`.

## Goal

Compress the VAR autoregressive attention KV-cache with a hardware-friendly
polar representation:

- encode each 64-dim K/V head vector into quantized polar angles plus one
  magnitude;
- use CORDIC vectoring instead of `atan2 + sqrt` in the encode path;
- keep decode simple enough to map to LUT/multiply datapaths;
- preserve enough software hooks to validate quality before FPGA integration.

## Core Representation

Each 64-dim vector is represented as:

| Field | Count | Default precision | Range |
| --- | ---: | --- | --- |
| `theta1` | 32 | INT6 uniform | `[-pi, pi]` |
| `theta2` | 31 | INT4 / FP4 / FP6 / K-means | `[0, pi/2]` |
| `z` | 1 | FP16 | magnitude |

The polar tree is:

```text
64 values
  -> 32 pairwise polar rotations: theta1 + 32 lengths
  -> 5-level merge tree: theta2 + root magnitude z
```

Current storage target for the default INT6/INT4 path:

```text
32 * 6 + 31 * 4 + 16 = 332 bits per vector
```

## Hardware Path

The software encode path uses `utils/cordic.py`:

```text
(x, y) -> cordic_vectoring(x, y, num_iters) -> (r, theta)
```

Important details:

- quadrant correction handles full `atan2(y, x)` range;
- gain compensation recovers the magnitude;
- `PolarQuantConfig.cordic_iters` selects the iteration count;
- `cordic_iters < 0` is the exact `sqrt + atan2` golden reference.

For FPGA work, the next useful steps are:

1. Freeze the exact bit layout for `q1`, `q2`, and `z`.
2. Decide whether `z` stays FP16 or moves to fixed-point/block-float.
3. Replace PyTorch decode trigonometry with explicit LUT tables.
4. Export small deterministic software vectors for RTL testbenches.
5. Build bit-packing/unpacking utilities matching the FPGA memory format.

## Kept Core Files

| Path | Purpose |
| --- | --- |
| `utils/polar_kv_quant.py` | encode/decode, polar cache container |
| `utils/angle_quant.py` | theta codebooks and quantization configs |
| `utils/cordic.py` | CORDIC vectoring-mode encode primitive |
| `utils/theta2_kmeans.py` | K-means codebook I/O and fitting |
| `utils/polar_angle_viz.py` | collection helpers used by codebook fitting |
| `utils/polar_kv_store.py` | optional quantized KV dump writer |
| `models/basic_var.py` | attention KV-cache integration |
| `models/var.py` | `set_polar_quant(...)` API |
| `exp/exp_theta2_kmeans.py` | fit/re-fit theta2 codebooks |
| `exp/exp_fid_sample.py` | generate samples for quality checks |
| `exp/exp_polar_kv_infer.py` | dump quantized KV cache snapshots |
| `exp/verify_fp6_e3m2.py` | small quantization verification |
| `exp/audit_fp6_e3m2.py` | FP6/polar audit helper |

## Codebooks

Version-controlled codebooks live in:

```text
configs/codebooks/
```

Generated plots, dumps, logs, and FID samples are intentionally ignored by git.

## Minimal Usage

Fit or refresh codebooks:

```bash
python exp/exp_theta2_kmeans.py --model-depth 30 --per-level --refit
```

Run a small sample check:

```bash
python exp/exp_fid_sample.py \
  --model-depth 30 \
  --polar-quant int6_kmeans_int4 \
  --theta2-levels 1,1,3,3,3 \
  --out-dir fid_samples/test_fpga_core \
  --class-start 0 --class-end 0 --samples-per-class 2
```

Dump quantized KV cache snapshots:

```bash
python exp/exp_polar_kv_infer.py
```

Run lightweight verification:

```bash
python exp/verify_fp6_e3m2.py
python exp/audit_fp6_e3m2.py
```
