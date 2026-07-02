# Polar Quant Core Scripts

This directory now keeps only scripts that are useful for polar KV-cache
quantization, codebook preparation, sampling, and low-level checks.

## Kept Scripts

| Script | Purpose |
| --- | --- |
| `exp_theta2_kmeans.py` | Collect theta2 samples and fit K/V codebooks. |
| `exp_fid_sample.py` | Generate ImageNet samples with baseline or polar KV cache. |
| `exp_polar_kv_infer.py` | Run one inference and dump quantized KV cache snapshots. |
| `verify_fp6_e3m2.py` | Quick FP6 E3M2 / polar roundtrip checks. |
| `audit_fp6_e3m2.py` | More detailed FP6/polar audit output. |
| `sbatch_fid_sample.sh` | TWCC SLURM wrapper for sample generation. |
| `sbatch_theta2_kmeans.sh` | TWCC SLURM wrapper for codebook fitting. |

Removed analysis scripts included cross-block propagation, attention drift,
f_hat sanity checks, QKT heatmaps, fc2 histograms, and plot-only angle
distribution comparisons. They are still available in the backup copy:

```text
../VAR_polarQuant_backup/
```

## Codebook Paths

Reusable codebooks are version-controlled under:

```text
configs/codebooks/
```

Generated outputs should go to ignored directories such as:

```text
artifacts/
fid_samples/
logs/
```

## Common Commands

Fit per-level K/V codebooks:

```bash
python exp/exp_theta2_kmeans.py \
  --model-depth 30 \
  --per-level \
  --per-level-k 16 16 16 4 4 \
  --refit
```

Generate a small polar-quant sample check:

```bash
python exp/exp_fid_sample.py \
  --model-depth 30 \
  --polar-quant int6_kmeans_int4 \
  --theta2-levels 1,1,3,3,3 \
  --out-dir fid_samples/test_fpga_core \
  --class-start 0 --class-end 0 \
  --samples-per-class 2
```

Dump quantized KV cache tensors:

```bash
python exp/exp_polar_kv_infer.py
```

Run quick numeric checks:

```bash
python exp/verify_fp6_e3m2.py
python exp/audit_fp6_e3m2.py
```
