# Polar Quant Commands

Run commands from the repo root:

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant
```

Default TWCC Python:

```text
/home/jasonpan0930/.conda/envs/var_env/bin/python
```

## Codebook Fitting

Single depth-matched codebook:

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_theta2_kmeans.py \
  --model-depth 30 --refit
```

Per-level K/V codebooks:

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_theta2_kmeans.py \
  --model-depth 30 \
  --per-level \
  --per-level-k 16 16 16 4 4 \
  --refit
```

SLURM:

```bash
sbatch exp/sbatch_theta2_kmeans.sh
```

Outputs:

```text
configs/codebooks/
```

## Sample Generation

Small local check:

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_fid_sample.py \
  --model-depth 30 \
  --polar-quant int6_kmeans_int4 \
  --theta2-levels 1,1,3,3,3 \
  --out-dir fid_samples/test_fpga_core \
  --class-start 0 --class-end 0 \
  --samples-per-class 2
```

FID-style sharded generation:

```bash
MODEL_DEPTH=30 THETA2_LEVELS=1,1,3,3,3 \
  sbatch --array=0-7 exp/sbatch_fid_sample.sh int6_kmeans_int4
```

Outputs:

```text
fid_samples/
```

## KV Dump

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_polar_kv_infer.py
```

Default output:

```text
artifacts/polar_kv_dumps/
```

## Numeric Checks

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/verify_fp6_e3m2.py
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/audit_fp6_e3m2.py
```

## Notes For FPGA Work

- `utils/cordic.py` is the current software model for vectoring-mode CORDIC.
- `utils/polar_kv_quant.py` defines the unpacked software representation:
  `q1`, `q2`, `z`.
- A future FPGA branch should add an explicit packed bitstream format and
  golden test-vector export.
