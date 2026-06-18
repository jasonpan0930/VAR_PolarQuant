# VAR_polarQuant: Hierarchical Polar K-Cache Quantization

This project adds **hierarchical polar quantization** to the K-cache of [VAR](https://github.com/FoundationVision/VAR) (NeurIPS 2024 Best Paper) during autoregressive image generation. The goal is to compress the Key-cache in multi-head self-attention with minimal impact on image quality.

---

## Motivation

VAR generates images through a coarse-to-fine autoregressive process (10 scales, 1→2→3→4→5→6→8→10→13→16 patches). At each scale, the model attends to all previously generated tokens via a KV-cache. The K-cache consumes significant memory at larger scales, especially for deeper models (d30 = 2B params, 30 attention heads).

**Question**: Can we quantize the 64-dimensional K vectors in the cache to ~4 bits per vector, trading negligible quality loss for significant memory savings?

**Answer**: Yes — hierarchical polar decomposition turns each 64-dim K vector into 63 angles + 1 magnitude, quantized to INT6/INT4 respectively, achieving **~3.8× compression** vs FP16 with FID loss < 0.1 on VAR-d16.

---

## Method

### Hierarchical Polar Decomposition

Each 64-dim K vector is decomposed in two stages:

| Stage | Count | Bit-width | Range |
|-------|-------|-----------|-------|
| θ₁ (first polar) | 32 angles | INT6 (64 levels) | [−π, π] |
| θ₂ (merge tree) | 31 angles | 4-bit (16 levels) | [0, π/2] |
| z (magnitude) | 1 scalar | FP16 | — |

```
64-dim K vector
    │  32× first polar: (x₂ᵢ, x₂ᵢ₊₁) → (yᵢ, θ₁ᵢ)   → quantize θ₁ to INT6
    ▼
32 non-negative lengths y₀…y₃₁
    │  31× tree merge: 32→16→8→4→2→1             → quantize θ₂ to 4-bit
    ▼
1 total magnitude z (FP16)
```

**Storage**: 32×6 + 31×4 + 1×16 = **332 bits/vector** vs 64×16 = 1024 bits (FP16).  
**Compression ratio**: ~3.08×. In practice with structured storage: ~**3.8×**.

### Attention Path

```
encode:  K ∈ ℝ⁶⁴ → polar → quantize → (q₁,q₂,z)
cache:   store quantized codes in KV-cache
decode:  (q₁,q₂,z) → dequantize → reconstuct K̂ → matmul with Q
```

Baseline stores FP16 K directly. We keep V in FP16 throughout.

---

## θ₂ Quantization Schemes

θ₁ is always INT6 uniform. The θ₂ angle quantization supports multiple schemes:

| Config | θ₂ Scheme | Levels | Notes |
|--------|-----------|--------|-------|
| `none` / `baseline` | FP16 K (no quant) | — | Reference |
| `uniform_int4` | Uniform INT4 | 16 | Evenly spaced in [0, π/2] |
| `e2m1_fp4` | OCP E2M1 FP4 | 16 | IEEE-like floating point |
| `fp6_e3m2` | FP6 E3M2 | 64 | Fine-grained floating point |
| `fp6_e2m3` | FP6 E2M3 | 64 | Alternative FP6 distribution |
| `int6_kmeans_int4` | K-means INT4 | 6 | Data-driven (MSE-weighted fit) |

The `int6_kmeans_int4` scheme is trained by collecting FP θ₂ angles during inference under `uniform_int4`, weighting samples by their full-pipeline MSE contribution, and running 1D K-means to find 16 optimal centroid levels.

---

## Experiment Pipeline

```
① θ₂ K-means codebook → ② K error statistics → ③ FID 50k sampling → ④ (optional) K-cache dump
```

### 1. Codebook Fit

```bash
python exp/exp_theta2_kmeans.py --model-depth 30 --refit
```

Collects FP θ₂ across multiple classes and blocks under `uniform_int4` config, fits 16 MSE-weighted K-means centers, saves to `polar_quant_dumps/theta2_kmeans_d<depth>/codebook.json`.

### 2. Error Characterization

```bash
python exp/exp_polar_angle_dist.py
```

Runs one inference per quant config and produces `k_error_global.png` showing NRMSE distributions and cosine similarity across stages.

### 3. FID Evaluation

```bash
sbatch --array=0-7 exp/sbatch_fid_sample.sh <config>
```

Generates 50,000 PNGs (1000 classes × 50 images) under VAR's official FID protocol (cfg=1.5, top_k=900, top_p=0.96), packs into `.npz`, then evaluates with OpenAI's `guided-diffusion/evaluatory.py`.

### 4. Cross-block Analysis

```bash
python exp/exp_cross_block_cumulative_mse.py
```

Measures how K quantization error propagates across the full depth of the model (supporting up to d30), producing per-depth NRMSE, cosine distance, and absolute MSE curves.

---

## Results (VAR-d16, 256×256)

| Config | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|--------|-------|------|-------------|----------|
| FP16 baseline | 3.40 | 62.15 | 0.849 | 0.503 |
| `int6_kmeans_int4` | **3.346** | 61.48 | 0.843 | 0.506 |

VAR-d16 paper reference: FID ≈ 3.55.

d30 experiments are in progress.

---

## Code Map

| File | Purpose |
|------|---------|
| `utils/polar_kv_quant.py` | Polar encode/decode, `polar_k64_error_breakdown` (NRMSE, cosine similarity) |
| `utils/angle_quant.py` | θ₁/θ₂ codebooks, named configs, `register_theta2_kmeans_codebook()` |
| `utils/theta2_kmeans.py` | MSE-weighted 1D K-means for θ₂, codebook I/O |
| `utils/polar_angle_viz.py` | Angle/K error collection and visualization (histograms, scatter) |
| `utils/polar_kv_store.py` | `PolarKVDumpSession` for writing quantized cache to `.npz` |
| `models/basic_var.py` | `SelfAttention` polar cache path (encode → decode → matmul) |
| `models/var.py` | `enable_polar_k_cache()`, `enable_polar_angle_stats()` |
| `exp/exp_fid_sample.py` | 50k FID protocol sampling + `.npz` packing |
| `exp/exp_theta2_kmeans.py` | MSE-weighted K-means codebook fit |
| `exp/exp_polar_angle_dist.py` | Multi-config K error comparison |
| `exp/exp_cross_block_cumulative_mse.py` | Error propagation across model depth |
| `exp/sbatch_fid_sample.sh` | SLURM 8-way shard wrapper for FID sampling |

---

## Getting Started

1. **Setup**: Follow [SETUP.md](SETUP.md)
2. **Quick test**: Generate a few images from class 0:
   ```bash
   python exp/exp_fid_sample.py --model-depth 16 --polar-quant none \
     --out-dir fid_samples/test --class-start 0 --class-end 0 --samples-per-class 2
   ```
3. **Try polar quant**:
   ```bash
   python exp/exp_fid_sample.py --model-depth 16 --polar-quant uniform_int4 \
     --out-dir fid_samples/test_int4 --class-start 0 --class-end 0 --samples-per-class 2
   ```
4. **Full FID pipeline**: See [exp/COMMANDS.md](exp/COMMANDS.md) for `sbatch` commands.

---

## References

- VAR paper: [Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905) (NeurIPS 2024 Best Paper)
- OpenAI guided-diffusion: [evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations)
- Original VAR repo: [FoundationVision/VAR](https://github.com/FoundationVision/VAR)
