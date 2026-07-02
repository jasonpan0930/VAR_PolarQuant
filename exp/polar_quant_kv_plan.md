# FPGA-Oriented Polar KV Quantization Plan

This note is the working hardware plan for the remaining core branch.

## 1. Software Reference

Current reference implementation:

- `utils/polar_kv_quant.py`
- `utils/angle_quant.py`
- `utils/cordic.py`
- integration point: `models/basic_var.py::_update_kv_cache`

The software path is:

```text
K/V vector[64]
  -> CORDIC vectoring pairwise polar
  -> theta1 quantization
  -> CORDIC merge tree over 32 magnitudes
  -> theta2 quantization
  -> store q1[32], q2[31], z
  -> decode before attention
```

## 2. Data Format To Freeze

Default unpacked software tensors:

| Field | Shape per vector | Current storage | Hardware target |
| --- | ---: | --- | --- |
| `q1` | 32 | `uint8` holding 0..63 | packed INT6 |
| `q2` | 31 | `uint8` holding codebook index | packed INT4/variable |
| `z` | 1 | FP16 | FP16 or fixed/block-float |

Default INT6/INT4 payload:

```text
32 * 6 + 31 * 4 + 16 = 332 bits/vector
```

Open hardware decisions:

1. exact bit ordering for packed `q1`;
2. exact bit ordering for packed `q2`;
3. byte/word alignment for cache memory;
4. whether `z` remains IEEE FP16;
5. whether V uses a separate codebook or shares K codebooks.

## 3. Encoder RTL Blocks

Required blocks:

1. `cordic_vectoring`
   - full quadrant correction;
   - configurable iteration count;
   - gain compensation.
2. `theta1_quant`
   - nearest-codebook or uniform-bin index;
   - default 64-entry uniform LUT.
3. `theta2_quant`
   - default uniform/codebook nearest-index;
   - per-level codebook support for T1..T5.
4. `polar_tree_merge`
   - fixed adjacent merge tree: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
5. `bit_packer`
   - packs q1/q2/z into the frozen cache format.

## 4. Decoder RTL Blocks

Required blocks:

1. `bit_unpacker`;
2. theta LUT lookup for q1 and q2;
3. reverse merge tree using cos/sin LUT values;
4. pairwise reconstruction of 64-dim K/V vector;
5. output format conversion to attention datapath type.

Decode can use LUTs because q1/q2 are discrete indices.

## 5. Golden Test Vectors

Add a future export script that writes deterministic vectors:

```text
test_vectors/
  random_k64_seed0.json
  real_cache_block00_stage09.json
  codebook_my11333.json
```

Each vector should contain:

- input K/V vector;
- expected q1/q2/z;
- packed bytes once format is frozen;
- decoded vector;
- max error / MSE vs software.

## 6. Immediate Next Tasks

1. Add pack/unpack helpers to `utils/polar_kv_quant.py`.
2. Add a small `exp/export_fpga_vectors.py` script.
3. Replace decode-time `torch.cos/sin` with explicit LUT construction in software.
4. Run bit-exact tests for CORDIC iteration counts 7, 8, and 12.
5. Decide final K-only vs K+V FPGA scope.

## 7. Keep Out Of This Branch

The following are intentionally not part of this core branch:

- cross-block propagation plots;
- attention-drift heatmaps;
- f_hat spatial diagnostics;
- fc2 activation histograms;
- historical sample image dumps.

Those files can be recovered from `../VAR_polarQuant_backup/` if needed.
