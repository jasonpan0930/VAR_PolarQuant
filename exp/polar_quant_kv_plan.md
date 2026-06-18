# VAR 推論 Hierarchical Polar Quant KV Cache — 軟體模擬計畫（修訂 v2）

**64 維 K 向量整體一起 quant**，不是每 4 維一組獨立打包。

壓縮結果：**32 個 INT6 角 + 31 個 INT4 角 + 1 個 FP16 總長度**（共 **63 個角度**）。

---

## 1. 目標

| 項目 | 內容 |
|------|------|
| 階段 | Inference 軟體模擬（PyTorch） |
| 對象 | **K cache**（V 先 fp16） |
| 量化 | 角度 **均分 bin**，不用 k-means |
| 單位 | **整條 64 維** 一個 quant 物件 |

---

## 2. 為什麼是 32 + 31 + 1？

兩段式 **階層 Polar**：

| 階段 | 輸入 | 輸出 | 角度 bit |
|------|------|------|----------|
| **第一次 Polar**（32 次） | 64 個 **可正可負** 的 scalar，兩兩成對 | 32 個 **長度** + 32 個 **θ₁** | 各 **INT6**，θ₁ ∈ **(-π, π)** |
| **第二次 Polar**（31 次，合併樹） | 32 個 **長度 ≥ 0**，兩兩成對 | 逐層合併 → 最後 **1 個總長 z** + **31 個 θ₂** | 各 **INT4**，θ₂ ∈ **(0, π/2)** |

```
64 維 (x₀…x₆₃)
    │  32× 第一次: (x₂ᵢ, x₂ᵢ₊₁) → (yᵢ, θ₁ᵢ)   θ₁ᵢ → INT6
    ▼
32 個長度 y₀…y₃₁
    │  31× 第二次: 成對 (yₐ,yᵦ) → (z, θ₂)     θ₂ → INT4
    │              32→16→8→4→2→1  （完全二元樹）
    ▼
1 個總長 z  → FP16
```

**32 + 31 = 63 個角度**，**1 個 FP16 長度**。

---

## 3. 第一次 Polar（×32，INT6）

對 i = 0..31，取 pair `(x₂ᵢ, x₂ᵢ₊₁)`：

```
yᵢ = √(x₂ᵢ² + x₂ᵢ₊₁²)          # 長度 ≥ 0
θ₁ᵢ = atan2(x₂ᵢ₊₁, x₂ᵢ)       # ∈ (-π, π)
```

**量化 θ₁ᵢ（INT6，64 bins，均分 [-π, π)）**：

```
bin_w = 2π / 64
q₁ᵢ = clamp(floor((θ₁ᵢ + π) / bin_w), 0, 63)
θ̂₁ᵢ = -π + (q₁ᵢ + 0.5) * bin_w
```

**還原該 pair**（dequant 第一步）：

```
x₂ᵢ   = yᵢ cos(θ̂₁ᵢ)
x₂ᵢ₊₁ = yᵢ sin(θ̂₁ᵢ)
```

中間 **32 個 yᵢ** 只在 encode 時保留，**不写入 cache**（由第二次 Polar + 最終 z 推回）。

---

## 4. 第二次 Polar（×31，INT4，合併樹）

將 `[y₀…y₃₁]` 視為 **32 個非負長度**。按 **固定二元樹** 合併（實作用 array index，ASIC 友好）：

```
layer 0: 32 個 node → 16 對 (y[0],y[1]), (y[2],y[3]), …
layer 1: 16 個 node → 8 對
layer 2: 8 → 4
layer 3: 4 → 2
layer 4: 2 → 1   → 根 z_root (FP16)
```

每一對 **(a, b)**，a,b ≥ 0：

```
z = √(a² + b²)
θ₂ = atan2(b, a)    # ∈ [0, π/2]
```

**量化 θ₂（INT4，16 bins，均分 (0, π/2)）**：

```
bin_w = (π/2) / 16
q₂ = clamp(floor(θ₂ / bin_w), 0, 15)
θ̂₂ = (q₂ + 0.5) * bin_w
```

**Dequant 該對**：

```
â = z cos(θ̂₂)
b̂ = z sin(θ̂₂)
```

31 次合併產生 **31 個 q₂**；最後剩下 **z_root → FP16**。

合併順序在 encode/decode **必須完全一致**（建議左子=偶索引、右子=奇索引）。

---

## 5. Cache 裡存什麼（每個 64 維 K 向量）

| 欄位 | 數量 | bit/個 | 小計 |
|------|------|--------|------|
| 第一次角 q₁ | 32 | 6 | 192 bit |
| 第二次角 q₂ | 31 | 4 | 124 bit |
| 總長 z | 1 | 16 (FP16) | 16 bit |
| **合計** | | | **332 bit ≈ 41.5 B** |

對照 fp16 原始：**64 × 16 = 1024 bit ≈ 128 B**  
→ 約 **32.4%** payload（~3.1× 壓 K 本体）

Peak K cache（L=680, H=16, 16 layers, batch=1）粗估：

- FP16 K：~22 MB  
- 本方案：~**7 MB** 量级  

（打包对齐可能略增几 byte。）

---

## 6. Encode / Decode 總流程

### Encode（写入 KV cache）

```
k[64] fp16/fp32
  → 32× first polar  → q₁[32], y[32]
  → 31× second polar tree on y → q₂[31], z_root
  → pack PolarK64 { q1[32], q2[31], z_fp16 }
```

### Decode（attention 前）

```
unpack → dequant z_root, q₂[31]  → 還原 y[32]（自葉向根逆推）
      → dequant q₁[32], y[32]    → 還原 k[64]
      → K̂ fp16 送 attention
```

**不做 k-means**；所有角度 bin **等距**。

---

## 7. 與先前版本的差異

| 錯誤版本 | 本版（你的規格） |
|----------|------------------|
| 16 組 × 4 維各自 quant | **整條 64 維** 一次 hierarchial polar |
| 16×(z, q₁, q₂) | **1×z + 32×q₁ + 31×q₂** |
| Stage 0/1 決定 bit 寬 | **第一次/第二次 Polar 階段** 決定 INT6/INT4 |

---

## 8. 軟體整合（計畫）

```
utils/polar_kv_quant.py
  PolarK64.encode(k: Tensor[64]) -> PolarK64
  PolarK64.decode() -> k_hat[64]
  pack/unpack bits（可選，第一版可用 struct + numpy）

models/basic_var.py
  cache 存 PolarK64（或 packed bytes）而非 fp16 tensor
  forward 前 decode → k_hat，形状 [B,L,H,64]

exp/exp_polar_kv_infer.py
  baseline vs polar-K，同 seed/class
```

第一版：**decode 成 fp16 K 再 Flash Attention**。

---

## 9. 驗證

1. Round-trip：random k → encode → decode，量 **MSE / max err**  
2. 確認 31 次 merge 後 **y 全非负**；θ₂ 落在 (0, π/2)  
3. 端到端生圖 vs fp16 baseline  
4. 統計 q₁/q₂ 的 bin 使用率（是否浪費 bin）

---

## 10. 待確認

1. **Pair 順序**：是否 `(k[0],k[1]), (k[2],k[3]), …`？  
2. **合併樹**：是否 **相鄰配对** ` (y[0],y[1]), (y[2],y[3]), …` 逐层？（plan 默认是）  
3. 是否在 **64 维第一次 Polar 前** 加全局正交旋转 R@k？（PolarQuant 可选；v1 可不加）

---

## 11. 一句話

**64 维 K 先 32 次 2D Polar（INT6 角）得到 32 个长度，再 31 次 Q1 Polar 二元树合并（INT4 角）得到 1 个 FP16 总长度；cache 只存 32×6b + 31×4b + 16b，attention 前 decode 回 64 维。**

确认 §10 后实现 `polar_kv_quant.py` 并挂 inference。
