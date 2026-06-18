# Polar Quant K Cache（VAR_polarQuant）

在 VAR **inference** 時對 **K cache** 做 hierarchical polar 量化；**V cache** 維持 FP16。本 repo 的實驗涵蓋：量化 K 存檔、θ₂ K-means codebook、各配置的 K 誤差統計圖、以及官方協議下的 FID/IS 基準比較。

指令用法（`srun` / `sbatch` / CLI 參數）見 [`COMMANDS.md`](COMMANDS.md)。

---

## K 向量編碼（64 維）

| 欄位 | 數量 | 位寬 | 說明 |
|------|------|------|------|
| θ₁ | 32 | 6 bit（INT6 uniform） | 第一次 polar 分解 |
| θ₂ | 31 | 4 bit | 合併樹上的第二次 polar 角 |
| z | 1 | FP16 | 總長度（模長） |

Attention 路徑：**encode → 存量化碼 → decode → matmul**；baseline 則全程 FP16 K。

---

## 程式地圖

| 路徑 | 作用 |
|------|------|
| `utils/polar_kv_quant.py` | encode/decode、`polar_k64_error_breakdown`（含 `k_norm`、cosine similarity） |
| `utils/angle_quant.py` | θ₁/θ₂ codebook、named configs、`register_theta2_kmeans_codebook()` |
| `utils/theta2_kmeans.py` | MSE 加權 K-means 擬合與 codebook 讀寫 |
| `utils/polar_angle_viz.py` | 角度/K 誤差收集、`k_error_global.png`、`k_error_mse.png` |
| `utils/polar_kv_store.py` | `PolarKVDumpSession` 寫 npz + manifest |
| `models/basic_var.py` | `SelfAttention` polar cache 路徑 |
| `models/var.py` | `enable_polar_k_cache()`、`enable_polar_angle_stats()` |

---

## Polar quant 配置（`polar_quant=`）

θ₁ 預設皆為 **INT6 uniform**（64 levels，\([-π, π]\)）。

| 配置名 | θ₂ 方案 | 說明 |
|--------|---------|------|
| `none` / `baseline` | — | FP16 K cache（對照組） |
| `uniform_int4` | uniform INT4 | 16 levels，\([0, π/2]\) 等距 |
| `e2m1_fp4` | OCP E2M1 FP4 | 16 codes，線性映到 \([0, π/2]\) |
| `fp6_e3m2` | FP6 E3M2 | 64 codes → 線性映到 \([0, π/2]\)，0 對應 π/4 |
| `fp6_e2m3` | FP6 E2M3 | 同上 |
| `int6_kmeans_int4` | K-means INT4 | 16 levels；**需先有** `codebook.json` |

FP 格式 θ₂ 映射：\(\theta_2 = \frac{v - v_{\min}}{v_{\max} - v_{\min}} \cdot \frac{\pi}{2}\)（零點 → π/4）。

啟用方式：`var.enable_polar_k_cache(True, polar_quant='<config>')`；K-means 配置需先 `register_theta2_kmeans_codebook()`（`exp_theta2_kmeans.py` / `exp_polar_angle_dist.py` 會自動載入）。

---

## 實驗總覽

| # | 腳本 | 做了什麼 | 主要產物 |
|---|------|----------|----------|
| 1 | `exp_theta2_kmeans.py` | 在 `uniform_int4` 下收集 FP θ₂，以 full-pipeline MSE 加權做 16 中心 K-means | `polar_quant_dumps/theta2_kmeans/codebook.json`、`codebook_vs_uniform.png` |
| 2 | `exp_polar_angle_dist.py` | 載入 codebook，對 5 種 polar 配置各跑一次推論，統計 K 量化誤差並出圖 | `angle_plots/<config>/k_error_global.png`、`k_error_mse.png`、`sample_var.png`；`sample_original.png`（FP16 baseline） |
| 3 | `exp_fid_sample.py` + `sbatch_fid_sample.sh` | 官方 FID 協議生成 50k PNG，打包 `.npz`，再交 OpenAI evaluator | `fid_samples/<config>_d16/`、`fid_samples/<config>_d16.npz` |
| 4 | `exp_polar_kv_infer.py` | 推論並 dump 量化 K tensor（研究用存檔） | `polar_quant_dumps/stage*/block*_k_polar.npz`、`manifest.json`、對照 sample 圖 |
| — | `verify_fp6_e3m2.py` / `audit_fp6_e3m2.py` | FP6 E3M2 解碼與 polar roundtrip 單元檢查 | stdout 通過/失敗 |
| — | `exp_kvCache.py` / `exp_fc2_input_hist.py` | 舊 VAR 分析腳本（QKT heatmap、fc2 輸入 histogram） | `kv_records/`、`fc2_records/` |

**建議依賴順序**：① K-means codebook → ② 角度/K 誤差圖 → ③ FID 50k → ④（可選）K cache dump。

---

## 實驗一：θ₂ K-means codebook

**目的**：用資料驅動的 16 個 θ₂ 中心取代 uniform INT4，供 `int6_kmeans_int4` 使用。

**做法**（`utils/theta2_kmeans.py`）：

1. `uniform_int4` 推論，收集 FP θ₂；
2. 每個 K 向量的 **full-pipeline MSE**（`mse_after_full`）作權重，對該向量 31 個 θ₂ 重複加權；
3. 一維加權 K-means（K=16），中心排序後寫入 JSON。

**產物**：

| 路徑 | 內容 |
|------|------|
| `polar_quant_dumps/theta2_kmeans/codebook.json` | 16 個 θ₂ 中心（rad）+ fit meta |
| `polar_quant_dumps/theta2_kmeans/codebook_vs_uniform.png` | 與 uniform 對照 |
| `polar_quant_dumps/angle_plots/int6_kmeans_int4/k_error_*.png` | 擬合後該配置單獨評估（腳本內可選） |

**注意**：codebook 只需在資料/設定變更時重做；`exp_polar_angle_dist.py` **不會**重新 K-means，只載入既有 JSON。

---

## 實驗二：各配置的 K 誤差圖

**目的**：在固定推論設定下，比較各 θ₂ 方案對 K 向量的量化誤差（預設只統計 **block 15**）。

**流程**：

1. 載入 `codebook.json`，註冊 `int6_kmeans_int4`；
2. 先跑 FP16 baseline，存 `sample_original.png`；
3. 依序跑 `uniform_int4`、`e2m1_fp4`、`fp6_e3m2`、`fp6_e2m3`、`int6_kmeans_int4`，各配置獨立一次推論。

**輸出**（每配置一個子目錄）：

```
polar_quant_dumps/angle_plots/
  sample_original.png
  uniform_int4/
    k_error_global.png    # NRMSE (%) + cosine similarity（雙 panel）
    k_error_mse.png       # 絕對 MSE histogram（三階段：僅 θ₁ / +θ₂ 樹 / 完整 pipeline）
    sample_var.png
  e2m1_fp4/ ...
  fp6_e3m2/ ...
  fp6_e2m3/ ...
  int6_kmeans_int4/ ...
```

| 圖 | 內容 |
|----|------|
| `k_error_global.png` | **NRMSE**（MSE/k_norm²，x 軸 0–1.5%，標 `%`）與 **cosine similarity** 分布 |
| `k_error_mse.png` | 舊版絕對 **MSE** 三階段 histogram |

其他角度圖預設關閉（`polar_angle_viz.py` 中 `PLOT_K_ERROR_GLOBAL_ONLY = True`）。

**可調參數**（腳本頂部常數）：`MODEL_DEPTH`、`TARGET_BLOCKS`（預設 `(15,)`）、`CLASS_LABELS`、`SEED`、`CFG` 等。缺少 `codebook.json` 時會報錯並提示先跑實驗一。

---

## 實驗三：存量化 K cache（可選）

**目的**：將量化後的 K cache（q1、q2、z）逐 stage/block 寫入 npz，供離線分析或視覺對照。

**產物**：

```
polar_quant_dumps/
  sample_baseline.png      # FP16 K
  sample_polar.png         # polar quant（腳本內 THETA2_QUANT）
  sample_compare.png
  manifest.json
  stage00_pn1/block00_k_polar.npz
  ...
```

每個 `.npz`：`q1` `(B,L,H,32)` uint8、`q2` `(B,L,H,31)` uint8、`z` float16、`_meta_json`。

---

## 實驗四：FID / IS / Precision / Recall

**目的**：在 VAR 官方基準設定下，比較 baseline（FP16 K）與各 polar 配置對生成品質的影響。

| 參數 | FID 基準值 |
|------|------------|
| `cfg` | 1.5 |
| `top_k` | 900 |
| `top_p` | 0.96 |
| `more_smooth` | **False** |
| 張數 | **50,000**（1000 類 × 50） |
| 格式 | PNG（勿 JPEG） |
| 模型 | `var_d16.pth`（`MODEL_DEPTH=16`） |

**生成**：8-way class shard（`sbatch --array=0-7`），輸出目錄 `fid_samples/<polar_quant>_d16/`，檔名 `{class:04d}_{sample:02d}.png`。全部分片完成後資料夾內應有 **50,000** 張 PNG，再 `--pack-npz-only` 得到同名的 `.npz`。

**評估**：OpenAI [guided-diffusion/evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations) 的 `evaluator.py`，reference 為 `VIRTUAL_imagenet256_labeled.npz`（256×256）。輸出 FID、sFID、Precision、Recall、Inception Score。

**本 repo 已跑完（50k PNG + `.npz`）**：

| 配置 | 目錄 / npz |
|------|------------|
| baseline | `none_d16` / `none_d16.npz` |
| `uniform_int4` | `uniform_int4_d16` |
| `e2m1_fp4` | `e2m1_fp4_d16` |
| `fp6_e2m3` | `fp6_e2m3_d16` |
| `int6_kmeans_int4` | `int6_kmeans_int4_d16` |

**尚未完成 FID 50k**：`fp6_e3m2`（angle 誤差圖已有，FID 樣本未生成）。

**已確認的 evaluator 結果**（depth-16，reference npz 正確時）：

| 配置 | FID | IS | Precision | Recall |
|------|-----|-----|-----------|--------|
| `none_d16`（baseline） | 3.404 | 62.15 | 0.849 | 0.503 |
| `int6_kmeans_int4_d16` | 3.346 | 61.48 | 0.843 | 0.506 |

（VAR-d16 論文參考 FID ≈ 3.55。）

**Polar 有生效**：log 中 baseline 約 ~10 s/class、`polar mode: baseline_fp16_k`；polar 配置約 ~24 s/class、`polar mode: <config>`。

---

## 輔助腳本

| 腳本 | 用途 |
|------|------|
| `verify_fp6_e3m2.py` | OCP FP6 E3M2 解碼 spot check + encode/decode roundtrip |
| `audit_fp6_e3m2.py` | 完整 FP6 E3M2 表格與 polar pipeline 審計 |
| `exp_kvCache.py` | 輸出 stage9/block15 QKT heatmap |
| `exp_fc2_input_hist.py` | fc2 輸入 GELU(fc1(x)) histogram |

---

## TWCC 注意事項

- **勿在 login 節點跑 CUDA 推論**；用 `srun` 或 `sbatch`（見 `COMMANDS.md`）。
- `gtest` walltime 上限 30 分；FID 全量建議 `gp2d` + array 分片；`--skip-existing` 可斷點續跑。
- FID evaluator 需 `module load cuda/11.7`（TF 2.12 + pip cuDNN）；勿用預設 cuda/12.8。
- Conda：`/home/jasonpan0930/.conda/envs/var_env/bin/python`。

更完整的設計說明見 [`polar_quant_kv_plan.md`](polar_quant_kv_plan.md)。
