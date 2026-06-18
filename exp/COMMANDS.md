# Polar Quant 指令用法

本文件只描述**怎麼跑**；實驗目的與產物說明見 [`README_polar.md`](README_polar.md)。

---

## 共通設定

```text
專案根目錄:  /home/jasonpan0930/var_research/VAR_polarQuant
Python:      /home/jasonpan0930/.conda/envs/var_env/bin/python
SLURM 帳號:  -A MST112145
```

所有 Python 實驗腳本預期在專案根目錄執行（腳本內會 `os.chdir(ROOT)`）：

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant
```

---

## TWCC：互動 GPU 單次任務（`srun`）

適合角度圖、codebook 擬合、除錯用 FID 小批次。

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant

srun -A MST112145 -p gtest --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=90G -t 00:30:00 \
  /home/jasonpan0930/.conda/envs/var_env/bin/python exp/<腳本名>.py
```

- `-t` 為時間上限；任務結束後 SLURM 自動釋放資源。
- `gtest` 最長 30 分；超時可改 `-p gp2d` 並加長 `-t`。
- 互動 shell：`srun ... --pty bash`（手動除錯用）。

---

## TWCC：批次 FID 分片（`sbatch`）

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant
mkdir -p logs

# 8 個 array task，每片約 125 個 class；$1 = polar 配置名
sbatch --array=0-7 exp/sbatch_fid_sample.sh none
sbatch --array=0-7 exp/sbatch_fid_sample.sh uniform_int4
sbatch --array=0-7 exp/sbatch_fid_sample.sh e2m1_fp4
sbatch --array=0-7 exp/sbatch_fid_sample.sh fp6_e3m2
sbatch --array=0-7 exp/sbatch_fid_sample.sh fp6_e2m3
sbatch --array=0-7 exp/sbatch_fid_sample.sh int6_kmeans_int4
```

- 輸出 log：`logs/fid_var_fid_<JOBID>_<ARRAYID>.out`
- 輸出 PNG：`fid_samples/<配置>_d16/`（由腳本內 `MODEL_DEPTH` 決定，預設 16）
- 環境變數（可選）：`MODEL_DEPTH=16`、`PYTHON=...`

---

## `exp/exp_theta2_kmeans.py`

無 CLI；常數在檔案頂部（`MODEL_DEPTH`、`CLASS_LABELS`、`TARGET_BLOCKS` 等）。

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_theta2_kmeans.py
```

產物：`polar_quant_dumps/theta2_kmeans/codebook.json`

---

## `exp/exp_polar_angle_dist.py`

無 CLI；常數在檔案頂部。需先有 `polar_quant_dumps/theta2_kmeans/codebook.json`。

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_polar_angle_dist.py
```

產物：`polar_quant_dumps/angle_plots/<config>/`

---

## `exp/exp_polar_kv_infer.py`

無 CLI；常數在檔案頂部（`THETA2_QUANT`、`DUMP_BLOCKS` 等）。

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_polar_kv_infer.py
```

產物：`polar_quant_dumps/`（npz + sample 圖）

---

## `exp/exp_fid_sample.py`

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--model-depth` | `16` | `16` / `20` / `24` / `30` |
| `--polar-quant` | `none` | `none`/`baseline` = FP16 K；否則 polar 配置名 |
| `--out-dir` | （必填） | PNG 輸出目錄 |
| `--class-start` | `0` | 起始 class（含） |
| `--class-end` | `999` | 結束 class（含） |
| `--samples-per-class` | `50` | 每類張數 |
| `--cfg` | `1.5` | classifier-free guidance |
| `--top-k` | `900` | |
| `--top-p` | `0.96` | |
| `--seed-base` | `0` | `g_seed = seed_base + class*100 + sample_idx` |
| `--kmeans-codebook` | `polar_quant_dumps/theta2_kmeans/codebook.json` | |
| `--skip-existing` | off | 已存在 PNG 則跳過 |
| `--dry-run` | off | 只印計畫不生成 |
| `--pack-npz-only` | off | 只把既有 PNG 目錄打包成 `.npz` |

`--polar-quant` 合法值：`none`、`baseline`、`uniform_int4`、`e2m1_fp4`、`fp6_e3m2`、`fp6_e2m3`、`int6_kmeans_int4`（後者需 codebook）。

### 除錯：少量 class

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant

srun -A MST112145 -p gtest --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=90G -t 02:00:00 \
  /home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_fid_sample.py \
  --polar-quant none \
  --out-dir fid_samples/none_d16 \
  --class-start 0 --class-end 9 \
  --samples-per-class 2 \
  --dry-run
```

去掉 `--dry-run` 即實際生成。

### 單機跑一段 class 區間

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_fid_sample.py \
  --polar-quant int6_kmeans_int4 \
  --out-dir fid_samples/int6_kmeans_int4_d16 \
  --class-start 0 --class-end 124 \
  --skip-existing
```

### 打包 50k PNG → npz

目錄內須恰好 **50,000** 張 `{class:04d}_{sample:02d}.png`：

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_fid_sample.py \
  --pack-npz-only \
  --out-dir fid_samples/none_d16
# → fid_samples/none_d16.npz
```

---

## `exp/sbatch_fid_sample.sh`

SLURM 批次腳本；**第一個 positional 參數**為 `--polar-quant` 值。

```bash
sbatch --array=0-7 exp/sbatch_fid_sample.sh <polar_quant>
```

腳本內固定：

- `-p gp2d`、`-t 04:00:00`、`--array=0-7`
- 自動計算 `CLASS_START` / `CLASS_END`（8 等分 1000 類）
- 呼叫 `exp_fid_sample.py` 並帶 `--skip-existing`

可編輯腳本頂部 `#SBATCH` 行改 partition、array 大小、log 路徑。

---

## FID 評估：`guided-diffusion/evaluations/run_fid_eval.sh`

Reference npz 預設為同目錄下的 `VIRTUAL_imagenet256_labeled.npz`。

```bash
cd /home/jasonpan0930/var_research/guided-diffusion/evaluations

# GPU（gtest 單次 srun 範例）
srun -A MST112145 -p gtest --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=32G -t 01:00:00 \
  bash run_fid_eval.sh \
  /home/jasonpan0930/var_research/VAR_polarQuant/fid_samples/none_d16.npz

# 指定 reference 路徑（可選第二參數）
bash run_fid_eval.sh SAMPLE.npz VIRTUAL_imagenet256_labeled.npz
```

腳本會 `module load cuda/11.7` 並設定 pip cuDNN/cuBLAS 的 `LD_LIBRARY_PATH`，再跑 `evaluator.py REF SAMPLE`。

直接呼叫 evaluator（需自行 load cuda/11.7 與 LD_LIBRARY_PATH）：

```bash
cd /home/jasonpan0930/var_research/guided-diffusion/evaluations
/home/jasonpan0930/.conda/envs/var_env/bin/python evaluator.py \
  VIRTUAL_imagenet256_labeled.npz \
  /home/jasonpan0930/var_research/VAR_polarQuant/fid_samples/none_d16.npz
```

---

## `exp/verify_fp6_e3m2.py` / `exp/audit_fp6_e3m2.py`

無 CLI；在 CPU 即可。

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/verify_fp6_e3m2.py
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/audit_fp6_e3m2.py
```

---

## `exp/exp_kvCache.py` / `exp/exp_fc2_input_hist.py`

無 CLI；常數在檔案頂部。

```bash
cd /home/jasonpan0930/var_research/VAR_polarQuant
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_kvCache.py
/home/jasonpan0930/.conda/envs/var_env/bin/python exp/exp_fc2_input_hist.py
```

---

## 常用檢查

```bash
# 某配置 PNG 數量（應為 50000）
find fid_samples/none_d16 -name '*.png' | wc -l

# 看 FID job log
tail -f logs/fid_var_fid_<JOBID>_0.out

# sbatch 佇列
squeue -u $USER
```
