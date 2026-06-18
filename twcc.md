# TWCC / 台灣杉二號使用筆記

> 個人備忘：這份筆記整理目前在 TWCC / 台灣杉二號跑 VAR 時用到的指令、環境、GPU 測試、互動模式與常見問題。

---

## 0. 目前已知設定

```text
TWCC Host: ln01.twcc.ai
TWCC 帳號: jasonpan0930
Project ID: MST112145
VAR repo path: ~/var_research/VAR
Conda env: var_env
測試 env: test
GPU: Tesla V100-SXM2-32GB
GPU driver: 535.161.08
Driver supported CUDA: 12.2
目前可用 PyTorch: torch 2.1.2+cu118
建議 NumPy: numpy < 2
```

---

## 1. SSH 登入 TWCC

### 基本登入

```bash
ssh jasonpan0930@ln01.twcc.ai
```

登入時通常需要：

1. 主機密碼
2. OTP 動態驗證碼

---

## 2. SSH Config 設定

在本機 Windows：

```powershell
notepad C:\Users\jason\.ssh\config
```

建議設定：

```sshconfig
Host twcc
    HostName ln01.twcc.ai
    User jasonpan0930
    Ciphers aes128-ctr,aes192-ctr,aes256-ctr
    MACs hmac-sha2-256,hmac-sha2-512
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

之後可以直接：

```powershell
ssh twcc
```

### 如果遇到 Corrupted MAC on input

錯誤類似：

```text
Corrupted MAC on input.
ssh_dispatch_run_fatal: Connection to 203.145.219.98 port 22: message authentication code incorrect
```

可以用指定 cipher / MAC 的方式登入：

```powershell
ssh -c aes128-ctr,aes192-ctr,aes256-ctr -m hmac-sha2-256,hmac-sha2-512 jasonpan0930@ln01.twcc.ai
```

如果成功，就把同樣設定寫進 `~/.ssh/config`。

---

## 3. 台灣杉二號基本觀念

台灣杉二號不是像本機 GPU 一樣直接跑：

```bash
python train.py
```

而是要透過 Slurm 排程系統：

```bash
sbatch job.sh
```

或使用互動測試模式：

```bash
srun ... --pty bash
```

### Login node vs GPU node

登入後通常在：

```bash
un-ln01
```

這是 login node。不要在這裡跑大型 GPU / CPU 程式。

進入 GPU node 後 hostname 會像：

```bash
gn1001.twcc.ai
```

---

## 4. 查看節點與 GPU

查看目前在哪台機器：

```bash
hostname
```

查看 GPU：

```bash
nvidia-smi
```

成功時會看到類似：

```text
Tesla V100-SXM2-32GB
Driver Version: 535.161.08
CUDA Version: 12.2
```

---

## 5. 查看 queue / job

查看 queue / partition：

```bash
sinfo
```

查看自己的 job：

```bash
squeue -u $USER
```

取消 job：

```bash
scancel JOBID
```

查看今天跑過的 job：

```bash
sacct -u $USER --starttime today \
  --format=JobID,JobName,Partition,Account,AllocGRES,Elapsed,State
```

查看特定 job：

```bash
sacct -j JOBID \
  --format=JobID,JobName,Partition,Account,AllocGRES,AllocTRES,Elapsed,Start,End,State
```

---

## 6. 30 分鐘互動 GPU 模式：gtest

適合 debug、測環境、跑小程式。

```bash
srun -A MST112145 \
  -p gtest \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=90G \
  -t 00:30:00 \
  --pty bash
```

成功後 prompt 會從：

```bash
[jasonpan0930@un-ln01 ...]$
```

變成類似：

```bash
[jasonpan0930@gn1001 ...]$
```

### 結束互動 GPU session

```bash
exit
```

回到 login node 後確認沒有繼續占 GPU：

```bash
squeue -u $USER
```

如果還有 job：

```bash
scancel JOBID
```

> `-t 00:30:00` 是最多 30 分鐘，不是一定會用滿。中途 `exit` 後資源會釋放。

---

## 7. Conda / Python 環境

### 重要觀念

TWCC 的 conda module 有時會出現初始化問題，例如：

```text
bash: conda: command not found
CondaError: Run 'conda init' before 'conda activate'
```

因此在 `sbatch` 或不穩定情境下，最穩定的做法是直接使用 env 內的絕對路徑 python。

例如：

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python script.py
```

或：

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/pip install PACKAGE
```

### 如果在 login node 可以正常 activate

```bash
conda activate var_env
```

確認是否真的進到對的 env：

```bash
which python
which pip
python -c "import sys; print(sys.executable)"
```

理想輸出：

```text
/home/jasonpan0930/.conda/envs/var_env/bin/python
/home/jasonpan0930/.conda/envs/var_env/bin/pip
```

---

## 8. PyTorch / CUDA 設定

目前測試可用：

```text
torch==2.1.2+cu118
torchvision==0.16.2+cu118
torchaudio==2.1.2+cu118
numpy<2
```

安裝方式：

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

固定 NumPy：

```bash
pip install "numpy<2"
```

檢查版本：

```bash
python -c "import torch, numpy; print(torch.__version__); print(torch.version.cuda); print(numpy.__version__)"
```

理想：

```text
2.1.2+cu118
11.8
1.26.x
```

### 測 GPU 是否可用

在 GPU node 上：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

理想：

```text
2.1.2+cu118
True
Tesla V100-SXM2-32GB
```

### 目前不建議直接升級到 cu126 / cu128

因為 `nvidia-smi` 顯示 driver 支援 CUDA 12.2，因此安全範圍大約是：

```text
cu118 ✅
cu121 可嘗試
cu124 / cu126 / cu128 不建議
```

---

## 9. VAR repo 位置與環境

目前 repo：

```bash
cd ~/var_research/VAR
```

內容應有：

```text
demo_sample.ipynb
models/
utils/
requirements.txt
train.py
trainer.py
```

目前權重：

```bash
ls -lh *.pth
```

應看到：

```text
vae_ch160v4096z32.pth
var_d16.pth
```

---

## 10. VAR requirements

官方 `requirements.txt`：

```text
torch~=2.1.0
Pillow
huggingface_hub
numpy
pytz
transformers
typed-argument-parser
```

因為 torch / numpy 需要固定版本，不建議直接無腦：

```bash
pip install -r requirements.txt
```

建議：

```bash
pip install Pillow huggingface_hub pytz transformers typed-argument-parser
```

並另外固定：

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2"
```

---

## 11. 下載 VAR 權重

在 `~/var_research/VAR`：

```bash
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth
```

若 `wget` 不行：

```bash
curl -L -O https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
curl -L -O https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth
```

確認：

```bash
ls -lh *.pth
```

---

## 12. 官方 VAR demo_sample.py 需要的小改動

官方 notebook / script 最後可能有：

```python
chw.show()
```

TWCC 沒有 GUI，會出現：

```text
xdg-open: no method available for opening ...
Unable to connect to VS Code server...
```

所以改成：

```python
chw.save("sample_d16_official.png")
print("saved to sample_d16_official.png")
```

### 官方核心流程

```python
from models import VQVAE, build_vae_var

MODEL_DEPTH = 16
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
)

vae.load_state_dict(torch.load("vae_ch160v4096z32.pth", map_location="cpu"), strict=True)
var.load_state_dict(torch.load("var_d16.pth", map_location="cpu"), strict=True)

recon_B3HW = var.autoregressive_infer_cfg(
    B=B,
    label_B=label_B,
    cfg=cfg,
    top_k=900,
    top_p=0.95,
    g_seed=seed,
    more_smooth=more_smooth,
)
```

---

## 13. 在 gtest 跑 VAR demo

進入互動 GPU：

```bash
srun -A MST112145 \
  -p gtest \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=90G \
  -t 00:30:00 \
  --pty bash
```

到 repo：

```bash
cd ~/var_research/VAR
```

如果 `conda activate` 不穩，直接用絕對路徑：

```bash
/home/jasonpan0930/.conda/envs/var_env/bin/python demo_sample.py
```

若 activate 正常：

```bash
conda activate var_env
python demo_sample.py
```

成功時會看到：

```text
prepare finished.
saved to sample_d16_official.png
```

確認圖片：

```bash
ls -lh sample_d16_official.png
```

---

## 14. 把圖片抓回本機

在本機 PowerShell：

```powershell
scp twcc:~/var_research/VAR/sample_d16_official.png .
```

或完整寫法：

```powershell
scp jasonpan0930@ln01.twcc.ai:~/var_research/VAR/sample_d16_official.png .
```

---

## 15. sbatch 範例：GPU 測試

建立 `gpu_check.sh`：

```bash
#!/bin/bash
#SBATCH -J gpucheck
#SBATCH -A MST112145
#SBATCH -p gtest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH -t 00:05:00

hostname
nvidia-smi

/home/jasonpan0930/.conda/envs/var_env/bin/python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
```

送出：

```bash
sbatch gpu_check.sh
```

查看：

```bash
squeue -u $USER
cat slurm-JOBID.out
```

---

## 16. sbatch 範例：跑 VAR demo

建立 `run_var_d16.sh`：

```bash
#!/bin/bash
#SBATCH -J var_d16
#SBATCH -A MST112145
#SBATCH -p gtest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH -t 00:30:00

cd ~/var_research/VAR
hostname
nvidia-smi

/home/jasonpan0930/.conda/envs/var_env/bin/python demo_sample.py
```

送出：

```bash
sbatch run_var_d16.sh
```

查結果：

```bash
squeue -u $USER
cat slurm-JOBID.out
ls -lh sample_d16_official.png
```

---

## 17. 查看 TWCC 花費 / 用量

SSH 裡可以看 job 使用時間，但正式額度 / 花費要去 TWAI / TWCC 網頁看。

網頁路徑大概是：

```text
TWAI / TWCC 網站 → Member Center → Project → Credits Information / Resource Usage
```

SSH 裡查目前 job：

```bash
squeue -u $USER
```

查今天 job：

```bash
sacct -u $USER --starttime today \
  --format=JobID,JobName,Partition,Account,AllocGRES,Elapsed,State
```

---

## 18. 常見錯誤與解法

### Missing assigned project

錯誤：

```text
sbatch: error: Missing assigned project, try to use --account=<project_id>
```

解法：加上：

```bash
#SBATCH -A MST112145
```

或 srun 用：

```bash
srun -A MST112145 ...
```

---

### CUDA initialization: driver too old

錯誤：

```text
The NVIDIA driver on your system is too old
```

目前解法：不要裝太新的 torch / CUDA wheel。使用：

```bash
torch==2.1.2+cu118
```

安裝：

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

---

### NumPy 2.x warning

錯誤：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

解法：

```bash
pip install "numpy<2"
```

---

### transformers warning

警告：

```text
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.1.2+cu118
```

目前 VAR demo 不依賴 HuggingFace `transformers` 的 model backend，所以可以先忽略。VAR 使用的是 repo 自己的 `models/`。

---

### xdg-open / VS Code server error

錯誤：

```text
xdg-open: no method available for opening ...
Unable to connect to VS Code server ...
```

原因：程式嘗試開圖片視窗。

解法：移除：

```python
chw.show()
```

改成：

```python
chw.save("sample_d16_official.png")
```

---

## 19. 目前研究下一步

已完成 baseline：

```text
官方 VAR-d16 inference 成功
輸出 sample_d16_official.png
```

後續建議：

1. 整理 `demo_sample.py` 成 `official_eval.py`
2. 加 CLI 參數：seed、cfg、class labels、output path
3. 加 activation hook
4. 收集 layer-wise / token-wise outlier
5. 做 PTQ calibration
6. 嘗試 token merging
7. 嘗試 polar quant on VAR activations / weights

---

## 20. 快速狀態摘要

之後開新對話可直接貼：

```text
TWCC VAR baseline 狀態：
- SSH: ssh twcc / ssh jasonpan0930@ln01.twcc.ai
- Project ID: MST112145
- repo: ~/var_research/VAR
- env: var_env
- torch: 2.1.2+cu118
- CUDA runtime: 11.8
- GPU node driver: 535.161.08, CUDA 12.2
- GPU: Tesla V100-SXM2-32GB
- numpy: <2
- weights:
  - vae_ch160v4096z32.pth
  - var_d16.pth
- official demo 已成功生成：sample_d16_official.png
- gtest command:
  srun -A MST112145 -p gtest --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=90G -t 00:30:00 --pty bash
```
