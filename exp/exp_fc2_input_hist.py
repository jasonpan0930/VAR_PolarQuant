"""
Histogram of fc2 input: GELU(fc1(x)) at stage9 / block15.

  cd VAR && python exp/exp_fc2_input_hist.py
  -> fc2_records/stage09_pn16_block15_fc2_input_hist.png
"""
import os
import sys
from pathlib import Path

import torch

VAR_ROOT = Path(__file__).resolve().parents[1]
if str(VAR_ROOT) not in sys.path:
    sys.path.insert(0, str(VAR_ROOT))
os.chdir(VAR_ROOT)

from models import build_vae_var
from utils.fc2_recorder import FC2InputRecorder

MODEL_DEPTH = 16
TARGET_STAGE, TARGET_BLOCK = 9, MODEL_DEPTH - 1
OUT_DIR = VAR_ROOT / "fc2_records"
BATCH_SIZE, CLASS_LABELS = 2, (437, 437)
SEED, CFG, TOP_K, TOP_P = 0, 4, 900, 0.95
HIST_BINS = 120

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

vae_ckpt = VAR_ROOT / 'vae_ch160v4096z32.pth'
var_ckpt = VAR_ROOT / f'var_d{MODEL_DEPTH}.pth'
hf = 'https://huggingface.co/FoundationVision/var/resolve/main'
for ckpt in (vae_ckpt, var_ckpt):
    if not ckpt.is_file():
        os.system(f'wget -q {hf}/{ckpt.name} -O {ckpt}')

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
)
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()

rec = FC2InputRecorder(
    str(OUT_DIR), target_stage=TARGET_STAGE, target_block=TARGET_BLOCK, bins=HIST_BINS,
)
rec.attach(var)

label_B = torch.tensor(CLASS_LABELS[:BATCH_SIZE], device=device)
with torch.inference_mode():
    with torch.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
        var.autoregressive_infer_cfg(
            B=BATCH_SIZE, label_B=label_B, cfg=CFG,
            top_k=TOP_K, top_p=TOP_P, g_seed=SEED,
        )

rec.detach(var)
