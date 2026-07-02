#!/usr/bin/env python3
"""Sanity check: run baseline twice with same seed, compare f_hat cosine distance."""
import math, random, sys
sys.path.insert(0, '.')
import numpy as np
import torch
import models.var as var_mod
from models import build_vae_var

MODEL_DEPTH = 30; SEED = 0; CFG = 4; TOP_K = 900; TOP_P = 0.95; BATCH_SIZE = 1
LABELS = [45, 135, 246, 346, 473, 574, 683, 734, 834, 985]

def set_seeds():
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    patch_nums = (1,2,3,4,5,6,8,10,13,16)
    vae, var = build_vae_var(V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums, num_classes=1000, depth=MODEL_DEPTH, shared_aln=False)
    vae.load_state_dict(torch.load('vae_ch160v4096z32.pth', map_location='cpu'), strict=True)
    var.load_state_dict(torch.load('var_d30.pth', map_location='cpu'), strict=True)
    vae.eval(); var.eval()
    num_stages = len(var.patch_nums)
    print(f'model loaded, num_stages={num_stages}')

    fhats_run1 = {}
    fhats_run2 = {}

    for label in LABELS:
        label_B = torch.tensor([label], device=device)
        for run_idx, store in enumerate([fhats_run1, fhats_run2]):
            f_hat_list = []
            vae_proxy = var.vae_quant_proxy[0]
            orig_fn = vae_proxy.get_next_autoregressive_input
            def patch(si, SN, f_hat, h_BChw):
                new_f_hat, ntm = orig_fn(si, SN, f_hat, h_BChw)
                f_hat_list.append(new_f_hat.detach().float().cpu().clone())
                return new_f_hat, ntm
            vae_proxy.get_next_autoregressive_input = patch
            try:
                set_seeds()
                var.set_polar_quant(None)
                with torch.inference_mode():
                    with torch.autocast('cuda', enabled=device=='cuda', dtype=torch.float16):
                        var.autoregressive_infer_cfg(B=BATCH_SIZE, label_B=label_B,
                            cfg=CFG, top_k=TOP_K, top_p=TOP_P, g_seed=SEED)
            finally:
                vae_proxy.get_next_autoregressive_input = orig_fn
            if len(f_hat_list) != num_stages:
                raise RuntimeError(f'expected {num_stages} stages, got {len(f_hat_list)}')
            store[label] = f_hat_list
        print(f'[label={label:>3d}] ok (run1 + run2)')

    # Compare f_hat per scale
    print()
    n_labels = len(LABELS)
    headings = ['scale'] + [f'l={l}' for l in LABELS] + ['mean']
    print('  '.join(f'{h:>12s}' for h in headings))
    all_cos_vals = None
    for si in range(num_stages):
        cos_vals = []
        for label in LABELS:
            bf = fhats_run1[label][si]
            qf = fhats_run2[label][si]
            dot = float((bf * qf).sum().item())
            rn = float(bf.pow(2).sum().item())
            cn = float(qf.pow(2).sum().item())
            denom = math.sqrt(max(rn, 0.0) * max(cn, 0.0))
            cos_sim = max(-1.0, min(1.0, dot / max(denom, 1e-12)))
            cos_dist = 1.0 - cos_sim
            cos_vals.append(cos_dist)
        mean_cos = float(np.mean(cos_vals))
        row = [f's{si}'] + [f'{v:.2e}' for v in cos_vals] + [f'{mean_cos:.2e}']
        print('  '.join(f'{x:>12s}' for x in row))
        if all_cos_vals is None:
            all_cos_vals = np.array(cos_vals)
        else:
            all_cos_vals = np.concatenate([all_cos_vals, cos_vals])

    print(f'\nOverall mean cos dist: {float(all_cos_vals.mean()):.6e}')
    print(f'Overall max  cos dist: {float(all_cos_vals.max()):.6e}')
    print(f'Expected: 0.0 (identical runs with same seed)')

if __name__ == '__main__':
    main()
