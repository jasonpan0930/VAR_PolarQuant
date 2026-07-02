# block 1
################## 1. Download checkpoints and build models
import argparse
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
from utils.angle_quant import register_theta2_kmeans_codebook, register_theta2_kmeans_codebook_v
from utils.theta2_kmeans import load_theta2_codebook


def parse_args():
    p = argparse.ArgumentParser(description='VAR d30 demo sampling with optional polar KV quantization')
    p.add_argument('--model-depth', type=int, default=30, choices=(16, 20, 24, 30))
    p.add_argument('--polar-quant', type=str, default='none',
                   help='none/baseline/off or int6_kmeans_int4')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cfg', type=float, default=4.0)
    p.add_argument('--top-k', type=int, default=900)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--class-labels', type=int, nargs='+',
                   default=[980, 980, 437, 437, 22, 22, 562, 562])
    p.add_argument('--out-path', type=str, default=None)
    p.add_argument('--kmeans-codebook', type=str, default=None)
    p.add_argument('--quant-v', action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


args = parse_args()

MODEL_DEPTH = args.model_depth
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt): 
    print(f"downloading {vae_ckpt} from {hf_home}")
    os.system(f'wget {hf_home}/{vae_ckpt}')
else:
    print(f"{vae_ckpt} already exists")
if not osp.exists(var_ckpt): 
    print(f"downloading {var_ckpt} from {hf_home}")
    os.system(f'wget {hf_home}/{var_ckpt}')
else:
    print(f"{var_ckpt} already exists")

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')


def setup_polar_quant(var, polar_quant: str):
    name = polar_quant.lower().strip()
    if name in ('none', 'baseline', 'fp16', 'off'):
        var.set_polar_quant(None)
        return 'none'
    if name != 'int6_kmeans_int4':
        raise ValueError(f'unsupported --polar-quant {polar_quant!r}; use none or int6_kmeans_int4')

    cb_path = args.kmeans_codebook
    if cb_path is None:
        cb_path = osp.join('polar_quant_dumps', f'theta2_kmeans_d{MODEL_DEPTH}', 'codebook.json')
        if MODEL_DEPTH == 16 and not osp.exists(cb_path):
            cb_path = osp.join('polar_quant_dumps', 'theta2_kmeans', 'codebook.json')
    centers, _ = load_theta2_codebook(cb_path)
    register_theta2_kmeans_codebook(centers)

    v_cb_path = osp.join(osp.dirname(osp.dirname(cb_path)), f'{osp.basename(osp.dirname(cb_path))}_v', 'codebook.json')
    if args.quant_v and osp.exists(v_cb_path):
        v_centers, _ = load_theta2_codebook(v_cb_path)
        register_theta2_kmeans_codebook_v(v_centers)
        print(f'loaded V codebook: {v_cb_path}')

    var.set_polar_quant('int6_kmeans_int4', quant_v=args.quant_v)
    return 'int6_kmeans_int4'


# block 2
############################# 2. Sample with classifier-free guidance

# set args
seed = args.seed
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = args.cfg
top_k = args.top_k
top_p = args.top_p
class_labels = tuple(args.class_labels)
more_smooth = False # True for more smooth output
quant_tag = setup_polar_quant(var, args.polar_quant)
print(f'polar_quant={quant_tag} quant_v={args.quant_v} seed={seed} cfg={cfg} top_k={top_k} top_p={top_p} class_labels={class_labels}')

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# sample
B = len(class_labels)
label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
# print(f"label_B: {label_B}") # label_B: tensor([980, 980, 437, 437,  22,  22, 562, 562], device='cuda:0')
with torch.inference_mode():
    with torch.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=seed, more_smooth=more_smooth)

chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
chw = PImage.fromarray(chw.astype(np.uint8))


out_path = args.out_path or f"sample_d{MODEL_DEPTH}_{quant_tag}.png"
chw.save(out_path)
print(f"saved to {out_path}")
