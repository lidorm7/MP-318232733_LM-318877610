"""
Baseline EDSR
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
import os, sys, pathlib
import numpy as np
import torch
from types import SimpleNamespace
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity   as ssim

# ───────────────────────── CONFIG ───────────────────────── #
EDSR_REPO   = "edsr-pytorch/src"                    # your local clone
CKPT_PATH    = "edsr_baseline_x4.pt"                 # downloaded weight file
VAL_LR_DIR   = "data/DIV2K_valid_LR_X4_blurred"
VAL_HR_DIR   = "data/DIV2K_val_HR"
TEST_LR_DIR  = "data/DIV2K_test_LR_X4_blurred"
TEST_HR_DIR  = "data/DIV2K_test_HR"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────── #

# import EDSR implementation
sys.path.append(EDSR_REPO)
from model import edsr

args = SimpleNamespace(
    n_resblocks = 16,
    n_feats     = 64,
    res_scale   = 0.1,
    scale       = [4],
    rgb_range   = 255,
    n_colors    = 3,
    no_upsampling = False,
)
baseline = edsr.EDSR(args).to(DEVICE).eval()
state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
baseline.load_state_dict(state_dict, strict=False)
print("EDSR-baseline-×4 weights loaded")

to_tensor = T.ToTensor()
fid       = FID(feature=2048, normalize=True).cpu()

def run_sr(lr_img: Image.Image) -> torch.Tensor:
    lr_t = to_tensor(lr_img).unsqueeze(0).to(DEVICE) * 255.0
    with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type=="cuda"):
        sr = baseline(lr_t).clamp(0,255) / 255.0
    return sr.squeeze(0).cpu()                       # (3,H*,W*)

def evaluate_baseline():
    idxs = [*range(801,891), *range(891,901)]
    examples: list[dict] = []
    psnr_vals, ssim_vals = [], []

    for idx in idxs:
        fname = f"{idx:04d}.png"
        lr_dir, hr_dir = (VAL_LR_DIR, VAL_HR_DIR) if idx < 891 else (TEST_LR_DIR, TEST_HR_DIR)

        lr_img = Image.open(os.path.join(lr_dir, fname)).convert("RGB")
        hr_img = Image.open(os.path.join(hr_dir, fname)).convert("RGB")

        sr_t = run_sr(lr_img)                               # torch 0-1
        sr_np = sr_t.detach().permute(1, 2, 0).numpy()
        hr_np = to_tensor(hr_img).permute(1,2,0).numpy()

        p = psnr(hr_np, sr_np, data_range=1.0)
        s = ssim(hr_np, sr_np, data_range=1.0, channel_axis=2)

        psnr_vals.append(p);  ssim_vals.append(s)
        fid.update(torch.from_numpy(hr_np).permute(2,0,1)[None], real=True)
        fid.update(torch.from_numpy(sr_np).permute(2,0,1)[None], real=False)

        examples.append({
            "fname": fname,
            "psnr" : p,
            "ssim" : s,
            "sr"   : Image.fromarray((sr_np*255).round().astype("uint8")),
            "hr"   : hr_img
        })

    # ── summary ────────────────────────────────────────────────────────
    print("\n────────  baseline EDSR-×4  ────────")
    print(f"Average PSNR : {np.mean(psnr_vals):.2f} dB")
    print(f"Average SSIM : {np.mean(ssim_vals):.4f}")
    print(f"FID          : {fid.compute().item():.2f}")

    # ── qualitative grid & PNG export ─────────────────────────────────
    examples.sort(key=lambda d: d["psnr"])
    rows   = examples[:2] + examples[-2:][::-1]
    titles = ["Worst-1", "Worst-2", "Best-1", "Best-2"]

    out_dir = "baseline_examples_SR_only"
    pathlib.Path(out_dir).mkdir(exist_ok=True)

    for tag, ex in zip(["worst1","worst2","best1","best2"], rows):
        ex["sr"].save(f"{out_dir}/{tag}_{ex['fname']}_SR.png", format="PNG")
        print("saved →", f"{out_dir}/{tag}_{ex['fname']}_SR.png")

    plt.figure(figsize=(8,10))
    for r, ex in enumerate(rows):
        for c, key in enumerate(["sr","hr"]):
            ax = plt.subplot(4,2,r*2+c+1)
            ax.imshow(ex[key]); ax.axis("off")
            if c == 0:
                ax.set_ylabel(titles[r], rotation=0, labelpad=35, va="center", fontsize=12)
                ax.set_title(f"{ex['fname']} | PSNR {ex['psnr']:.2f} | "
                             f"SSIM {ex['ssim']:.4f}", fontsize=9, pad=6)
    plt.suptitle("Baseline EDSR-×4 – qualitative examples", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.96])
    plt.show()


evaluate_baseline()
