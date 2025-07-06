"""
Cascade image worst/best
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models import Cascade2x

to_tensor = T.ToTensor()


@torch.no_grad()
def qualitative_examples(model_path, seed=0, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = Cascade2x().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Data roots
    val_lr_dir = 'data/DIV2K_valid_LR_X4_blurred'
    val_hr_dir = 'data/DIV2K_val_HR'
    test_lr_dir = 'data/DIV2K_test_LR_X4_blurred'
    test_hr_dir = 'data/DIV2K_test_HR'

    # Image indices
    idxs = list(range(801, 891)) + list(range(891, 901))

    fid = FID(feature=2048, normalize=True).cpu()
    psnr_vals, ssim_vals, examples = [], [], []

    def run_sr(lr_path):
        lr_img = Image.open(lr_path).convert('RGB')
        lr_t = to_tensor(lr_img).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            _, sr_t = model(lr_t)
        return lr_img, sr_t.squeeze(0).cpu()

    for idx in idxs:
        fname = f"{idx:04d}.png"
        lr_dir = val_lr_dir if idx < 891 else test_lr_dir
        hr_dir = val_hr_dir if idx < 891 else test_hr_dir

        _, sr_t = run_sr(os.path.join(lr_dir, fname))
        hr_img = Image.open(os.path.join(hr_dir, fname)).convert('RGB')
        sr_np = sr_t.permute(1, 2, 0).numpy()
        hr_np = to_tensor(hr_img).permute(1, 2, 0).numpy()

        p = psnr(hr_np, sr_np, data_range=1.)
        s = ssim(hr_np, sr_np, data_range=1., channel_axis=2)

        psnr_vals.append(p);
        ssim_vals.append(s)
        fid.update(torch.from_numpy(hr_np).permute(2, 0, 1)[None], real=True)
        fid.update(torch.from_numpy(sr_np).permute(2, 0, 1)[None], real=False)

        examples.append({
            "fname": fname,
            "psnr": p,
            "ssim": s,
            "sr": Image.fromarray((sr_np * 255).round().astype('uint8')),
            "hr": hr_img
        })

    print(f"\nCascade2× (seed-{seed}) — full-image results")
    print(f"Average PSNR : {np.mean(psnr_vals):.2f} dB")
    print(f"Average SSIM : {np.mean(ssim_vals):.4f}")
    print(f"FID          : {fid.compute().item():.2f}")

    # Save worst and best 2 SR images
    examples.sort(key=lambda d: d["psnr"])
    rows = examples[:2] + examples[-2:][::-1]
    titles = ["Worst-1", "Worst-2", "Best-1", "Best-2"]

    out_dir = f"qual_examples_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)

    for tag, ex in zip(["worst1", "worst2", "best1", "best2"], rows):
        save_path = os.path.join(out_dir, f"{tag}_{ex['fname']}_SR.png")
        ex["sr"].save(save_path, format="PNG")
        print("saved →", save_path)

    # Plot
    plt.figure(figsize=(8, 10))
    for r, ex in enumerate(rows):
        for c, key in enumerate(["sr", "hr"]):
            ax = plt.subplot(4, 2, r * 2 + c + 1)
            ax.imshow(ex[key])
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(titles[r], rotation=0, labelpad=35, va='center', fontsize=12)
                ax.set_title(f"{ex['fname']}  |  PSNR {ex['psnr']:.2f}  SSIM {ex['ssim']:.4f}",
                             fontsize=9, pad=6)
    plt.suptitle(f"Cascade2× (seed-{seed}) – qualitative examples", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Show worst-2 / best-2 Cascade2× examples + metrics")
    p.add_argument("--model-path", required=True, type=str,
                   help="Path to *.pth checkpoint to evaluate")
    p.add_argument("--seed", type=int, default=0,
                   help="Label used only for the figure / folder name")
    p.add_argument("--device", type=str, default="cuda",
                   help="'cuda' or 'cpu' (default: cuda if available)")
    args = p.parse_args()

    qualitative_examples(
        model_path=args.model_path,
        seed=args.seed,
        device=args.device
    )
