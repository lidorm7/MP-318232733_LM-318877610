"""
Show baseline-EDSR ×4 vs. Cascade2× outputs on one DIV2K image.
AAuthors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity   as ssim
from models import Cascade2x, build_edsr

@torch.no_grad()
def _sr_forward(model, lr_tensor, device):
    expects_255 = hasattr(model, "url")
    inp = lr_tensor * 255.0 if expects_255 else lr_tensor
    with torch.autocast(device_type=device.type):
        out = (
            model(inp)[-1]
            if isinstance(model, Cascade2x)
            else model(inp)
        )
    if expects_255:
        out = out.clamp(0, 255) / 255.0
    return out.squeeze(0).cpu()


def show_pair(
        idx: int,
        baseline_ckpt: str,
        cascade_ckpt:  str,
        save_dir: str | None = None
):
    """
    Parameters
    ----------
    idx            : DIV2K image index (801-900)
    baseline_ckpt  : path to `edsr_baseline_x4*.pt`
    cascade_ckpt   : path to `cascade_best_seed*.pth`
    save_dir       : folder to save PNGs (None/'' → no saving)
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = T.ToTensor()
    to_pil    = T.ToPILImage()

    # load models
    baseline = build_edsr(scale=4, n_resblocks=16, n_feats=64).to(device).eval()
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device), strict=False)

    cascade  = Cascade2x().to(device).eval()
    cascade.load_state_dict(torch.load(cascade_ckpt,  map_location=device))

    # pick LR / HR paths
    lr_dir = "data/DIV2K_valid_LR_X4_blurred" if idx < 891 else "data/DIV2K_test_LR_X4_blurred"
    hr_dir = "data/DIV2K_val_HR"              if idx < 891 else "data/DIV2K_test_HR"
    fname  = f"{idx:04d}.png"

    lr_path, hr_path = Path(lr_dir)/fname, Path(hr_dir)/fname

    # 3. ── run inference ────────────────────────────────────
    lr_img   = Image.open(lr_path).convert("RGB")
    hr_img   = Image.open(hr_path).convert("RGB")
    lr_t     = to_tensor(lr_img).unsqueeze(0).to(device)

    sr_base  = _sr_forward(baseline, lr_t, device)
    sr_casc  = _sr_forward(cascade,  lr_t, device)

    # compute metrics
    hr_np    = to_tensor(hr_img).permute(1,2,0).numpy()
    base_np  = sr_base.permute(1,2,0).numpy()
    casc_np  = sr_casc.permute(1,2,0).numpy()

    def _psnr(a, b): return psnr(a, b, data_range=1.)
    def _ssim(a, b): return ssim(a, b, data_range=1., channel_axis=2)

    print(f"► Baseline  PSNR {_psnr(hr_np, base_np):.2f} dB  |  "
          f"SSIM {_ssim(hr_np, base_np):.4f}")
    print(f"► Cascade   PSNR {_psnr(hr_np, casc_np):.2f} dB  |  "
          f"SSIM {_ssim(hr_np, casc_np):.4f}")

    # visualise side-by-side
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(to_pil(sr_base)); plt.title("Baseline EDSR-×4"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(to_pil(sr_casc)); plt.title("Cascade 2×+2×");    plt.axis("off")
    plt.suptitle(f"DIV2K {idx:04d}")
    plt.tight_layout(); plt.show()

    # optional save
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        to_pil(sr_base).save(out/f"{idx:04d}_sr_baseline.png")
        to_pil(sr_casc).save(out/f"{idx:04d}_sr_cascade.png")
        print(f"✓ PNGs written to {out}/")



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Visualise one DIV2K frame: baseline-EDSR ×4 vs Cascade2×")
    p.add_argument("--idx",            type=int, required=True,
                   help="Image index 801-900 (valid+test split)")
    p.add_argument("--baseline-ckpt",  type=str, required=True,
                   help="Path to edsr_baseline_x4*.pt")
    p.add_argument("--cascade-ckpt",   type=str, required=True,
                   help="Path to cascade_best_seed*.pth")
    p.add_argument("--save-dir",       type=str, default="",
                   help="Optional folder to save the two SR PNGs")
    args = p.parse_args()
    show_pair(
        idx           = args.idx,
        baseline_ckpt = args.baseline_ckpt,
        cascade_ckpt  = args.cascade_ckpt,
        save_dir      = args.save_dir or None
    )
