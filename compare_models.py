"""
Compare Baseline-EDSR ×4 vs. Cascade2×
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
import os
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import torch
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as psnr
from models import Cascade2x, build_edsr     # ← your modules


@torch.no_grad()
def _sr(model, lr_path, device, to_tensor):
    img = Image.open(lr_path).convert("RGB")
    t   = to_tensor(img).unsqueeze(0).to(device)
    needs_255 = hasattr(model, "url")
    if needs_255:
        t *= 255.
    with torch.autocast(device_type=device.type):
        out = model(t)[-1] if isinstance(model, Cascade2x) else model(t)
    if needs_255:
        out = out.clamp(0, 255.) / 255.
    return out.squeeze(0).permute(1, 2, 0).cpu().numpy()


def compare(cfg):
    device, to_tensor = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        T.ToTensor()
    )

    # load models
    baseline = build_edsr(scale=4, n_resblocks=16, n_feats=64).to(device).eval()
    baseline.load_state_dict(torch.load(cfg.baseline_ckpt, map_location=device),
                             strict=False)

    cascade = Cascade2x().to(device).eval()
    cascade.load_state_dict(torch.load(cfg.cascade_ckpt, map_location=device))

    # evaluation loop
    pairs = [
        ("data/DIV2K_valid_LR_X4_blurred", "data/DIV2K_val_HR"),
        ("data/DIV2K_test_LR_X4_blurred",  "data/DIV2K_test_HR")
    ]
    results = []

    for lr_root, hr_root in pairs:
        for fname in sorted(os.listdir(hr_root)):
            lr_p, hr_p = Path(lr_root)/fname, Path(hr_root)/fname
            hr_np = to_tensor(Image.open(hr_p)).permute(1,2,0).numpy()

            sr_base = _sr(baseline, lr_p, device, to_tensor)
            sr_casc = _sr(cascade , lr_p, device, to_tensor)

            p_base = psnr(hr_np, sr_base, data_range=1.)
            p_casc = psnr(hr_np, sr_casc, data_range=1.)

            # keep cases where the requested model wins
            if (cfg.winner == "baseline" and p_base > p_casc) or \
               (cfg.winner == "cascade"  and p_casc > p_base):

                results.append({
                    "fname": fname,
                    "psnr_baseline": p_base,
                    "psnr_cascade" : p_casc,
                    "delta": abs(p_base - p_casc),
                    "sr": sr_base if cfg.winner == "baseline" else sr_casc
                })

    # keep top-k and save SR images
    results.sort(key=lambda d: d["delta"], reverse=True)
    top = results[: cfg.top_k]

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for rank, info in enumerate(top, 1):
        Image.fromarray((info["sr"]*255).round().astype("uint8")).save(
            out_dir / f"{cfg.winner.upper()}_better_{rank:02d}_{info['fname']}"
        )

    # 4. console summary
    print(f"\n{len(top)} example where {cfg.winner} beats the other model:")
    for info in top:
        print(f"  {info['fname']}  ΔPSNR = {info['delta']:.2f} dB "
              f"(base {info['psnr_baseline']:.2f}  |  cas {info['psnr_cascade']:.2f})")
    print(f"Saved {len(top)} SR image(s) to  {out_dir}/")


def compare_paths(baseline_ckpt: str,
                  cascade_ckpt : str,
                  out_dir      : str = "comparisons",
                  winner       : str = "baseline",
                  top_k        : int = 1):
    """
    Evaluate two checkpoints and keep the *top_k* images where `winner`
    ('baseline' or 'cascade') achieves higher PSNR.
    """
    winner = winner.lower()
    assert winner in ("baseline", "cascade")
    compare(SimpleNamespace(
        baseline_ckpt = baseline_ckpt,
        cascade_ckpt  = cascade_ckpt,
        out_dir       = f"{out_dir}/{winner}_better",
        winner        = winner,
        top_k         = top_k
    ))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument("--baseline-ckpt", required=True,
                        help="Path to baseline EDSR-x4 .pt file")
    parser.add_argument("--cascade-ckpt",  required=True,
                        help="Path to Cascade2× checkpoint (.pth)")
    parser.add_argument("--out-dir",       default="comparisons",
                        help="Root folder where results will be written")
    parser.add_argument("--winner",        choices=["baseline", "cascade"],
                        default="baseline",
                        help="Which model should be treated as the winner "
                             "(i.e. images kept when it has higher PSNR)")
    parser.add_argument("--top-k",         type=int, default=1,
                        help="How many best-gap images to save")

    args = parser.parse_args()
    compare_paths(
        baseline_ckpt=args.baseline_ckpt,
        cascade_ckpt=args.cascade_ckpt,
        out_dir=args.out_dir,
        winner=args.winner,
        top_k=args.top_k
    )
