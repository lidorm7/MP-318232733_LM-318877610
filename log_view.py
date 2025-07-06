"""
Plot L1-loss & PSNR curves saved by train.py
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt

# ↳ accept paths on the CLI:  python log_view.py checkpoints/seed*/log_seed*.npy
log_files = sys.argv[1:] or [
    "checkpoints/seed0/log_seed0.npy",
    "checkpoints/seed1/log_seed1.npy",
    "checkpoints/seed2/log_seed2.npy",
]

for path in sorted(log_files):
    log = np.load(path, allow_pickle=True).item()
    seed = os.path.splitext(os.path.basename(path))[0].split("seed")[-1]

    ep = range(1, len(log["tr_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    # ── L1 subplot ───────────────────────────────────────────────────────
    plt.subplot(1, 2, 1)
    plt.plot(ep, log["tr_loss"], label="train")
    plt.plot(ep, log["vl_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("avg L1")
    plt.title(f"Seed {seed} • Loss"); plt.legend()

    # ── PSNR subplot ────────────────────────────────────────────────────
    plt.subplot(1, 2, 2)
    plt.plot(ep, log["tr_psnr"], label="train")
    plt.plot(ep, log["vl_psnr"], label="val")
    plt.xlabel("epoch"); plt.ylabel("dB")
    plt.title(f"Seed {seed} • PSNR"); plt.legend()

    plt.tight_layout()
    plt.show()
