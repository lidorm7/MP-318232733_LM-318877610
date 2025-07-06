"""
Main run and test
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
from argparse import Namespace
from train import CascadeTrainer
from test import test
import numpy as np
import os
import glob

def aggregate():
    LOG_DIR = "checkpoints"
    pattern = os.path.join(LOG_DIR, "seed*/log_seed*.npy")

    psnr, ssim, fid = [], [], []

    # Load metrics from each seed's log
    for npy_path in sorted(glob.glob(pattern)):
        d = np.load(npy_path, allow_pickle=True).item()  # dict with keys: psnr, ssim, fid
        psnr.append(d['test_psnr'])
        ssim.append(d['test_ssim'])
        fid .append(d['test_fid'])
        print(f"{os.path.basename(npy_path):25s} ➜  "
              f"PSNR {d['test_psnr']:.2f}  SSIM {d['test_ssim']:.4f}  FID {d['test_fid']:.2f}")

    # Convert to numpy arrays for aggregation
    psnr, ssim, fid = map(np.asarray, (psnr, ssim, fid))

    # Print aggregated results
    print("\n──────── Aggregated over seeds ────────")
    print(f"PSNR  mean: {psnr.mean():.2f} ± {psnr.std():.2f} dB")
    print(f"SSIM  mean: {ssim.mean():.4f} ± {ssim.std():.4f}")
    print(f"FID   mean: {fid.mean():.2f} ± {fid.std():.2f}")


def run(seed):
    # Define training configuration
    cfg = Namespace(
        train_hr='data/DIV2K_train_HR/DIV2K_train_HR',
        train_lr4='data/DIV2K_train_LR_X4_blurred',
        train_lr2='data/DIV2K_train_LR_X2_blurred',
        val_hr='data/DIV2K_val_HR',
        val_lr4='data/DIV2K_valid_LR_X4_blurred',
        epochs=150, batch=8, patch=48, lr=1e-4,
        lambda_mid=0.3, lambda_ssim=0.1,
        seed=seed,
        save_dir=f"checkpoints/seed{seed}",
        checkpoint_name=f"cascade_best_seed{seed}.pth"
    )

    # Train the model
    trainer = CascadeTrainer(cfg)
    trainer.train()
    metrics = test(model_path=f"{cfg.save_dir}/{cfg.checkpoint_name}")

    # Append metrics to the training log file
    log_path = f"{cfg.save_dir}/log_seed{seed}.npy"
    if os.path.isfile(log_path):
        log_data = np.load(log_path, allow_pickle=True).item()
    else:
        log_data = {}
    log_data.update({f"test_{k}": v for k, v in metrics.items()})
    np.save(log_path, log_data)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="Space-separated list of seeds to train (default: 0 1 2)")
    args = p.parse_args()
    for s in args.seeds:
        run(s)
    aggregate()        # after all seeds are done
