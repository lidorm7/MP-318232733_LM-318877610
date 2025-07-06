"""
Train Cascade2× on DIV2K
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from datasets import DIV2KTriplet, DIV2KFull
from models import Cascade2x, DIV2K_RGB_MEAN
import random

class CascadeTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(cfg.seed)

        # Datasets
        self.train_set = DIV2KTriplet(cfg.train_lr4, cfg.train_lr2, cfg.train_hr,
                                      patch=cfg.patch, train=True, augment=True, random_crop=True)
        self.val_set = DIV2KFull(cfg.val_lr4, cfg.val_hr)

        self.train_loader = DataLoader(self.train_set, batch_size=cfg.batch,
                                       shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_set, batch_size=1,
                                     shuffle=False, num_workers=4, pin_memory=True)

        # Model and Optimization
        self.model = Cascade2x().to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.scaler = torch.amp.GradScaler('cuda')

        self.ssim_metric = SSIM(data_range=1.).to(self.device)
        self.div2k_mean = torch.tensor(DIV2K_RGB_MEAN).view(1, 3, 1, 1).to(self.device)

        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_psnr = -1.0

        # Logs
        self.tr_loss, self.vl_loss = [], []
        self.tr_psnr, self.vl_psnr = [], []

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _psnr(self, a, b):
        mse = F.mse_loss(a, b)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            self._train_epoch()
            self._validate_epoch()

            # Checkpoint
            if self.vl_psnr[-1] > self.best_psnr:
                self.best_psnr = self.vl_psnr[-1]
                torch.save(self.model.state_dict(), self.save_dir / self.cfg.checkpoint_name)

        np.save(self.save_dir / f"log_seed{self.cfg.seed}.npy",
                dict(tr_loss=self.tr_loss, vl_loss=self.vl_loss,
                     tr_psnr=self.tr_psnr, vl_psnr=self.vl_psnr))

    def _train_epoch(self):
        self.model.train()
        total_loss, total_psnr = 0., 0.

        for lr4, lr2, hr in self.train_loader:
            lr4, lr2, hr = [x.to(self.device) for x in (lr4, lr2, hr)]

            with torch.amp.autocast('cuda'):
                mid, sr = self.model(lr4)
                loss = (self.cfg.lambda_mid * self.criterion(mid + self.div2k_mean, lr2)
                        + self.criterion(sr, hr)
                        + self.cfg.lambda_ssim * (1 - self.ssim_metric(sr, hr)))

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * lr4.size(0)
            total_psnr += self._psnr(sr, hr).item() * lr4.size(0)

        self.scheduler.step()
        N = len(self.train_loader.dataset)
        self.tr_loss.append(total_loss / N)
        self.tr_psnr.append(total_psnr / N)

    def _validate_epoch(self):
        self.model.eval()
        total_l1, total_psnr = 0., 0.

        with torch.no_grad():
            for lr4, hr in self.val_loader:
                lr4, hr = lr4.to(self.device), hr.to(self.device)
                with torch.amp.autocast('cuda'):
                    _, sr = self.model(lr4)
                total_l1 += F.l1_loss(sr, hr, reduction="sum").item()
                total_psnr += self._psnr(sr, hr).item()

        N = len(self.val_loader.dataset)
        self.vl_loss.append(total_l1 / (N * hr.numel() / hr.shape[0]))
        self.vl_psnr.append(total_psnr / N)
