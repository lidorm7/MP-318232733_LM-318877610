"""
Super-resolution models for the DIV2K project
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
from types import SimpleNamespace
from typing  import Tuple
import sys
import torch
import torch.nn as nn
sys.path.append("EDSR-PyTorch/src")  # EDSR repo
from model import edsr


# Mean-shift (pixel-wise normalization / denormalization)
class MeanShift(nn.Conv2d):
    """Add or subtract a fixed RGB mean (1×1 convolution)."""
    def __init__(
        self,
        rgb_range: int,
        rgb_mean : Tuple[float, float, float],
        rgb_std  : Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sign: int = -1,
    ):
        super().__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


# EDSR helper (×2 by default)
def build_edsr(
    scale: int      = 2,
    n_resblocks: int = 16,
    n_feats: int     = 64,
    res_scale: float = 0.1,
    rgb_range: int   = 1,
):
    """Return an EDSR model for arbitrary scale (×2 default)."""
    args = SimpleNamespace(
        n_resblocks=n_resblocks,
        n_feats=n_feats,
        res_scale=res_scale,
        scale=[scale],
        rgb_range=rgb_range,
        n_colors=3,
        no_upsampling=False,
    )
    return edsr.EDSR(args)


# Cascade2× (ours) – two successive ×2 EDSR stages = overall ×4
DIV2K_RGB_MEAN = (0.4488, 0.4371, 0.4040)  # paper values, 0-1 range
RGB_STD        = (1.0,    1.0,    1.0)     # keep unity std

class Cascade2x(nn.Module):
    """
    Progressive ×4 super-resolution:
        LR×4 → EDSR-A (×2) → LR×2
             → EDSR-B (×2) → HR
    Intermediate ×2 output is returned for auxiliary loss.
    """
    def __init__(
        self,
        n_resblocks: int = 16,
        n_feats: int     = 64,
        res_scale: float = 0.1,
    ):
        super().__init__()
        # mean-shift layers
        self.sub_mean = MeanShift(1, DIV2K_RGB_MEAN, RGB_STD, sign=-1)
        self.add_mean = MeanShift(1, DIV2K_RGB_MEAN, RGB_STD, sign=+1)
        # two ×2 stages
        self.edsr_a = build_edsr(scale=2, n_resblocks=n_resblocks,
                                 n_feats=n_feats, res_scale=res_scale)
        self.edsr_b = build_edsr(scale=2, n_resblocks=n_resblocks,
                                 n_feats=n_feats, res_scale=res_scale)

    def forward(self, lr4: torch.Tensor):
        x = self.sub_mean(lr4)          # subtract dataset mean
        lr2_pred = self.edsr_a(x)
        hr_pred  = self.edsr_b(lr2_pred)
        hr_pred  = self.add_mean(hr_pred)
        return lr2_pred, hr_pred
