"""
DIV2K dataset helpers
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DIV2KTriplet(Dataset):
    """(LR×4, LR×2, HR) triplets with optional aug + random crop."""
    def __init__(self, lr4_dir, lr2_dir, hr_dir,
                 patch=48, train=True, augment=True, random_crop=True):
        self.lr4_dir, self.lr2_dir, self.hr_dir = map(Path, (lr4_dir, lr2_dir, hr_dir))
        self.files = sorted(self.hr_dir.glob("*.png"))
        self.patch, self.train = patch, train
        self.augment, self.random_crop = augment, random_crop
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def _aligned_crop(self, lr4, lr2, hr):
        w4, h4 = lr4.size
        if self.train and self.random_crop:
            x4 = torch.randint(0, w4 - self.patch + 1, ()).item()
            y4 = torch.randint(0, h4 - self.patch + 1, ()).item()
        else:
            x4 = (w4 - self.patch) // 2
            y4 = (h4 - self.patch) // 2
        lr4 = lr4.crop((x4, y4, x4+self.patch, y4+self.patch))
        lr2 = lr2.crop((x4*2, y4*2, x4*2+self.patch*2, y4*2+self.patch*2))
        hr  = hr .crop((x4*4, y4*4, x4*4+self.patch*4, y4*4+self.patch*4))
        return lr4, lr2, hr

    def __getitem__(self, idx):
        fname = self.files[idx].name
        lr4 = Image.open(self.lr4_dir / fname).convert("RGB")
        lr2 = Image.open(self.lr2_dir / fname).convert("RGB")
        hr  = Image.open(self.hr_dir  / fname).convert("RGB")
        lr4, lr2, hr = self._aligned_crop(lr4, lr2, hr)

        # optional flips / rotations
        if self.train and self.augment:
            if torch.rand(1) < 0.5:
                lr4, lr2, hr = (im.transpose(Image.FLIP_LEFT_RIGHT) for im in (lr4, lr2, hr))
            k = torch.randint(0, 4, ()).item()
            if k:
                lr4, lr2, hr = (im.rotate(90*k, expand=True) for im in (lr4, lr2, hr))

        return self.to_tensor(lr4), self.to_tensor(lr2), self.to_tensor(hr)

class DIV2KFull(Dataset):
    """Full-resolution LR×4 → HR pairs, no aug."""
    def __init__(self, lr4_dir, hr_dir):
        self.lr4_dir, self.hr_dir = map(Path, (lr4_dir, hr_dir))
        self.files = sorted(self.hr_dir.glob("*.png"))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx].name
        lr = Image.open(self.lr4_dir / fname).convert("RGB")
        hr = Image.open(self.hr_dir  / fname).convert("RGB")
        return self.to_tensor(lr), self.to_tensor(hr)
