"""
Test model
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from models import Cascade2x

@torch.no_grad()
def test(model_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = Cascade2x().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = T.ToTensor()
    fid = FID(feature=2048, normalize=True).cpu()

    psnr_list, ssim_list = [], []

    # Relative paths
    val_dirs = [
        ('data/DIV2K_valid_LR_X4_blurred', 'data/DIV2K_val_HR'),
        ('data/DIV2K_test_LR_X4_blurred',  'data/DIV2K_test_HR')
    ]

    for root_lr, root_hr in val_dirs:
        for fname in sorted(os.listdir(root_hr)):
            hr_path = os.path.join(root_hr, fname)
            lr_path = os.path.join(root_lr, fname)

            if not os.path.exists(lr_path):
                print(f"Skipping missing LR file: {lr_path}")
                continue

            lr = transform(Image.open(lr_path)).unsqueeze(0).to(device)
            hr = transform(Image.open(hr_path)).unsqueeze(0)

            with torch.amp.autocast(device_type=device.type):
                _, sr = model(lr)

            sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
            hr_np = hr.squeeze(0).permute(1, 2, 0).numpy()

            psnr_list.append(psnr(hr_np, sr_np, data_range=1.))
            ssim_list.append(ssim(hr_np, sr_np, data_range=1., channel_axis=2))

            fid.update(torch.from_numpy(hr_np).permute(2, 0, 1)[None], real=True)
            fid.update(torch.from_numpy(sr_np).permute(2, 0, 1)[None], real=False)
    return {
        'psnr': np.mean(psnr_list),
        'ssim': np.mean(ssim_list),
        'fid': fid.compute().item()
    }
