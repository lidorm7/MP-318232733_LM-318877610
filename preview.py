"""
DIV2K augmentation preview
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets import DIV2KTriplet

def preview_patches(*,
                    train=True,
                    augment=True,
                    random_crop=True,
                    seed=0,
                    n_show=4):
    ds = DIV2KTriplet(
        'data/DIV2K_train_LR_X4_blurred',
        'data/DIV2K_train_LR_X2_blurred',
        'data/DIV2K_train_HR/DIV2K_train_HR',
        patch=48,
        train=train,
        augment=augment,
        random_crop=random_crop
    )

    torch.manual_seed(seed)  # reproducible crop / aug
    lr4, _, _ = next(iter(DataLoader(ds, batch_size=n_show, num_workers=0)))

    # upscale to 192×192 for easy viewing
    view = lr4.shape[-1] * 4
    lr4_big = TF.resize(lr4, [view, view], TF.InterpolationMode.BICUBIC)

    # plot
    fig, ax = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    label = ("Rand-crop" if random_crop else "Centre") + \
            (" • aug" if augment else " • no-aug")
    for i in range(n_show):
        ax[i].imshow(lr4_big[i].permute(1, 2, 0).clamp(0, 1))
        ax[i].axis("off")
        ax[i].set_title(label, fontsize=10)
    plt.tight_layout()
    plt.show()


# EXAMPLES ───────────────────────────────────────────────
preview_patches(augment=True, random_crop=True, seed=1)  # random + aug
preview_patches(augment=False, random_crop=True, seed=1)  # random + no-aug
preview_patches(augment=True, random_crop=False, seed=1)  # centre + aug
preview_patches(augment=False, random_crop=False, seed=1)  # centre + no-aug