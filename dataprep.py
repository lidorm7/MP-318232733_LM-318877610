"""
DIV2K dataset downsampler
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
"""
import os, shutil
from PIL import Image, ImageFilter

def data_preparation():
    hr_dir='data/DIV2K_train_HR/DIV2K_train_HR'
    lr_2_dir = 'data/DIV2K_train_LR_X2_blurred'
    lr_4_dir = 'data/DIV2K_train_LR_X4_blurred'
    val_dir = 'data/DIV2K_valid_HR/DIV2K_valid_HR'
    test_hr_dir = 'data/DIV2K_test_HR'
    val_hr_dir = 'data/DIV2K_val_HR'
    val_lr_4_dir = 'data/DIV2K_valid_LR_X4_blurred'
    test_lr_4_dir = 'data/DIV2K_test_LR_X4_blurred'

    os.makedirs(lr_2_dir, exist_ok=True)
    os.makedirs(lr_4_dir, exist_ok=True)
    os.makedirs(val_hr_dir, exist_ok=True)
    os.makedirs(test_hr_dir, exist_ok=True)
    os.makedirs(val_lr_4_dir, exist_ok=True)
    os.makedirs(test_lr_4_dir, exist_ok=True)

    # Sort and process each file in the original validation HR set
    for filename in sorted(os.listdir(val_dir)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_id = int(os.path.splitext(filename)[0])
            src_path = os.path.join(val_dir, filename)
        if 801 <= img_id <= 890:
            shutil.copy2(src_path, os.path.join(val_hr_dir, filename))
        elif 891 <= img_id <= 900:
            shutil.copy2(src_path, os.path.join(test_hr_dir, filename))


    def downsample_with_blur(hr_dir, lr_dir, scale=4, blur_radius=1.0):
        for fname in os.listdir(hr_dir):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                hr_path = os.path.join(hr_dir, fname)
                img = Image.open(hr_path)

                # Apply Gaussian Blur
                blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

                # Downsample
                w, h = blurred.size
                lr_img = blurred.resize((w // scale, h // scale), Image.BICUBIC)

                # Save LR image
                lr_img.save(os.path.join(lr_dir, fname))

        print(f"Saved downsampled blurred images to: {lr_dir}")


    downsample_with_blur(hr_dir, lr_2_dir, scale=2, blur_radius=1.0)
    downsample_with_blur(hr_dir, lr_4_dir, scale=4, blur_radius=1.0)
    downsample_with_blur(val_hr_dir, val_lr_4_dir, scale=4, blur_radius=1.0)
    downsample_with_blur(test_hr_dir, test_lr_4_dir, scale=4, blur_radius=1.0)

data_preparation()
