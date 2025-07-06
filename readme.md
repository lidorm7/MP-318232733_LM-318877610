# Cascade2× vs. Baseline EDSR — project guide
Authors: Maxim Piscklich (318232733), Lidor Mizrahi (318877610)
## Overview

This repository reproduces the **Cascade2×** progressive super‑resolution model, compares it with the classic **EDSR‑baseline ×4**, and supplies compact utilities for training, evaluation, qualitative inspection and metric aggregation on the DIV2K dataset.

---

## Prerequisites

| # | Requirement                                                 | Reason                                                    | One‑liner to get it                                                                                  |
| - | ----------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1 | **DIV2K dataset** (HR PNGs for train / valid / test)        | Source images used for training & evaluation              | `kaggle datasets download -d joe1995/div2k-dataset -p data --unzip`                                  |
| 2 | **EDSR‑PyTorch repo**                                       | Baseline model code & building blocks reused by Cascade2× | `git clone https://github.com/sanghyun-son/EDSR-PyTorch.git`                                         |
| 3 | **EDSR baseline ×4 weights** `edsr_baseline_x4-6b446fab.pt` | Official checkpoint for a fair comparison                 | `wget https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt -O edsr_baseline_x4.pt` |

> After downloading DIV2K your tree should initially look like
>
> ```text
> data/
> ├── DIV2K_train_HR/   (800 PNGs)
> └── DIV2K_valid_HR/   (100 PNGs 801‑900)
> ```
>
> The low‑resolution folders are generated in **Step 1** below.
### Python packages

Install all runtime dependencies in one go:

```bash
pip install torch torchvision torchaudio numpy matplotlib pillow scikit-image torchmetrics kaggle tqdm
pip install torch-fidelity
```

>To run with GPU use the CUDA‑specific: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
---

## Step 1 Prepare low‑resolution data

Down‑sample and blur the DIV2K HR images. This creates all LR folders required for training & testing:

```bash
python dataprep.py
```

New folders (after the script finishes):

```
data/
├── DIV2K_train_LR_X2_blurred/   (800 PNGs)
├── DIV2K_train_LR_X4_blurred/   (800 PNGs)
├── DIV2K_val_HR/                (90  PNGs)
├── DIV2K_valid_LR_X4_blurred/   (90  PNGs)
├── DIV2K_test_HR/               (10  PNGs)
└── DIV2K_test_LR_X4_blurred/    (10  PNGs)
```

---

## Step 2 Train Cascade2×

```bash
# full training on seeds 0 1 2 (default)
python main.py                

# quick run on a single seed, e.g. seed 1
python main.py --seeds 1

# multi-seed run
python main.py --seeds 5 6 7
```

For every seed the script will

1. create `checkpoints/seed<seed>/`.
2. save the best model as `cascade_best_seed<seed>.pth`.
3. write training curves to `log_seed<seed>.npy`.
4. evaluate on DIV2K *val + test* (PSNR / SSIM / FID) and append the results to the same log.

After the last seed finishes an aggregate table (mean ± std across seeds) is printed automatically.

---

## Step 3 Cascade2× evaluation & qualitative examples

Generate averaged metrics and a shows example images (2 worst + 2 best) for one checkpoint:

```bash
python cascade_image.py --model-path checkpoints/seed1/cascade_best_seed1.pth --seed 1
```

The script prints PSNR / SSIM / FID, saves four PNGs to `qual_examples_seed1/` and shows an SR‑vs‑HR figure.

---

## Step 4 Baseline EDSR‑×4 evaluation

Run the pre-trained baseline on the same splits for a fair comparison:

```bash
python baseline_edsr.py
```

Outputs:

- averaged PSNR / SSIM / FID.
- four illustrative PNGs (2 worst + 2 best) stored in `baseline_examples/`.

You can edit the paths at the top of `baseline_edsr.py` if your folder layout is different.

---

## Step 5 Comparison

Find the single image on which **either** model beats the other by the largest PSNR margin.

```bash
# (a) baseline wins the most
python compare_models.py --baseline-ckpt edsr_baseline_x4.pt --cascade-ckpt  checkpoints/seed1/cascade_best_seed1.pth --winner baseline

# (b) cascade wins the most
python compare_models.py --baseline-ckpt edsr_baseline_x4.pt --cascade-ckpt  checkpoints/seed1/cascade_best_seed1.pth --winner cascade
```

Each call super‑resolves all 100 LR inputs with **both** networks, computes PSNR for every image, selects the strongest win for *winner*, saves that SR PNG to `comparisons/<winner>_better/`, and prints a one‑line table with file name and metric gap.



---

## Extra utilities

### Quick augmentation preview (`preview.py`)

Visualise what the training data‑loader actually delivers – shows cropping and augmentation choices.

```bash
python preview.py
```

Adjust the flags at the bottom of the file to toggle:

- `augment` – enable/disable flips & rotations.
- `random_crop` – random crop vs. centre crop.

---

### Training‑curve viewer (`log_view.py`)

Plot L1‑loss and PSNR curves that **train.py** stores in every `log_seed*.npy`.

```bash
python log_view.py checkpoints/seed0/log_seed0.npy checkpoints/seed1/log_seed1.npy checkpoints/seed2/log_seed2.npy
```

Produces side‑by‑side L1 and PSNR plots for every seed.

---

### Visual inspection (`example_output.py`)

Show the super‑resolved outputs of **both** networks on a *single* DIV2K image, together with PSNR/SSIM numbers.

```bash
python example_output.py --idx 844 --baseline-ckpt edsr_baseline_x4.pt --cascade-ckpt  checkpoints/seed1/cascade_best_seed1.pth 

```

The script:

1. Picks LR/HR paths for the requested `idx` (801‑900).
2. Runs forward‑passes through baseline EDSR‑×4 and Cascade2×.
3. Prints PSNR & SSIM for both models.
4. Opens a side‑by‑side figure; optionally saves the two SR PNGs if `--save-dir` is given.

