"""
dataset.py

NOTE ON PROVENANCE:
The original script had no Dataset/DataLoader class — it only did a flat
`glob.glob(...)` + Python for-loop over .npy files for evaluation. The
file-pairing convention (NoisyLR/*.npy <-> GT/*.npy) and the exact
preprocessing steps below (the `if arr.max() > 2.0: arr /= 255.0` check,
and clamped uint8 conversion) are taken verbatim from the original
`infer_image()` and dataset-evaluation loop.

The `NpyPairDataset` class itself, `__getitem__`, and the DataLoader
factory function are NEW code written to satisfy the requested file
structure -- they did not exist as a class in the original file. Review
them; they are not "identical functionality preserved" because there was
no prior Dataset implementation to preserve.

One thing this file CANNOT reproduce faithfully: the original script's
per-pixel normalization used `a` and `b` (min/max baselines) loaded from
the *training checkpoint* (`ckpt['a']`, `ckpt['b']`), not computed inside
the dataset loading code. There is no source code anywhere in the
original file that computes `a`/`b` from the dataset itself. This file
takes `a`/`b` as constructor arguments so the caller (train.py) must
supply/compute them explicitly rather than the dataset silently deriving
them -- avoiding a fabricated normalization step.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def find_pairs(dataset_root):
    """
    Reproduces the exact glob pattern and filtering used in the original
    script's Section 3 (full dataset evaluation):
        train_files = glob.glob("./dataset/**/NoisyLR/*.npy", recursive=True)
        train_files = [f for f in train_files if "__MACOSX" not in f]
        gt_path = in_path.replace("NoisyLR", "GT")
    """
    noisy_files = glob.glob(os.path.join(dataset_root, "**", "NoisyLR", "*.npy"), recursive=True)
    noisy_files = [f for f in noisy_files if "__MACOSX" not in f]
    pairs = [(f, f.replace("NoisyLR", "GT")) for f in noisy_files]
    return pairs


def load_and_normalize_npy(path, max_val_threshold=2.0):
    """
    Exact preprocessing branch from the original infer_image()/eval loop:
        arr = np.load(npy_path).astype(np.float32)
        if arr.max() > 2.0:
            arr = arr / 255.0
    """
    arr = np.load(path).astype(np.float32)
    if arr.max() > max_val_threshold:
        arr = arr / 255.0
    return arr


class NpyPairDataset(Dataset):
    """
    NEW scaffolding class (see module docstring). Wraps the NoisyLR/GT
    pairing and the [0,1]-range normalization branch from the original
    script into a standard torch Dataset. Does not add augmentation,
    resizing, or any transform that wasn't present in the original code.

    a, b: normalization baselines. In the original script these came from
    `ckpt['a']`, `ckpt['b']` at inference time -- there is no dataset-side
    computation of them in the source, so they must be supplied here.
    """
    def __init__(self, dataset_root, a=0.0, b=1.0):
        self.pairs = find_pairs(dataset_root)
        self.a = a
        self.b = b

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]

        noisy_arr = load_and_normalize_npy(noisy_path)
        gt_arr = load_and_normalize_npy(gt_path)

        # Same normalization formula used in infer_image():
        # in_norm = (arr - a) / (b - a + 1e-6)
        noisy_norm = (noisy_arr - self.a) / (self.b - self.a + 1e-6)

        noisy_tensor = torch.tensor(noisy_norm).unsqueeze(0)  # (1, H, W)
        gt_tensor = torch.tensor(gt_arr).unsqueeze(0)          # (1, H, W), left in raw [0,1] scale

        return noisy_tensor, gt_tensor, noisy_path


def build_dataloader(dataset_root, a=0.0, b=1.0, batch_size=8, shuffle=True, num_workers=0):
    """NEW scaffolding: standard DataLoader factory, no augmentation added."""
    dataset = NpyPairDataset(dataset_root, a=a, b=b)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
