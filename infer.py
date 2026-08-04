"""
infer.py

Responsible only for: loading checkpoint, restoring images, saving/
displaying outputs. This is a near-verbatim reassembly of the original
script's Sections 2-4 (LOAD MODEL & INFERENCE HELPER, FULL DATASET
EVALUATION, VISUAL COMPARISONS) using the moved-out pieces from
model.py and utils.py. No math, shapes, or logic changed.
"""
import os
import glob
import random
import numpy as np
import torch
from tqdm import tqdm

from model import RepPhyDAS_Trial12
from utils import (
    compute_psnr, compute_ssim, load_checkpoint,
    denormalize_and_clamp, clamp_to_uint8,
    plot_train_comparisons, plot_test_comparisons,
)

MODEL_PATH = "trial_12_pro_model.pth"
DATASET_GLOB = "./dataset/**/NoisyLR/*.npy"
TEST_DIR = "./test_data"
BASE_WIDTH = 32


def infer_image(model, npy_path, a, b, device):
    """
    Verbatim from the original infer_image():
        arr = np.load(npy_path).astype(np.float32)
        if arr.max() > 2.0:
            arr = arr / 255.0
        in_norm = (arr - a) / (b - a + 1e-6)
        in_tensor = torch.tensor(in_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            out_tensor = model(in_tensor)
        ...
    """
    arr = np.load(npy_path).astype(np.float32)
    if arr.max() > 2.0:
        arr = arr / 255.0

    in_norm = (arr - a) / (b - a + 1e-6)
    in_tensor = torch.tensor(in_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        out_tensor = model(in_tensor)

    out_clamped = denormalize_and_clamp(out_tensor, a, b)
    input_clamped = clamp_to_uint8(arr)

    return input_clamped, out_clamped


def run_full_dataset_evaluation(model, a, b, device):
    """Verbatim from original Section 3."""
    print("\n--- RUNNING FULL DATASET METRIC EVALUATION (ALL 3200 PAIRS) ---")
    train_files = glob.glob(DATASET_GLOB, recursive=True)
    train_files = [f for f in train_files if "__MACOSX" not in f]

    psnr_list = []
    ssim_list = []

    for in_path in tqdm(train_files, desc="Evaluating Full Dataset"):
        gt_path = in_path.replace("NoisyLR", "GT")
        gt_arr = np.load(gt_path).astype(np.float32)
        if gt_arr.max() > 2.0:
            gt_arr /= 255.0
        gt_clamped = clamp_to_uint8(gt_arr)

        _, out_img = infer_image(model, in_path, a, b, device)

        p_val = compute_psnr(gt_clamped, out_img, data_range=255)
        s_val = compute_ssim(gt_clamped, out_img, data_range=255)
        psnr_list.append(p_val)
        ssim_list.append(s_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n=======================================================")
    print(f"  TRIAL 12 PRO RESULTS (3,200 Pairs Evaluated)")
    print(f"  -> AVERAGE PSNR : {avg_psnr:.4f} dB")
    print(f"  -> AVERAGE SSIM : {avg_ssim:.4f}")
    print("=======================================================\n")

    return train_files, avg_psnr, avg_ssim


def run_visual_comparisons(model, a, b, device, train_files):
    """Verbatim from original Section 4."""
    print("--- GENERATING VISUAL COMPARISONS ON TRAINING DATA ---")
    sample_trains = random.sample(train_files, 3)

    train_records = []
    for in_path in sample_trains:
        gt_path = in_path.replace("NoisyLR", "GT")
        gt_arr = np.load(gt_path).astype(np.float32)
        if gt_arr.max() > 2.0:
            gt_arr /= 255.0
        gt_clamped = clamp_to_uint8(gt_arr)
        in_img, out_img = infer_image(model, in_path, a, b, device)
        p_val = compute_psnr(gt_clamped, out_img, data_range=255)
        train_records.append((in_img, gt_clamped, out_img, p_val))

    plot_train_comparisons(train_records)

    test_files = glob.glob(os.path.join(TEST_DIR, "**", "*.npy"), recursive=True)
    test_files = [f for f in test_files if "__MACOSX" not in f]

    if len(test_files) > 0:
        sample_tests = random.sample(test_files, min(3, len(test_files)))
        test_records = []
        for t_path in sample_tests:
            in_img, out_img = infer_image(model, t_path, a, b, device)
            test_records.append((in_img, out_img))
        plot_test_comparisons(test_records)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RepPhyDAS_Trial12(base_width=BASE_WIDTH).to(device)
    a, b = load_checkpoint(MODEL_PATH, model, device)
    if a is None:
        return

    train_files, avg_psnr, avg_ssim = run_full_dataset_evaluation(model, a, b, device)
    run_visual_comparisons(model, a, b, device, train_files)


if __name__ == "__main__":
    main()
