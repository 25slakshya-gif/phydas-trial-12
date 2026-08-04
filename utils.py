"""
utils.py

PSNR/SSIM computation, checkpoint loading, and plotting logic moved
verbatim (same calls, same args) from the original single-file script.
`save_checkpoint` did not exist in the original (only loading did) and
is added as a small, clearly-marked convenience so train.py has
something to call.
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_psnr(gt_img, out_img, data_range=255):
    """Same call as original: psnr(gt_clamped, out_img, data_range=255)"""
    return psnr(gt_img, out_img, data_range=data_range)


def compute_ssim(gt_img, out_img, data_range=255):
    """Same call as original: ssim(gt_clamped, out_img, data_range=255)"""
    return ssim(gt_img, out_img, data_range=data_range)


def load_checkpoint(model_path, model, device):
    """
    Verbatim from the original loading block:
        ckpt = torch.load(model_path, map_location=device)
        a, b = ckpt['a'], ckpt['b']
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
    """
    if not os.path.exists(model_path):
        print(f"[!] Error: {model_path} not found. Ensure Trial 12 training completed successfully.")
        return None, None

    ckpt = torch.load(model_path, map_location=device)
    a, b = ckpt['a'], ckpt['b']
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Trial 12 RepPhyDAS-Pro Model loaded successfully! Normalization baselines -> a: {a:.6f}, b: {b:.6f}")
    return a, b


def save_checkpoint(model_path, model, a, b):
    """
    NOT present in the original script (which only ever loaded a
    checkpoint, never saved one). Added as a minimal counterpart so
    train.py can persist weights + the a/b normalization baselines in
    the same dict shape the original loader expects
    (`ckpt['model_state_dict']`, `ckpt['a']`, `ckpt['b']`).
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'a': a,
        'b': b,
    }, model_path)


def denormalize_and_clamp(out_tensor, a, b):
    """
    Verbatim from infer_image():
        out_arr = out_tensor.squeeze().cpu().numpy()
        out_denorm = out_arr * (b - a) + a
        out_clamped = np.clip(out_denorm * 255.0, 0, 255).astype(np.uint8)
    """
    out_arr = out_tensor.squeeze().cpu().numpy()
    out_denorm = out_arr * (b - a) + a
    out_clamped = np.clip(out_denorm * 255.0, 0, 255).astype(np.uint8)
    return out_clamped


def clamp_to_uint8(arr):
    """Verbatim from original: np.clip(arr * 255.0, 0, 255).astype(np.uint8)"""
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def plot_train_comparisons(sample_records):
    """
    Verbatim structure from original Section 4 (training-data comparison
    figure). `sample_records` is a list of
    (in_img, gt_clamped, out_img, psnr_value) tuples, exactly the values
    the original loop computed before plotting.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Trial 12: Input vs Ground Truth vs Restored Output (VGG + Omni-Detail)", fontsize=16)

    for i, (in_img, gt_clamped, out_img, psnr_value) in enumerate(sample_records):
        axes[i, 0].imshow(in_img, cmap='gray'); axes[i, 0].set_title("Degraded Input (128x128)"); axes[i, 0].axis('off')
        axes[i, 1].imshow(gt_clamped, cmap='gray'); axes[i, 1].set_title("Ground Truth (256x256)"); axes[i, 1].axis('off')
        axes[i, 2].imshow(out_img, cmap='gray'); axes[i, 2].set_title(f"Restored (PSNR: {psnr_value:.2f})"); axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_test_comparisons(sample_records):
    """
    Verbatim structure from original Section 4 (unseen test-data figure).
    `sample_records` is a list of (in_img, out_img) tuples.
    """
    n = len(sample_records)
    fig2, axes2 = plt.subplots(n, 2, figsize=(12, 4 * n))
    fig2.suptitle("Trial 12 Unseen Test Data: Input vs Restored Output", fontsize=16)

    for i, (in_img, out_img) in enumerate(sample_records):
        ax_in = axes2[i, 0] if n > 1 else axes2[0]
        ax_out = axes2[i, 1] if n > 1 else axes2[1]
        ax_in.imshow(in_img, cmap='gray'); ax_in.set_title("Noisy Test Input (128x128)"); ax_in.axis('off')
        ax_out.imshow(out_img, cmap='gray'); ax_out.set_title("Trial 12 Restored Output (256x256)"); ax_out.axis('off')

    plt.tight_layout()
    plt.show()
