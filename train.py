"""
train.py

NOTE ON PROVENANCE:
The original single-file script has NO training loop, NO optimizer, NO
scheduler, and NO epoch/step logic anywhere in it. It only loads an
already-trained checkpoint and runs evaluation/plotting. So there is
nothing to "move" into this file -- everything below is NEW scaffolding
written to satisfy the requested responsibilities (config, dataset,
model, optimizer, scheduler, loop, validation, checkpoint saving).

Concretely fabricated here, with no source basis, and left as
placeholders you should replace with the real training recipe:
  - optimizer type/learning rate (Adam, lr=1e-4 is a generic default)
  - scheduler type/params (StepLR is a generic default)
  - number of epochs, batch size
  - validation split strategy
  - the `a`, `b` normalization baselines (in the original, these were
    only ever *loaded* from a checkpoint, never computed from a dataset;
    here I compute them as the global min/max over the training set the
    first time, as a placeholder -- this is NOT verified against any
    original training code, because none was provided)

Do not treat these defaults as "identical functionality" to whatever
originally produced trial_12_pro_model.pth -- there is no way to know
that from the source you gave me.
"""
import torch
from torch.utils.data import random_split

from model import RepPhyDAS_Trial12
from dataset import NpyPairDataset, find_pairs, load_and_normalize_npy
from loss import get_loss_fn
from utils import save_checkpoint, compute_psnr, compute_ssim, clamp_to_uint8, denormalize_and_clamp

import numpy as np

# ---------------------------------------------------------------------
# Configuration (placeholders -- not extracted from any source material)
# ---------------------------------------------------------------------
CONFIG = {
    "dataset_root": "./dataset",
    "checkpoint_out": "trial_12_pro_model.pth",
    "base_width": 32,
    "batch_size": 8,
    "epochs": 50,
    "lr": 1e-4,
    "val_fraction": 0.1,
    "scheduler_step_size": 20,
    "scheduler_gamma": 0.5,
    "loss_name": "l1",
}


def compute_normalization_baselines(dataset_root):
    """
    PLACEHOLDER, not from source. Original script only ever *loaded* a
    and b from a checkpoint; it never showed how they were computed.
    This takes the global min/max over the NoisyLR training files as a
    stand-in so the pipeline is runnable end-to-end.
    """
    pairs = find_pairs(dataset_root)
    mins, maxs = [], []
    for noisy_path, _ in pairs:
        arr = load_and_normalize_npy(noisy_path)
        mins.append(arr.min())
        maxs.append(arr.max())
    return float(np.min(mins)), float(np.max(maxs))


def validate(model, val_loader, device):
    """PLACEHOLDER validation loop -- not from source."""
    model.eval()
    psnr_list, ssim_list = [], []
    with torch.inference_mode():
        for noisy, gt, _ in val_loader:
            noisy = noisy.to(device)
            out = model(noisy)
            out_np = out.squeeze(1).cpu().numpy()
            gt_np = gt.squeeze(1).cpu().numpy()
            for o, g in zip(out_np, gt_np):
                o_clamped = clamp_to_uint8(o)
                g_clamped = clamp_to_uint8(g)
                psnr_list.append(compute_psnr(g_clamped, o_clamped, data_range=255))
                ssim_list.append(compute_ssim(g_clamped, o_clamped, data_range=255))
    model.train()
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- normalization baselines (placeholder, see docstring) ---
    a, b = compute_normalization_baselines(CONFIG["dataset_root"])

    # --- dataset ---
    full_dataset = NpyPairDataset(CONFIG["dataset_root"], a=a, b=b)
    val_size = int(len(full_dataset) * CONFIG["val_fraction"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )

    # --- model ---
    model = RepPhyDAS_Trial12(base_width=CONFIG["base_width"]).to(device)

    # --- optimizer / scheduler (placeholder choices, see docstring) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CONFIG["scheduler_step_size"], gamma=CONFIG["scheduler_gamma"]
    )

    criterion = get_loss_fn(CONFIG["loss_name"])

    best_psnr = -1.0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0

        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)

            optimizer.zero_grad()
            out = model(noisy)
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)

        scheduler.step()
        avg_train_loss = running_loss / train_size

        val_psnr, val_ssim = validate(model, val_loader, device)
        print(f"[Epoch {epoch+1}/{CONFIG['epochs']}] "
              f"train_loss={avg_train_loss:.6f}  val_psnr={val_psnr:.4f}  val_ssim={val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(CONFIG["checkpoint_out"], model, a, b)
            print(f"  -> New best checkpoint saved (val_psnr={val_psnr:.4f})")


if __name__ == "__main__":
    main()
