"""
loss.py

NOTE ON PROVENANCE:
The original single-file script contains NO loss function of any kind.
It is an inference + full-dataset PSNR/SSIM evaluation script -- there is
no `.backward()`, no `optimizer`, no `criterion`, no loss class or
function anywhere in the source you provided.

Everything below is therefore NEW code, not a refactor of existing
logic. I am not going to guess what loss RepPhyDAS Trial 12 was actually
trained with (L1? Charbonnier? perceptual/VGG? some combination weighted
by the physics terms like RCE/g_n?) -- the comment in the original
plotting title says "VGG + Omni-Detail" which suggests a VGG perceptual
term was part of training, but no VGG loss code exists in the file, so I
am not fabricating one.

What's provided here is a minimal, clearly-labeled placeholder
(plain L1) so train.py has something callable. Replace `get_loss_fn`
with whatever the actual training objective was -- I have no source
evidence for it.
"""
import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    PLACEHOLDER ONLY -- not extracted from any original source code.
    Plain L1 between model output and ground truth.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(pred, target)


def get_loss_fn(name="l1"):
    """
    PLACEHOLDER factory -- not extracted from source.
    Swap in the real training loss(es) here once you have that logic.
    """
    if name == "l1":
        return ReconstructionLoss()
    raise ValueError(f"Unknown loss name: {name} (no other loss was defined in the source material)")
