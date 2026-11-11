"""
models.py – Phase 3: Model definitions for PlantVillage classification.

Provides:
  SmallCNN    – 4-block CNN (Conv/BN/ReLU/MaxPool) + GAP + Dropout + Linear.
                Input: 128×128 RGB.  Small enough to train on CPU.
  build_resnet18(num_classes, pretrained)
                – ImageNet-pretrained ResNet18 with the classifier head replaced.
                  Input: 224×224 (or 160×160 to save CPU time).

Usage:
    from src.models import SmallCNN, build_resnet18
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# 1. Small CNN
#    Architecture: 4 × (Conv2d → BN → ReLU → MaxPool2d)
#                  filters: 32 → 64 → 128 → 256
#                  Global Average Pooling → Dropout(0.4) → Linear
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + ReLU + MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, pool: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool, stride=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNN(nn.Module):
    """
    Small CNN for PlantVillage (128×128 input).

    Block layout (after each MaxPool spatial dims halve):
      Input 128×128 →  32ch → 64×64
                     →  64ch → 32×32
                     → 128ch → 16×16
                     → 256ch →  8×8
      Global Average Pool → (256,)
      Dropout(0.4) → Linear(256, num_classes)

    Parameters
    ----------
    num_classes : int  – number of output classes
    dropout     : float – dropout probability before the classifier head
    """

    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(3,   32),
            _ConvBlock(32,  64),
            _ConvBlock(64,  128),
            _ConvBlock(128, 256),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)       # Global Average Pooling
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (B, 256, H', W')
        x = self.gap(x)            # (B, 256, 1, 1)
        x = x.flatten(1)           # (B, 256)
        return self.head(x)        # (B, num_classes)


# ---------------------------------------------------------------------------
# 2. ResNet18 with replaced head
# ---------------------------------------------------------------------------

def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Load ResNet18 (optionally with ImageNet weights) and replace the final
    fully-connected layer for *num_classes* outputs.

    Fine-tuning stages (managed in train_dl.py):
      Stage 1 – freeze the backbone, train only the new head.
      Stage 2 – unfreeze layer4 (and head), fine-tune with a lower LR.

    Parameters
    ----------
    num_classes : int
    pretrained  : bool – use IMAGENET1K_V1 weights (default True)

    Returns
    -------
    nn.Module with `.fc` replaced by Linear(512, num_classes)
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the final `fc` layer (ResNet18 stage 1)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_layer4(model: nn.Module) -> None:
    """Unfreeze layer4 + fc for stage-2 fine-tuning (ResNet18)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4.") or name.startswith("fc.")


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

