"""
Computer Vision Architectures for Candlestick Pattern Recognition.

Includes:
1. CandlestickCNN: Custom 4-Block CNN with Global Average Pooling (422,659 params).
2. CandlestickResNet18: Transfer learning fine-tuned ResNet-18 (11,178,051 params).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

RESNET18_WEIGHTS_PATH = Path(
    os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
)


class CandlestickCNN(nn.Module):
    """
    Custom 4-Block Convolutional Neural Network for Candlestick Chart Classification.
    Layer-by-layer spec matches PPT_FACT_SHEET.md:
      Block 1: Conv2d(3, 32, 3, pad=1) -> BN -> ReLU -> MaxPool(2,2)
      Block 2: Conv2d(32, 64, 3, pad=1) -> BN -> ReLU -> MaxPool(2,2)
      Block 3: Conv2d(64, 128, 3, pad=1) -> BN -> ReLU -> MaxPool(2,2)
      Block 4: Conv2d(128, 256, 3, pad=1) -> BN -> ReLU -> MaxPool(2,2)
      GAP: AdaptiveAvgPool2d((1,1))
      Classifier: Linear(256, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 3)
    Total parameters: 422,659.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def get_target_layer(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM hooks."""
        return self.block4[0]


class CandlestickResNet18(nn.Module):
    """
    Pretrained ResNet-18 Transfer Learning Model for Candlestick Pattern Recognition.
    Replaces 1000-class fc head with Dropout(0.3) -> Linear(512, 3).
    Supports frozen backbone mode or end-to-end fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = 3,
        freeze_backbone: bool = False,
        weights_path: Optional[Path] = None,
    ):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        
        # Load local pretrained checkpoint if available
        ckpt_path = weights_path or RESNET18_WEIGHTS_PATH
        if ckpt_path.is_file():
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded pretrained ResNet-18 weights from {ckpt_path}")
        else:
            print(f"Warning: Pretrained weights not found at {ckpt_path}, initialized randomly.")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM hooks."""
        return self.backbone.layer4[-1].conv2

