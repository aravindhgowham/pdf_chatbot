from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RawAudioCNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBlock1D(in_channels, c, kernel_size=7, stride=2, padding=3, dropout=dropout),
            ConvBlock1D(c, c * 2, kernel_size=5, stride=2, padding=2, dropout=dropout),
            ConvBlock1D(c * 2, c * 4, kernel_size=5, stride=2, padding=2, dropout=dropout),
            ConvBlock1D(c * 4, c * 4, kernel_size=3, stride=2, padding=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(c * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, time]
        feats = self.features(x)
        pooled = self.pool(feats)  # [batch, channels, 1]
        logits = self.classifier(pooled)
        return logits