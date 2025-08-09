import torch
import torch.nn as nn


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1, se: bool = True, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SqueezeExcite1D(out_ch) if se else nn.Identity()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class TemporalAttentionPool(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.attn = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        logits = self.attn(x)               # [B, 1, T]
        weights = torch.softmax(logits, dim=-1)
        pooled = (x * weights).sum(dim=-1)  # [B, C]
        return pooled


class RawAudioResNet1D(nn.Module):
    """
    Deeper residual 1D CNN for raw waveform classification.
    First conv uses large receptive field; subsequent residual stages downsample.
    """
    def __init__(self, num_classes: int = 2, base_channels: int = 32, dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=80, stride=4, padding=38, bias=False),  # large RF at 16k SR
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock1D(c, c, kernel_size=3, stride=1, dilation=1, se=True, dropout=dropout),
            ResBlock1D(c, c, kernel_size=3, stride=1, dilation=1, se=True, dropout=dropout),
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(c, c * 2, kernel_size=3, stride=2, dilation=1, se=True, dropout=dropout),
            ResBlock1D(c * 2, c * 2, kernel_size=3, stride=1, dilation=1, se=True, dropout=dropout),
        )
        self.layer3 = nn.Sequential(
            ResBlock1D(c * 2, c * 4, kernel_size=3, stride=2, dilation=2, se=True, dropout=dropout),
            ResBlock1D(c * 4, c * 4, kernel_size=3, stride=1, dilation=2, se=True, dropout=dropout),
        )
        self.layer4 = nn.Sequential(
            ResBlock1D(c * 4, c * 4, kernel_size=3, stride=2, dilation=4, se=True, dropout=dropout),
            ResBlock1D(c * 4, c * 4, kernel_size=3, stride=1, dilation=4, se=True, dropout=dropout),
        )
        self.attn_pool = TemporalAttentionPool(c * 4) if use_attention else None
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(c * 4, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(c * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.attn_pool is not None:
            feat = self.attn_pool(x)    # [B, C]
        else:
            feat = self.gap(x).squeeze(-1)  # [B, C]
        logits = self.head(feat)
        return logits