"""NAFNet-style x2 grayscale restoration network used by the checkpoint."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        expanded = channels * dw_expand
        ffn_channels = channels * ffn_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.conv2 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.gate = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded // 2, expanded // 2, 1),
        )
        self.conv3 = nn.Conv2d(expanded // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm1(x)
        residual = self.conv1(residual)
        residual = self.conv2(residual)
        residual = self.gate(residual)
        residual = residual * self.channel_attention(residual)
        x = x + self.beta * self.conv3(residual)
        residual = self.conv4(self.norm2(x))
        residual = self.gate(residual)
        return x + self.gamma * self.conv5(residual)


def _blocks(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(NAFBlock(channels) for _ in range(count)))


class NAFNetSR(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        scale_factor: int = 2,
        width: int = 24,
        enc_blocks: Sequence[int] = (1, 1, 2),
        middle_blocks: int = 2,
        dec_blocks: Sequence[int] | None = None,
    ):
        super().__init__()
        if scale_factor != 2:
            raise ValueError("This NAFNet-SR implementation supports scale_factor=2 only")
        if width < 4:
            raise ValueError("width must be at least 4")
        if not enc_blocks or any(count < 1 for count in enc_blocks):
            raise ValueError("enc_blocks must contain one or more positive block counts")
        if dec_blocks is None:
            dec_blocks = tuple(reversed(tuple(enc_blocks)))
        if len(dec_blocks) != len(enc_blocks) or any(count < 1 for count in dec_blocks):
            raise ValueError("dec_blocks must have one positive count per encoder stage")

        self.scale_factor = scale_factor
        self.padder_size = 2 ** len(enc_blocks)
        self.intro = nn.Conv2d(input_channels, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for count in enc_blocks:
            self.encoders.append(_blocks(channels, count))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = _blocks(channels, middle_blocks)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for count in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(_blocks(channels, count))

        self.ending = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.Conv2d(width, output_channels * scale_factor * scale_factor, 3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        if degraded.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], received {tuple(degraded.shape)}")
        height, width = degraded.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        degraded_padded = F.pad(degraded, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else degraded
        features = self.intro(degraded_padded)
        skips: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle(features)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            features = decoder(up(features) + skip)
        correction = self.ending(features)[..., : height * self.scale_factor, : width * self.scale_factor]
        base = F.interpolate(degraded, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        return base + correction
