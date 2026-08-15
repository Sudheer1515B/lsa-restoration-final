"""A lightweight, single-image NAFNet-style backbone for x2 restoration.

This module intentionally uses only standard PyTorch operations so a checkpoint
can be trained on CPU/MPS and restored on CUDA without architectural changes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Layer normalization over channels for a channels-first image tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        """Initialize learnable per-channel scale/bias.

        Args:
            channels: Number of feature channels to normalize.
            eps: Small constant added to variance for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize channels at every spatial location without changing layout."""
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    """Multiply the two equal channel halves used by NAFNet blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gating operation to an even-channel feature tensor."""
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Normalization-free attention block from the NAFNet family."""

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        """Build the block's spatial-attention branch and feed-forward branch.

        Args:
            channels: Number of input/output feature channels for this block.
            dw_expand: Channel expansion factor used before the depthwise
                convolution and simple-gate channel attention branch.
            ffn_expand: Channel expansion factor used inside the feed-forward
                (FFN) branch.
        """
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
        """Apply spatial mixing, channel attention, and the feed-forward residual."""
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
    """Create a sequential stack of ``count`` NAFBlocks at a fixed channel width.

    Args:
        channels: Feature channel width shared by every block in the stack.
        count: Number of NAFBlocks to chain together.

    Returns:
        An ``nn.Sequential`` containing ``count`` NAFBlock instances.
    """
    return nn.Sequential(*(NAFBlock(channels) for _ in range(count)))


class NAFNetSR(nn.Module):
    """NAFNet encoder/decoder with a residual PixelShuffle x2 head.

    The encoder and decoder operate at low resolution.  The final head predicts
    a correction over bicubic x2 upsampling, which gives a useful stable
    initialization for the first super-resolution baseline.
    """

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
        """Configure and build the encoder/middle/decoder stages and heads.

        Args:
            input_channels: Number of channels in the low-resolution input
                (1 for grayscale).
            output_channels: Number of channels in the restored output.
            scale_factor: Spatial upsampling factor; only 2 is supported by
                this baseline.
            width: Base feature channel width at the top (finest) encoder
                stage; doubles at each downsampling step.
            enc_blocks: Number of NAFBlocks per encoder stage, from finest to
                coarsest resolution. Its length also sets the number of
                downsampling steps.
            middle_blocks: Number of NAFBlocks in the bottleneck between the
                encoder and decoder.
            dec_blocks: Number of NAFBlocks per decoder stage, from coarsest
                to finest resolution. Defaults to ``enc_blocks`` reversed so
                the decoder mirrors the encoder.

        Raises:
            ValueError: If ``scale_factor`` is not 2, ``width`` is too small,
                or ``enc_blocks``/``dec_blocks`` are empty, mismatched in
                length, or contain non-positive counts.
        """
        super().__init__()
        if scale_factor != 2:
            raise ValueError("This first NAFNet-SR baseline supports scale_factor=2 only")
        if width < 4:
            raise ValueError("width must be at least 4")
        if not enc_blocks or any(count < 1 for count in enc_blocks):
            raise ValueError("enc_blocks must contain one or more positive block counts")
        if dec_blocks is None:
            dec_blocks = tuple(reversed(tuple(enc_blocks)))
        if len(dec_blocks) != len(enc_blocks) or any(count < 1 for count in dec_blocks):
            raise ValueError("dec_blocks must have one positive count per encoder stage")

        self.scale_factor = scale_factor
        # Each encoder stage halves spatial resolution once, so the input must be a
        # multiple of 2 ** (number of stages) for the encoder/decoder skip shapes to
        # line up again after upsampling.
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
        """Restore a low-resolution batch and return an exactly 2× spatial output."""
        if degraded.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], received {tuple(degraded.shape)}")
        height, width = degraded.shape[-2:]
        # Pad up to the nearest multiple of padder_size (via replication, to avoid
        # introducing artificial edges) so every downsampling step divides evenly;
        # the padding is cropped back off below using the original height/width.
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        if pad_h or pad_w:
            degraded_padded = F.pad(degraded, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            degraded_padded = degraded
        features = self.intro(degraded_padded)
        skips: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle(features)
        # Decoder stages consume skip connections in reverse (finest-first
        # collected, coarsest-first consumed) to match the mirrored decoder order.
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            features = up(features)
            features = decoder(features + skip)
        correction = self.ending(features)[..., : height * self.scale_factor, : width * self.scale_factor]
        # The network only has to learn a residual correction on top of a cheap
        # bicubic upsample, which gives a stable, well-behaved training target.
        base = F.interpolate(degraded, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        return base + correction
