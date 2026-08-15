"""Small device helpers shared by training, inference, and evaluation."""

from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, then MPS, then CPU, or validate an explicit device request."""
    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "auto":
        return torch.device("cuda" if has_cuda else "mps" if has_mps else "cpu")
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not has_mps:
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work so a timing measurement is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
