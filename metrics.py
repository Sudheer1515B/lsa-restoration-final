"""Validation metrics expressed in the same fixed normalized image range.

Provides per-image PSNR/SSIM computation used for evaluation/validation
(as opposed to the training-time losses in ``losses.py``), plus a small
helper to count a model's parameters.
"""

from __future__ import annotations

import torch

from losses import ssim_per_image


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute fixed-range PSNR independently for each item in a batch.

    Args:
        prediction: Predicted images, shape [B, C, H, W].
        target: Ground-truth images, shape [B, C, H, W], matching ``prediction``.
        data_range: Value range of the pixel data (e.g. 1.0 for images
            normalized to [0, 1]), used as the peak signal value.

    Returns:
        A tensor of shape [B] with the PSNR (in dB) for each image.

    Raises:
        ValueError: If ``prediction`` and ``target`` shapes differ.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"PSNR shapes differ: {tuple(prediction.shape)} and {tuple(target.shape)}")
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    # Clamp MSE away from zero to avoid a division blow-up (and log10(inf)/NaN)
    # for a perfectly matching image.
    return 10.0 * torch.log10(torch.as_tensor(data_range**2, device=prediction.device, dtype=prediction.dtype) / mse.clamp_min(1e-12))


def metric_per_image(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PSNR and SSIM together, aligned per-image.

    Args:
        prediction: Predicted images, shape [B, C, H, W].
        target: Ground-truth images, shape [B, C, H, W], matching ``prediction``.
        data_range: Value range of the pixel data, forwarded to both metrics.

    Returns:
        A tuple ``(psnr, ssim)`` of tensors, each of shape [B].
    """
    return psnr_per_image(prediction, target, data_range), ssim_per_image(prediction, target, data_range)


def bucket_by_severity(records: list[dict[str, float | str]], num_buckets: int = 3) -> list[dict[str, float | int]]:
    """Group per-image validation records into noise-severity buckets and summarize each.

    A single dataset-wide mean PSNR/SSIM can hide a model that restores mildly
    corrupted images well while barely touching the most heavily corrupted
    ones (e.g. speckle-heavy images), because the easy majority dominates the
    average. This splits records into equal-count groups ordered by each
    record's noise severity (ascending) and reports metrics per group, so the
    most-corrupted third (or other fraction) can be tracked on its own.

    Args:
        records: Per-image dicts, each with at least ``psnr``, ``ssim``, and
            ``noise_std`` keys (``noise_std`` is a per-image severity proxy,
            e.g. the standard deviation of the raw degraded input).
        num_buckets: Number of equal-count severity groups to form, ordered
            from least to most severe. Clamped down to ``len(records)`` if
            there are fewer records than buckets (e.g. a tiny smoke run).

    Returns:
        A list of ``num_buckets`` dicts (least to most severe), each with
        ``bucket`` (0-indexed), ``count``, ``noise_std_min``,
        ``noise_std_max``, ``mean_psnr``, and ``mean_ssim``.

    Raises:
        ValueError: If ``records`` is empty or ``num_buckets`` is not a
            positive integer.
    """
    if not records:
        raise ValueError("records must be non-empty")
    if num_buckets < 1:
        raise ValueError("num_buckets must be a positive integer")
    num_buckets = min(num_buckets, len(records))

    ordered = sorted(records, key=lambda record: float(record["noise_std"]))
    boundaries = [round(index * len(ordered) / num_buckets) for index in range(num_buckets + 1)]
    summaries: list[dict[str, float | int]] = []
    for bucket in range(num_buckets):
        group = ordered[boundaries[bucket] : boundaries[bucket + 1]]
        summaries.append(
            {
                "bucket": bucket,
                "count": len(group),
                "noise_std_min": float(group[0]["noise_std"]),
                "noise_std_max": float(group[-1]["noise_std"]),
                "mean_psnr": sum(float(item["psnr"]) for item in group) / len(group),
                "mean_ssim": sum(float(item["ssim"]) for item in group) / len(group),
            }
        )
    return summaries


def parameter_count(model: torch.nn.Module) -> int:
    """Count all trainable and non-trainable parameter elements in a model.

    Args:
        model: The PyTorch module to inspect.

    Returns:
        The total number of scalar elements across all parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters())
