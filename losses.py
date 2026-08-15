"""Reconstruction losses for image restoration.

Provides the pixel-space and structural loss terms used to train the
NAFNet-SR model: a differentiable SSIM implementation, a Charbonnier
(smooth L1) pixel loss, and a combined reconstruction loss that mixes a
configurable pixel loss with an SSIM penalty.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(-2, -1)


def sobel_gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between Sobel-filtered prediction and target (x and y derivatives).

    Rewards matching edge location/strength directly, rather than relying on
    the pixel and SSIM terms to reproduce edges as a side effect.

    Args:
        prediction: Predicted images, shape [B, 1, H, W] (single-channel).
        target: Ground-truth images, shape [B, 1, H, W], matching ``prediction``.

    Returns:
        A scalar tensor: mean L1 distance over both Sobel-x and Sobel-y responses.

    Raises:
        ValueError: If either tensor doesn't have exactly one channel.
    """
    if prediction.shape[1] != 1:
        raise ValueError("sobel_gradient_loss expects single-channel [B, 1, H, W] input")
    sobel_x = _SOBEL_X.to(prediction.device, prediction.dtype)
    sobel_y = _SOBEL_Y.to(prediction.device, prediction.dtype)
    prediction_gx, target_gx = F.conv2d(prediction, sobel_x, padding=1), F.conv2d(target, sobel_x, padding=1)
    prediction_gy, target_gy = F.conv2d(prediction, sobel_y, padding=1), F.conv2d(target, sobel_y, padding=1)
    return F.l1_loss(prediction_gx, target_gx) + F.l1_loss(prediction_gy, target_gy)


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute differentiable SSIM independently for each item in a batch.

    Args:
        prediction: Predicted images, shape [B, C, H, W].
        target: Ground-truth images, shape [B, C, H, W], matching ``prediction``.
        data_range: Value range of the pixel data (e.g. 1.0 for images
            normalized to [0, 1]), used to scale the stability constants.
        window_size: Size of the square sliding window used to estimate local
            statistics (means, variances, covariance). Clamped down to fit the
            image and forced odd.

    Returns:
        A tensor of shape [B] with the mean SSIM score for each image.

    Raises:
        ValueError: If shapes mismatch, tensors aren't 4-D, or the resulting
            window is smaller than 3x3.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"SSIM shapes differ: {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )

    if prediction.ndim != 4:
        raise ValueError("SSIM expects [B, C, H, W]")

    window = min(window_size, prediction.shape[-2], prediction.shape[-1])

    if window % 2 == 0:
        window -= 1

    if window < 3:
        raise ValueError("SSIM requires images at least 3x3")

    # Local means/variances/covariance are estimated with a box filter
    # (avg_pool2d) over a sliding window, following the standard SSIM formula.
    padding = window // 2

    mu_x = F.avg_pool2d(
        prediction,
        window,
        stride=1,
        padding=padding,
    )

    mu_y = F.avg_pool2d(
        target,
        window,
        stride=1,
        padding=padding,
    )

    sigma_x = (
        F.avg_pool2d(
            prediction.square(),
            window,
            stride=1,
            padding=padding,
        )
        - mu_x.square()
    )

    sigma_y = (
        F.avg_pool2d(
            target.square(),
            window,
            stride=1,
            padding=padding,
        )
        - mu_y.square()
    )

    sigma_xy = (
        F.avg_pool2d(
            prediction * target,
            window,
            stride=1,
            padding=padding,
        )
        - mu_x * mu_y
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    score = (
        (2 * mu_x * mu_y + c1)
        * (2 * sigma_xy + c2)
    ) / (
        (mu_x.square() + mu_y.square() + c1)
        * (sigma_x + sigma_y + c2)
        + 1e-12
    )

    return score.mean(dim=(1, 2, 3))


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Compute the Charbonnier (smooth L1) loss between prediction and target.

    A differentiable approximation of L1 loss that avoids the non-smooth
    gradient at zero error.

    Args:
        prediction: Predicted tensor of any shape.
        target: Ground-truth tensor, matching ``prediction`` in shape.
        epsilon: Small constant controlling the smoothing near zero error.

    Returns:
        A scalar tensor with the mean Charbonnier loss over all elements.
    """

    return torch.mean(
        torch.sqrt(
            (prediction - target).square()
            + epsilon**2
        )
    )


def hard_pixel_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    fraction: float = 0.05,
) -> torch.Tensor:
    """Compute mean squared error over each image's hardest (highest-residual) pixels.

    Charbonnier and L1 losses deliberately saturate to a bounded gradient for
    large residuals, and SSIM's windowed statistics dilute the influence of a
    few outlier pixels — both are robust to sparse, large-magnitude errors by
    design. That robustness is exactly wrong for sparse high-magnitude
    corruption such as speckle noise: it gives the network little pressure to
    actually cancel those pixels rather than merely average them away. This
    term restores that pressure with a plain, unbounded squared-error penalty
    applied only to the top ``fraction`` of each image's pixels by residual
    magnitude, so it can grow without saturating and specifically targets the
    pixels the other loss terms under-penalize.

    Args:
        prediction: Predicted images, shape [B, C, H, W].
        target: Ground-truth images, shape [B, C, H, W], matching ``prediction``.
        fraction: Fraction (0, 1] of each image's pixels (by squared error,
            highest first) to include in the mean.

    Returns:
        A scalar tensor: the mean squared error over the selected top-error
        pixels, averaged across the batch.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1]``.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    squared_error = (prediction - target).square().flatten(1)
    pixel_count = max(1, round(squared_error.shape[1] * fraction))
    hardest, _ = squared_error.topk(pixel_count, dim=1)
    return hardest.mean()


def lpips_loss(prediction: torch.Tensor, target: torch.Tensor, lpips_model) -> torch.Tensor:
    """Mean LPIPS perceptual distance between prediction and target.

    LPIPS expects 3-channel input in [-1, 1]; the single grayscale channel
    is replicated and rescaled from the model's [0, 1] range to match.

    Args:
        prediction: Predicted images, shape [B, 1, H, W], values in [0, 1]
            (not yet clamped; this function clamps before rescaling).
        target: Ground-truth images, shape [B, 1, H, W], values in [0, 1].
        lpips_model: A loaded ``lpips.LPIPS`` network in eval mode.

    Returns:
        A scalar tensor: the batch-mean LPIPS distance.
    """
    prediction_lpips = prediction.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
    target_lpips = target.repeat(1, 3, 1, 1) * 2 - 1
    return lpips_model(prediction_lpips, target_lpips).mean()


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float = 0.1,
    data_range: float = 1.0,
    pixel: str = "charbonnier",
    pixel_weight: float = 1.0,
    charbonnier_epsilon: float = 1e-3,
    hard_pixel_weight: float = 0.0,
    hard_pixel_fraction: float = 0.05,
    lambda_gradient: float = 0.0,
    lambda_lpips: float = 0.0,
    lpips_model=None,
) -> torch.Tensor:
    """Combine a pixel-space loss, an SSIM penalty, and optional hard-pixel/gradient terms.

    Args:
        prediction: Predicted images, shape [B, C, H, W].
        target: Ground-truth images, shape [B, C, H, W], matching ``prediction``.
        lambda_ssim: Weight applied to the ``(1 - SSIM)`` penalty term.
        data_range: Value range of the pixel data, forwarded to
            :func:`ssim_per_image`.
        pixel: Which pixel loss to use: ``"charbonnier"`` or ``"l1"``.
        pixel_weight: Weight applied to the pixel loss term.
        charbonnier_epsilon: Smoothing constant forwarded to
            :func:`charbonnier_loss` when ``pixel == "charbonnier"``.
        hard_pixel_weight: Weight applied to :func:`hard_pixel_loss`. ``0.0``
            (the default) skips computing the term entirely, so existing
            configs that don't set it behave exactly as before.
        hard_pixel_fraction: Fraction of each image's highest-residual pixels
            to penalize, forwarded to :func:`hard_pixel_loss`.
        lambda_gradient: Weight applied to :func:`sobel_gradient_loss`. ``0.0``
            (the default) skips computing the term entirely, so existing
            configs that don't set it behave exactly as before.
        lambda_lpips: Weight applied to :func:`lpips_loss`. ``0.0`` (the
            default) skips computing the term entirely, so existing configs
            that don't set it behave exactly as before. Requires
            ``lpips_model`` when nonzero.
        lpips_model: A loaded ``lpips.LPIPS`` network in eval mode, forwarded
            to :func:`lpips_loss`. Only required when ``lambda_lpips`` is
            nonzero; the caller owns constructing/placing it on the correct
            device so this function doesn't import ``lpips`` or pay that cost
            when it's unused.

    Returns:
        A scalar tensor: ``pixel_weight * pixel_loss + lambda_ssim * (1 - mean
        SSIM)``, plus ``hard_pixel_weight * hard_pixel_loss(...)`` when
        ``hard_pixel_weight`` is nonzero, plus ``lambda_gradient *
        sobel_gradient_loss(...)`` when ``lambda_gradient`` is nonzero, plus
        ``lambda_lpips * lpips_loss(...)`` when ``lambda_lpips`` is nonzero.

    Raises:
        ValueError: If ``pixel`` is not one of the supported options, or if
            ``lambda_lpips`` is nonzero but ``lpips_model`` is ``None``.
    """

    if pixel == "charbonnier":
        pixel_loss = charbonnier_loss(
            prediction,
            target,
            epsilon=charbonnier_epsilon,
        )

    elif pixel == "l1":
        pixel_loss = F.l1_loss(prediction, target)

    else:
        raise ValueError(
            f"Unsupported pixel loss: {pixel}"
        )

    ssim_penalty = 1.0 - ssim_per_image(
        prediction,
        target,
        data_range=data_range,
    ).mean()

    total = (
        pixel_weight * pixel_loss
        + lambda_ssim * ssim_penalty
    )

    if hard_pixel_weight:
        total = total + hard_pixel_weight * hard_pixel_loss(prediction, target, fraction=hard_pixel_fraction)

    if lambda_gradient:
        total = total + lambda_gradient * sobel_gradient_loss(prediction, target)

    if lambda_lpips:
        if lpips_model is None:
            raise ValueError("lambda_lpips is nonzero but lpips_model was not provided")
        total = total + lambda_lpips * lpips_loss(prediction, target, lpips_model)

    return total
