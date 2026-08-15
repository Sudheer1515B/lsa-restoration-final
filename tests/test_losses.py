"""Contract tests for the active reconstruction loss, including the hard-pixel and gradient terms."""

import unittest

import torch

from losses import hard_pixel_loss, reconstruction_loss, sobel_gradient_loss


class HardPixelLossTests(unittest.TestCase):
    """Verify hard_pixel_loss targets sparse, large-magnitude residuals correctly."""

    def test_zero_for_identical_images(self):
        """Require a perfect prediction to score exactly zero."""
        target = torch.rand(2, 1, 8, 8)
        self.assertAlmostEqual(float(hard_pixel_loss(target, target)), 0.0, places=6)

    def test_reacts_to_a_sparse_outlier_more_than_mean_squared_error_would_suggest(self):
        """A single large-magnitude pixel error should dominate a small top-fraction selection."""
        target = torch.zeros(1, 1, 10, 10)
        prediction = torch.zeros(1, 1, 10, 10)
        prediction[0, 0, 0, 0] = 2.0  # one severe outlier pixel among 100
        # Selecting only the top 1% (a single pixel) isolates that outlier's squared error.
        loss = hard_pixel_loss(prediction, target, fraction=0.01)
        self.assertAlmostEqual(float(loss), 4.0, places=5)

    def test_rejects_invalid_fraction(self):
        """Require fraction to be in (0, 1]."""
        target = torch.zeros(1, 1, 4, 4)
        with self.assertRaises(ValueError):
            hard_pixel_loss(target, target, fraction=0.0)
        with self.assertRaises(ValueError):
            hard_pixel_loss(target, target, fraction=1.5)

    def test_supports_backward(self):
        """Require the loss to be differentiable end to end."""
        prediction = torch.rand(2, 1, 8, 8, requires_grad=True)
        target = torch.rand(2, 1, 8, 8)
        hard_pixel_loss(prediction, target).backward()
        self.assertIsNotNone(prediction.grad)


class ReconstructionLossHardPixelIntegrationTests(unittest.TestCase):
    """Verify reconstruction_loss wires the optional hard-pixel term in correctly."""

    def test_default_weight_matches_pre_existing_behavior(self):
        """A zero (default) hard_pixel_weight must not change the loss value."""
        prediction = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        baseline = reconstruction_loss(prediction, target, pixel="charbonnier", lambda_ssim=0.2)
        with_default = reconstruction_loss(prediction, target, pixel="charbonnier", lambda_ssim=0.2, hard_pixel_weight=0.0)
        self.assertEqual(float(baseline), float(with_default))

    def test_nonzero_weight_penalizes_a_sparse_outlier_that_charbonnier_alone_barely_notices(self):
        """A sparse severe residual should move the loss much more with the hard-pixel term than without it."""
        pixel_count = 32 * 32
        target = torch.full((1, 1, 32, 32), 0.5)
        prediction = target.clone()
        prediction[0, 0, 0, 0] = target[0, 0, 0, 0] + 2.0  # one severe outlier pixel among 1024

        without_hard_term = float(reconstruction_loss(prediction, target, pixel="charbonnier", lambda_ssim=0.0, hard_pixel_weight=0.0))
        # fraction chosen so exactly the single outlier pixel is selected (round(1024 * fraction) == 1).
        with_hard_term = float(reconstruction_loss(prediction, target, pixel="charbonnier", lambda_ssim=0.0, hard_pixel_weight=1.0, hard_pixel_fraction=1.0 / pixel_count))

        # Charbonnier's per-pixel gradient saturates for large residuals, so a single
        # outlier among 1024 pixels barely moves the plain pixel-loss average (well
        # under 0.01 here); the hard-pixel term (unbounded squared error over just
        # that one outlier, exactly 2.0**2 == 4.0) should dominate once added.
        self.assertLess(without_hard_term, 0.01)
        self.assertGreater(with_hard_term - without_hard_term, 3.0)


class SobelGradientLossTests(unittest.TestCase):
    """Verify sobel_gradient_loss compares image derivatives, not raw pixel values."""

    def test_zero_for_identical_images(self):
        """Require a perfect prediction to score exactly zero."""
        target = torch.rand(2, 1, 8, 8)
        self.assertAlmostEqual(float(sobel_gradient_loss(target, target)), 0.0, places=6)

    def test_rejects_multi_channel_input(self):
        """Require single-channel [B, 1, H, W] input, matching the model's output contract."""
        target = torch.zeros(1, 3, 8, 8)
        with self.assertRaises(ValueError):
            sobel_gradient_loss(target, target)

    def test_reacts_more_to_a_moved_edge_than_to_a_uniform_brightness_shift(self):
        """Relocating an edge should score worse than an equally-sized constant offset.

        A true constant shift has zero interior Sobel gradient (the operator is a
        finite difference), but `padding=1` zero-pads the border, so a whole-image
        offset still perturbs the outermost ring slightly. That's expected boundary
        behavior, not something this test asserts away — instead it checks the
        qualitative property the loss exists for: moving where an edge sits should
        cost more than shifting overall brightness by a similar amount.
        """
        target = torch.zeros(1, 1, 10, 10)
        target[:, :, :, 5:] = 1.0  # a vertical step edge down the middle

        uniform_offset = target + 0.3
        uniform_offset_loss = float(sobel_gradient_loss(uniform_offset, target))

        shifted_edge = torch.zeros(1, 1, 10, 10)
        shifted_edge[:, :, :, 6:] = 1.0  # same edge, moved one column over
        shifted_edge_loss = float(sobel_gradient_loss(shifted_edge, target))

        self.assertGreater(shifted_edge_loss, uniform_offset_loss)

    def test_supports_backward(self):
        """Require the loss to be differentiable end to end."""
        prediction = torch.rand(2, 1, 8, 8, requires_grad=True)
        target = torch.rand(2, 1, 8, 8)
        sobel_gradient_loss(prediction, target).backward()
        self.assertIsNotNone(prediction.grad)


class ReconstructionLossGradientIntegrationTests(unittest.TestCase):
    """Verify reconstruction_loss wires the optional Sobel gradient term in correctly."""

    def test_default_weight_matches_pre_existing_behavior(self):
        """A zero (default) lambda_gradient must not change the loss value."""
        prediction = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        baseline = reconstruction_loss(prediction, target, pixel="charbonnier", lambda_ssim=0.2)
        with_default = reconstruction_loss(prediction, target, pixel="charbonnier", lambda_ssim=0.2, lambda_gradient=0.0)
        self.assertEqual(float(baseline), float(with_default))

    def test_nonzero_weight_penalizes_an_edge_mismatch_that_pixel_loss_alone_barely_notices(self):
        """A shifted edge should move the loss much more with the gradient term than without it."""
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, :, 8:] = 1.0  # a vertical step edge down the middle

        prediction = torch.zeros(1, 1, 16, 16)
        prediction[:, :, :, 9:] = 1.0  # same edge, shifted one column over (one column of pixels differs)

        without_gradient_term = float(reconstruction_loss(prediction, target, pixel="charbonnier", lambda_ssim=0.0, lambda_gradient=0.0))
        with_gradient_term = float(reconstruction_loss(prediction, target, pixel="charbonnier", lambda_ssim=0.0, lambda_gradient=1.0))

        # Only 1 of 16 columns differs, so the plain pixel loss barely moves; the
        # gradient term reacts directly to the edge displacement and should dominate.
        self.assertLess(without_gradient_term, 0.1)
        self.assertGreater(with_gradient_term - without_gradient_term, 0.2)


if __name__ == "__main__":
    unittest.main()
