"""Behavioral tests for PairedRestorationDataset's crop-position selection.

Covers the default uniform-random crop (and that it is unchanged by adding
texture-aware sampling as an opt-in), the texture-aware 50/30/20 mixture
policy itself, and an unmocked end-to-end smoke test through real files.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dataset import PairedRestorationDataset, find_pairs


def _make_dataset(**kwargs) -> PairedRestorationDataset:
    """Build a dataset instance with no pairs, only to exercise the crop helpers directly."""
    return PairedRestorationDataset(pairs=[], floating_point_scale=1.0, **kwargs)


class DefaultCropPositionTests(unittest.TestCase):
    """Verify texture_aware_sampling=False (the default) behaves exactly as before."""

    def test_uses_a_single_uniform_random_draw_per_axis(self):
        """Require exactly the original two random.randint(0, source_dim - crop) calls."""
        dataset = _make_dataset()
        source = torch.zeros(1, 8, 8)
        with patch("dataset.random.randint", side_effect=[3, 5]) as mock_randint:
            position = dataset._choose_crop_position(source, source_h=8, source_w=8, crop=4)
        self.assertEqual(position, (3, 5))
        self.assertEqual(mock_randint.call_args_list, [((0, 4),), ((0, 4),)])

    def test_crop_keeps_source_and_target_spatially_aligned(self):
        """The GT crop must start at exactly 2x the chosen LR crop's (top, left)."""
        dataset = _make_dataset(crop_size_lr=4)
        source = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
        target = torch.arange(256, dtype=torch.float32).reshape(1, 16, 16)
        with patch("dataset.random.randint", side_effect=[3, 1]):
            cropped_source, cropped_target = dataset._crop(source, target)
        self.assertTrue(torch.equal(cropped_source, source[:, 3:7, 1:5]))
        self.assertTrue(torch.equal(cropped_target, target[:, 6:14, 2:10]))


class TextureAwareCropPositionTests(unittest.TestCase):
    """Verify the 50% random / 30% high-detail / 20% medium-detail mixture policy."""

    def test_low_policy_draw_falls_back_to_plain_uniform_random(self):
        """A policy draw below 0.5 must behave identically to the default (no candidate scoring)."""
        dataset = _make_dataset(texture_aware_sampling=True)
        source = torch.zeros(1, 8, 8)
        with patch("dataset.random.random", return_value=0.1), patch("dataset.random.randint", side_effect=[2, 6]) as mock_randint:
            position = dataset._choose_crop_position(source, source_h=8, source_w=8, crop=4)
        self.assertEqual(position, (2, 6))
        self.assertEqual(mock_randint.call_count, 2)

    def test_high_policy_draw_picks_the_highest_detail_candidate(self):
        """A policy draw in [0.5, 0.8) must pick the candidate with the largest pixel std."""
        dataset = _make_dataset(texture_aware_sampling=True, texture_candidate_count=3)
        source = torch.zeros(1, 8, 8)
        low, mid, high = (0, 0), (0, 4), (4, 0)
        source[:, mid[0] : mid[0] + 2, mid[1] : mid[1] + 2] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        source[:, high[0] : high[0] + 2, high[1] : high[1] + 2] = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
        # low stays the all-zero background (std 0); mid and high are drawn as candidates too.
        with patch("dataset.random.random", return_value=0.6), patch(
            "dataset.random.randint",
            side_effect=[low[0], low[1], mid[0], mid[1], high[0], high[1]],
        ):
            position = dataset._choose_crop_position(source, source_h=8, source_w=8, crop=2)
        self.assertEqual(position, high)

    def test_medium_policy_draw_picks_the_median_ranked_candidate(self):
        """A policy draw in [0.8, 1.0) must pick the middle candidate by detail score, not the best."""
        dataset = _make_dataset(texture_aware_sampling=True, texture_candidate_count=3)
        source = torch.zeros(1, 8, 8)
        low, mid, high = (0, 0), (0, 4), (4, 0)
        source[:, mid[0] : mid[0] + 2, mid[1] : mid[1] + 2] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        source[:, high[0] : high[0] + 2, high[1] : high[1] + 2] = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
        with patch("dataset.random.random", return_value=0.9), patch(
            "dataset.random.randint",
            side_effect=[low[0], low[1], mid[0], mid[1], high[0], high[1]],
        ):
            position = dataset._choose_crop_position(source, source_h=8, source_w=8, crop=2)
        self.assertEqual(position, mid)


class TextureAwareEndToEndTests(unittest.TestCase):
    """Unmocked smoke test: real files through __getitem__ with texture-aware sampling on."""

    def test_getitem_produces_correctly_shaped_aligned_crops(self):
        """Repeated __getitem__ calls must stay well-formed across the full random policy mixture."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            degraded, ground_truth = root / "degraded", root / "ground_truth"
            degraded.mkdir()
            ground_truth.mkdir()
            rng = np.random.default_rng(0)
            np.save(degraded / "000000.npy", rng.standard_normal((16, 16)).astype(np.float32))
            np.save(ground_truth / "000000.npy", rng.standard_normal((32, 32)).astype(np.float32))

            pairs = find_pairs(degraded, ground_truth)
            dataset = PairedRestorationDataset(
                pairs,
                floating_point_scale=1.0,
                training=True,
                crop_size_lr=8,
                texture_aware_sampling=True,
                texture_candidate_count=4,
            )
            for _ in range(20):
                source_tensor, target_tensor = dataset[0]
                self.assertEqual(tuple(source_tensor.shape), (1, 8, 8))
                self.assertEqual(tuple(target_tensor.shape), (1, 16, 16))


if __name__ == "__main__":
    unittest.main()
