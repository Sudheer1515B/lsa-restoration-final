"""Shape-contract tests for the active NAFNet-SR architecture."""

import unittest

import torch

from model import NAFNetSR


class NAFNetSRTests(unittest.TestCase):
    """Verify NAFNet-SR handles padding while preserving exact x2 output size."""
    def test_supports_odd_spatial_sizes_and_x2_output(self):
        """Verify internal padding is removed and odd inputs still produce exact x2 output."""
        model = NAFNetSR(width=8, enc_blocks=(1, 1), middle_blocks=1, dec_blocks=(1, 1))
        output = model(torch.zeros(2, 1, 127, 125))
        self.assertEqual(tuple(output.shape), (2, 1, 254, 250))
