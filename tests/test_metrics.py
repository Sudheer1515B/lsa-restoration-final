"""Contract tests for the active per-image metrics, including severity bucketing."""

import unittest

from metrics import bucket_by_severity


def _record(sample_id: str, psnr: float, ssim: float, noise_std: float) -> dict[str, float | str]:
    """Build one per-image validation record for the tests below."""
    return {"sample_id": sample_id, "psnr": psnr, "ssim": ssim, "noise_std": noise_std}


class BucketBySeverityTests(unittest.TestCase):
    """Verify records are grouped ascending by noise_std with correct per-bucket means."""

    def test_splits_into_equal_count_ascending_buckets(self):
        """Six records into three buckets should give two low/mid/high-noise pairs."""
        records = [
            _record("a", psnr=30.0, ssim=0.9, noise_std=0.01),
            _record("b", psnr=28.0, ssim=0.85, noise_std=0.02),
            _record("c", psnr=20.0, ssim=0.6, noise_std=0.10),
            _record("d", psnr=18.0, ssim=0.55, noise_std=0.11),
            _record("e", psnr=10.0, ssim=0.3, noise_std=0.40),
            _record("f", psnr=8.0, ssim=0.25, noise_std=0.45),
        ]
        summary = bucket_by_severity(records, num_buckets=3)
        self.assertEqual(len(summary), 3)
        self.assertEqual([group["count"] for group in summary], [2, 2, 2])
        self.assertAlmostEqual(summary[0]["mean_psnr"], 29.0)
        self.assertAlmostEqual(summary[-1]["mean_psnr"], 9.0)
        # The highest-severity bucket should have both the worst mean PSNR/SSIM
        # and the highest noise_std range, confirming ascending ordering.
        self.assertLess(summary[-1]["mean_psnr"], summary[0]["mean_psnr"])
        self.assertGreater(summary[-1]["noise_std_min"], summary[0]["noise_std_max"])

    def test_clamps_bucket_count_to_available_records(self):
        """A tiny validation set (e.g. an overfit/smoke run) must not raise."""
        records = [_record("a", 20.0, 0.5, 0.1), _record("b", 22.0, 0.6, 0.2)]
        summary = bucket_by_severity(records, num_buckets=3)
        self.assertEqual(len(summary), 2)

    def test_rejects_empty_records(self):
        """Require at least one record to bucket."""
        with self.assertRaises(ValueError):
            bucket_by_severity([], num_buckets=3)

    def test_rejects_non_positive_bucket_count(self):
        """Require num_buckets to be a positive integer."""
        with self.assertRaises(ValueError):
            bucket_by_severity([_record("a", 20.0, 0.5, 0.1)], num_buckets=0)


if __name__ == "__main__":
    unittest.main()
