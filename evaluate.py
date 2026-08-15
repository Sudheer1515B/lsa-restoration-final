"""Evaluate a NAFNet-SR checkpoint on a directory of paired grayscale images.

Entrypoint for scoring a trained checkpoint against a directory of
genuinely paired degraded/ground_truth grayscale images (i.e. real
low-resolution/high-resolution pairs, not synthetically degraded
ground truth). It runs the model over every pair, computes PSNR and
SSIM (and optionally LPIPS, if the ``lpips`` package and its weights
are available), measures inference time, and prints an aggregate
summary. It can optionally write a per-image CSV report.

Usage:
    python evaluate.py --checkpoint weights/final_model.pt --data data/val \\
        [--degraded-dir DIR] [--gt-dir DIR] [--device auto|cuda|mps|cpu] \\
        [--batch-size N] [--amp] [--output metrics.csv] [--no-lpips]

By default, degraded/ground-truth images are read from ``<data>/degraded``
and ``<data>/ground_truth``; ``--degraded-dir``/``--gt-dir`` override these.
This script is the model-quality evaluation counterpart to ``infer.py``
(which only performs restoration and does not require paired ground truth).
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
import time

import torch

from dataset import PairedRestorationDataset, ShapeBatchSampler, deterministic_split, find_pairs
from infer import D4_TRANSFORMS, apply_d4, invert_d4
from metrics import metric_per_image
from model import NAFNetSR
from runtime import select_device, synchronize


def parse_args() -> argparse.Namespace:
    """Parse paired-directory evaluation and optional reporting arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments controlling the checkpoint,
        data locations, device, batching, AMP, optional CSV output path,
        and whether to skip LPIPS.
    """
    parser = argparse.ArgumentParser(description="Evaluate PSNR, SSIM, optional LPIPS, and inference time.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None, help="Directory containing degraded/ and ground_truth/ subdirectories. Ignored if --val-split is set.")
    parser.add_argument(
        "--val-split",
        action="store_true",
        help=(
            "Evaluate on the genuine held-out validation split the checkpoint was "
            "trained with, instead of --data. Reconstructs it deterministically "
            "from the checkpoint's own embedded config (paths.train_degraded_dir, "
            "paths.train_ground_truth_dir, data.validation_fraction, seed) using "
            "the same dataset.deterministic_split() call train.py used, so it is "
            "guaranteed to be images the model never trained on and guaranteed to "
            "be genuinely paired -- unlike a hand-built directory, which can "
            "silently contain mismatched degraded/ground_truth pairs (see "
            "docs/methodology.md for the evaluation protocol)."
        ),
    )
    parser.add_argument("--degraded-dir", default=None)
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default=None, help="Optional per-image CSV output path.")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--tta", action="store_true", help="D4 test-time self-ensemble (see infer.py --tta); measures the effect on PSNR/SSIM/LPIPS before deciding whether to enable it for submission.")
    return parser.parse_args()


def load_lpips(device: torch.device):
    """Load optional LPIPS safely; primary metric evaluation remains available offline.

    Args:
        device: Device to move the LPIPS network to.

    Returns:
        A ``lpips.LPIPS`` model in eval mode on ``device``, or ``None`` if
        the ``lpips`` package is not installed or its pretrained weights
        cannot be fetched (e.g. no network access). LPIPS is treated as a
        best-effort extra metric so evaluation can still run offline and
        report PSNR/SSIM even when LPIPS is unavailable.
    """
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as error:  # LPIPS is optional and may lack cached weights offline.
        print(f"LPIPS unavailable ({error}); reporting PSNR/SSIM only.")
        return None


def main() -> None:
    """Evaluate every true pair and optionally write a per-image CSV report.

    Loads the checkpoint (using its embedded config), builds the matching
    NAFNet-SR model, discovers degraded/ground-truth pairs, runs inference
    over them in shape-grouped batches, computes per-image PSNR/SSIM (and
    LPIPS if available), times inference, prints an aggregate summary, and
    writes a per-image CSV if ``--output`` was given.
    """
    args = parse_args()
    if not args.val_split and not args.data:
        raise ValueError("Either --data <dir> or --val-split is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise ValueError("Checkpoint has no embedded config")
    config = checkpoint["config"]
    if args.val_split:
        train_cfg = config.get("paths", {})
        if not train_cfg.get("train_degraded_dir") or not train_cfg.get("train_ground_truth_dir"):
            raise ValueError("--val-split requires the checkpoint's config to embed paths.train_degraded_dir/train_ground_truth_dir")
        all_pairs = find_pairs(train_cfg["train_degraded_dir"], train_cfg["train_ground_truth_dir"], required_scale=int(config["data"]["scale_factor"]))
        _, pairs = deterministic_split(all_pairs, float(config["data"]["validation_fraction"]), int(config["seed"]))
        print(f"--val-split: reconstructed {len(pairs)} held-out pairs (of {len(all_pairs)} total) using seed={config['seed']}, validation_fraction={config['data']['validation_fraction']}")
    else:
        root = Path(args.data)
        degraded_dir = Path(args.degraded_dir) if args.degraded_dir else root / "degraded"
        gt_dir = Path(args.gt_dir) if args.gt_dir else root / "ground_truth"
        pairs = find_pairs(degraded_dir, gt_dir, required_scale=int(config["data"]["scale_factor"]))
    dataset = PairedRestorationDataset(pairs, config["data"].get("floating_point_scale"), return_metadata=True)
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=ShapeBatchSampler(dataset, args.batch_size, False), pin_memory=device.type == "cuda")
    model = NAFNetSR(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    lpips_model = None if args.no_lpips else load_lpips(device)
    data_range = float(config["data"].get("metric_data_range", 1.0))
    rows: list[dict[str, float | str | None]] = []
    elapsed = 0.0
    with torch.no_grad():
        for source, target, sample_ids in loader:
            source = source.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            synchronize(device)
            start = time.perf_counter()
            context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with context:
                if args.tta:
                    accumulator = None
                    for flip, k in D4_TRANSFORMS:
                        variant = invert_d4(model(apply_d4(source, flip, k)), flip, k)
                        accumulator = variant if accumulator is None else accumulator + variant
                    prediction = accumulator / len(D4_TRANSFORMS)
                else:
                    prediction = model(source)
            synchronize(device)
            elapsed += time.perf_counter() - start
            psnr_values, ssim_values = metric_per_image(prediction.float(), target.float(), data_range=data_range)
            lpips_values = None
            if lpips_model:
                # LPIPS expects 3-channel input in [-1, 1]; replicate the grayscale
                # channel and rescale from the model's [0, 1] range.
                lpips_values = lpips_model(prediction.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1, target.repeat(1, 3, 1, 1) * 2 - 1).flatten()
            for index, sample_id in enumerate(sample_ids):
                rows.append({
                    "sample_id": sample_id,
                    "psnr": float(psnr_values[index].cpu()),
                    "ssim": float(ssim_values[index].cpu()),
                    "lpips": float(lpips_values[index].cpu()) if lpips_values is not None else None,
                })
    mean_psnr = sum(float(row["psnr"]) for row in rows) / len(rows)
    mean_ssim = sum(float(row["ssim"]) for row in rows) / len(rows)
    lpips_rows = [float(row["lpips"]) for row in rows if row["lpips"] is not None]
    mean_lpips = sum(lpips_rows) / len(lpips_rows) if lpips_rows else None
    average_ms = elapsed * 1000.0 / len(rows)
    print(f"Images: {len(rows)}")
    print(f"PSNR: {mean_psnr:.2f} dB")
    print(f"SSIM: {mean_ssim:.4f}")
    print(f"LPIPS: {mean_lpips:.4f}" if mean_lpips is not None else "LPIPS: N/A")
    print(f"Average inference time: {average_ms:.2f} ms")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "psnr", "ssim", "lpips"))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-image metrics: {output}")


if __name__ == "__main__":
    main()
