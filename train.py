"""Train the portable NAFNet-SR x2 semiconductor restoration baseline.

This is the main, actively maintained training entry point for the project.
It trains a NAFNet-SR model that takes 128x128 noisy grayscale semiconductor
images and restores/upsamples them to clean 256x256 images.

Invocation (CLI):
    python train.py --config configs/baseline.yaml [--resume PATH]
        [--device {auto,cuda,mps,cpu}] [--amp] [--epochs N]
        [--max-train-samples N] [--max-val-samples N]
        [--overfit-samples N]

    See ``parse_args`` below for the full set of flags. Most run behavior
    (dataset paths, model architecture, optimizer, loss weighting, epoch
    count, batch size, etc.) is driven by the YAML file passed via
    ``--config``; CLI flags only override a small number of values for
    smoke-testing and resuming interrupted runs.

Role in the pipeline:
    ``train.py`` is the script that actually produces the checkpoints
    consumed by ``infer.py`` / ``evaluate.py``. Per epoch it:
      1. Trains one pass over the training split.
      2. Evaluates on the validation split, computing PSNR/SSIM per image.
      3. Writes CSV history, refreshes a learning-curve figure, and saves
         labeled before/after/ground-truth image panels for the best and
         worst five validation samples.
      4. Persists ``latest.pth``, and updates ``best_psnr.pth`` /
         ``best_ssim.pth`` whenever validation metrics improve.
    At the end of training it writes a Markdown experiment summary
    referencing all produced artifacts.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from copy import deepcopy
import math
import os
import random
from pathlib import Path
import time
from typing import Any

_MATPLOTLIB_CACHE = Path(__file__).resolve().parent / "outputs" / ".matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import PairedRestorationDataset, ShapeBatchSampler, deterministic_split, deterministic_subset, find_pairs
from losses import reconstruction_loss
from metrics import bucket_by_severity, metric_per_image, parameter_count
from model import NAFNetSR
from runtime import select_device, synchronize


def parse_args() -> argparse.Namespace:
    """Parse reproducible training, resume, device, and diagnostic overrides.

    CLI surface:
        --config              Path to the YAML run configuration (data paths,
                               model, optimizer, loss, and epoch settings).
        --resume               Path to a checkpoint (``latest.pth``,
                               ``best_psnr.pth``, or ``best_ssim.pth``) to
                               resume training from.
        --device               Force a specific accelerator, or ``auto`` to
                               pick the best available (CUDA > MPS > CPU).
        --amp                  Request CUDA mixed-precision training (no-op
                               off CUDA).
        --epochs               Override the epoch count from the config.
        --max-train-samples    Cap training to a deterministic first-N
                                subset, for fast smoke tests.
        --max-val-samples      Cap validation the same way.
        --overfit-samples      Use the same tiny N-sample subset for both
                                training and validation, to sanity-check
                                that the model can memorize/overfit.

    Returns:
        The parsed ``argparse.Namespace`` of CLI options.
    """
    parser = argparse.ArgumentParser(description="Train a NAFNet-SR x2 restoration model.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--resume", default=None, help="Portable latest/best checkpoint to resume.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision when available.")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Deterministic first-N subset for smoke runs.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Deterministic first-N validation subset.")
    parser.add_argument("--overfit-samples", type=int, default=0, help="Train and validate on this tiny shared subset.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch generators for deterministic split setup.

    Args:
        seed: Integer seed applied to ``random``, ``numpy``, and
            ``torch`` (including CUDA generators if available).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: PairedRestorationDataset, batch_size: int, shuffle: bool, device: torch.device, workers: int) -> DataLoader:
    """Build a shape-aware loader with accelerator-specific transfer settings.

    Args:
        dataset: The paired restoration dataset to iterate.
        batch_size: Number of samples per batch (grouped via
            ``ShapeBatchSampler`` so images with matching shapes are batched
            together).
        shuffle: Whether batches should be shuffled each epoch.
        device: Target device; used only to decide whether to enable
            pinned-memory transfers (CUDA only).
        workers: Number of DataLoader worker processes. When greater than
            zero, workers are kept alive across epochs.

    Returns:
        A configured ``torch.utils.data.DataLoader``.
    """
    kwargs: dict[str, Any] = {
        "batch_sampler": ShapeBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def autocast_context(device: torch.device, enabled: bool):
    """Return CUDA FP16 autocast only when the caller explicitly enabled it.

    Args:
        device: The active training device.
        enabled: Whether AMP was requested and is applicable (CUDA only).

    Returns:
        A ``torch.autocast`` context manager when enabled, otherwise a
        ``contextlib.nullcontext`` that is a no-op on MPS/CPU.
    """
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def cpu_copy(value: Any) -> Any:
    """Recursively copy checkpoint values onto CPU for portable serialization.

    Tensors nested inside model/optimizer/scheduler state dicts may live on
    CUDA or MPS. Moving everything to CPU before ``torch.save`` keeps the
    resulting checkpoint file loadable on any device (e.g. training on a
    CUDA machine, then resuming or evaluating on a Mac with MPS).

    Args:
        value: Any checkpoint-serializable value: a tensor, or a
            dict/list/tuple that may contain tensors nested arbitrarily
            deep, or a plain Python value.

    Returns:
        The same structure with every tensor detached and copied to CPU;
        non-tensor leaves are deep-copied.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return deepcopy(value)


def checkpoint_payload(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_psnr: float,
    best_ssim: float,
    config: dict,
    run_metadata: dict[str, object],
) -> dict:
    """Build a self-contained, device-portable checkpoint dictionary.

    Args:
        epoch: The last completed epoch number.
        model: The model whose ``state_dict`` will be saved.
        optimizer: The optimizer whose ``state_dict`` will be saved.
        scheduler: The LR scheduler whose ``state_dict`` will be saved.
        best_psnr: Best validation PSNR observed so far in this run.
        best_ssim: Best validation SSIM observed so far in this run.
        config: The full run configuration (as loaded from YAML), stored
            for provenance/reproducibility.
        run_metadata: Dataset split bookkeeping (sample IDs, split
            description, config path) so a resumed run can reconstruct the
            exact same train/validation split.

    Returns:
        A plain dict ready to pass to ``torch.save``, with all tensors
        moved to CPU via ``cpu_copy``.
    """
    return {
        "epoch": epoch,
        "model_state_dict": cpu_copy(model.state_dict()),
        "optimizer_state_dict": cpu_copy(optimizer.state_dict()),
        "scheduler_state_dict": cpu_copy(scheduler.state_dict()),
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "config": config,
        "run_metadata": run_metadata,
    }


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move loaded optimizer tensors to the active device after a CPU restore.

    ``optimizer.load_state_dict`` restores tensors (e.g. Adam moment
    buffers) on whatever device they were saved from, i.e. CPU, per
    ``cpu_copy``. This mutates them in place onto the current training
    device so subsequent optimizer steps don't hit a device mismatch.

    Args:
        optimizer: The optimizer whose state was just loaded from a
            checkpoint.
        device: The device training will continue on.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def to_display_image(tensor: torch.Tensor) -> Image.Image:
    """Convert one normalized tensor to an RGB preview without affecting model data.

    Args:
        tensor: A single-channel image tensor in ``[0, 1]`` range (any
            leading singleton dimensions are squeezed out). Detached and
            copied so the source tensor/graph is untouched.

    Returns:
        An RGB ``PIL.Image`` scaled to 8-bit for saving/plotting.
    """
    array = tensor.detach().float().squeeze().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L").convert("RGB")


def save_comparison(path: Path, sample_id: str, source: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, psnr: float, ssim: float) -> None:
    """Save a labeled input/restored/ground-truth visual comparison panel.

    Args:
        path: Output PNG path; parent directories are created as needed.
        sample_id: Identifier of the sample, used in the panel caption.
        source: The degraded/noisy low-resolution input tensor.
        prediction: The model's restored output tensor.
        target: The clean ground-truth tensor.
        psnr: Per-image PSNR to display in the caption.
        ssim: Per-image SSIM to display in the caption.
    """
    source_image = to_display_image(source)
    restored_image = to_display_image(prediction)
    target_image = to_display_image(target)
    source_image = source_image.resize(restored_image.size, Image.Resampling.BICUBIC)
    label_height, gap = 42, 4
    width, height = restored_image.size
    canvas = Image.new("RGB", (width * 3 + gap * 2, height + label_height), "white")
    for index, (label, image) in enumerate((("INPUT", source_image), ("RESTORED", restored_image), ("GROUND TRUTH", target_image))):
        x = index * (width + gap)
        canvas.paste(image, (x, label_height))
        ImageDraw.Draw(canvas).text((x + 3, 3), label, fill="black")
    ImageDraw.Draw(canvas).text((3, 21), f"{sample_id} | PSNR={psnr:.2f} dB | SSIM={ssim:.4f}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_ranked_examples(
    output_dir: Path,
    epoch: int,
    ranked: list[dict[str, float | str]],
    dataset: PairedRestorationDataset,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    data_range: float,
    name: str,
) -> None:
    """Re-run and save the five best or worst validation examples for an epoch.

    Args:
        output_dir: Root output directory; panels are written under
            ``output_dir/epoch_{epoch:03d}/{name}/``.
        epoch: Current epoch number, used in the output path.
        ranked: Validation records (with ``sample_id``) already sorted by
            the caller; only the first five are used.
        dataset: Validation dataset used to re-fetch the raw source/target
            tensors for each ranked sample by ID.
        model: The model to run inference with (switched to eval mode).
        device: Device to run inference on.
        use_amp: Whether to use CUDA autocast during inference.
        data_range: Value range of the pixel data, passed through to the
            metric computation.
        name: Subdirectory name, typically ``"best"`` or ``"worst"``.
    """
    by_id = {pair.sample_id: index for index, pair in enumerate(dataset.pairs)}
    model.eval()
    with torch.no_grad():
        for rank, record in enumerate(ranked[:5], start=1):
            sample_id = str(record["sample_id"])
            source, target = dataset[by_id[sample_id]][:2]
            with autocast_context(device, use_amp):
                prediction = model(source.unsqueeze(0).to(device))[0].cpu()
            psnr, ssim = metric_per_image(prediction.unsqueeze(0), target.unsqueeze(0), data_range=data_range)
            save_comparison(
                output_dir / f"epoch_{epoch:03d}" / name / f"{rank:02d}_{sample_id}.png",
                sample_id,
                source,
                prediction,
                target,
                float(psnr.item()),
                float(ssim.item()),
            )


def print_ranked(title: str, records: list[dict[str, float | str]]) -> None:
    """Print a compact terminal ranking of validation examples.

    Args:
        title: Heading printed above the ranking (e.g. "BEST 5 VALIDATION
            IMAGES").
        records: Validation records already sorted by the caller; only the
            first five are printed.
    """
    print(f"\n{title}")
    for index, record in enumerate(records[:5], start=1):
        print(f"{index}. {record['sample_id']}    PSNR={float(record['psnr']):.2f}    SSIM={float(record['ssim']):.4f}")


def print_severity_breakdown(summary: list[dict[str, float | int]]) -> None:
    """Print each noise-severity bucket's sample count and mean PSNR/SSIM.

    Args:
        summary: Buckets as returned by :func:`metrics.bucket_by_severity`,
            ordered from least to most severe.
    """
    print("\nVALIDATION BY NOISE SEVERITY (least to most corrupted)")
    for group in summary:
        print(
            f"Bucket {int(group['bucket']) + 1}/{len(summary)}  n={int(group['count']):>3}  "
            f"noise_std=[{group['noise_std_min']:.4f}, {group['noise_std_max']:.4f}]  "
            f"PSNR={group['mean_psnr']:.2f} dB  SSIM={group['mean_ssim']:.4f}"
        )


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write epoch-level training and validation history as stable CSV columns.

    Args:
        path: Destination CSV path (overwritten each call).
        rows: One dict per epoch with the fixed set of history fields. Rows
            loaded from a history file written before ``high_severity_psnr``/
            ``high_severity_ssim`` existed simply leave those cells blank.
    """
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "psnr",
        "ssim",
        "epoch_time",
        "learning_rate",
        "val_inference_ms",
        "high_severity_psnr",
        "high_severity_ssim",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_validation_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Persist all per-image validation PSNR, SSIM, and noise severity for an epoch.

    Args:
        path: Destination CSV path; parent directories are created as
            needed.
        rows: One dict per validation sample with ``sample_id``, ``psnr``,
            ``ssim``, and ``noise_std``; written sorted by ``sample_id`` for
            stable diffs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "psnr", "ssim", "noise_std"))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["sample_id"])))


def write_severity_breakdown(path: Path, summary: list[dict[str, float | int]]) -> None:
    """Persist one epoch's noise-severity bucket summary as a small CSV.

    Args:
        path: Destination CSV path; parent directories are created as
            needed.
        summary: Buckets as returned by :func:`metrics.bucket_by_severity`,
            written one row per bucket, least to most severe.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("bucket", "count", "noise_std_min", "noise_std_max", "mean_psnr", "mean_ssim"))
        writer.writeheader()
        writer.writerows(summary)


def update_curves(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Regenerate the compact loss, PSNR, and SSIM learning-curve figure.

    Args:
        path: Destination PNG path for the figure (overwritten each call).
        rows: Full epoch history (as produced by ``write_history``/
            ``load_history``); no-op if empty.
    """
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["psnr"] for row in rows], label="overall")
    axes[1].set(title="Validation PSNR", xlabel="Epoch", ylabel="dB")
    axes[2].plot(epochs, [row["ssim"] for row in rows], label="overall")
    axes[2].set(title="Validation SSIM", xlabel="Epoch")
    # Overlay the most-noise-corrupted third's mean PSNR/SSIM (see
    # metrics.bucket_by_severity) so a model that only improves on easy,
    # mildly-corrupted images doesn't look identical to one that's actually
    # fixing the hardest (e.g. speckle-heavy) cases. Older history rows
    # written before this tracking existed simply have no entry here.
    severity_rows = [row for row in rows if "high_severity_psnr" in row]
    if severity_rows:
        axes[1].plot([row["epoch"] for row in severity_rows], [row["high_severity_psnr"] for row in severity_rows], label="high-noise third", linestyle="--")
        axes[2].plot([row["epoch"] for row in severity_rows], [row["high_severity_ssim"] for row in severity_rows], label="high-noise third", linestyle="--")
        axes[1].legend()
        axes[2].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def load_history(path: Path) -> list[dict[str, float | int]]:
    """Load a prior CSV history so a resumed run keeps its complete record.

    Args:
        path: Path to a previously written history CSV.

    Returns:
        A list of per-epoch dicts (``epoch`` parsed as int, all other
        present fields as float), or an empty list if the file does not
        exist. Columns added after a row was written (e.g.
        ``high_severity_psnr``) are simply absent from that row's dict
        rather than raising, since older history files predate them.
    """
    if not path.exists():
        return []
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float | int] = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                elif value != "":
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def read_excluded_ids(path_value: str | None) -> set[str]:
    """Read optional sample IDs that must not enter training or validation.

    Args:
        path_value: Path to a text file with one sample ID per line, or
            ``None``/empty to skip exclusion entirely.

    Returns:
        A set of sample ID strings (empty if no path was given).

    Raises:
        FileNotFoundError: If a path was given but does not exist.
    """
    if not path_value:
        return set()
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Excluded-ID file does not exist: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    """Execute training, validation, checkpoint selection, and experiment reporting.

    This is the script's entry point (invoked when run as
    ``python train.py ...``). It performs, in order: config/checkpoint
    loading, device and AMP selection, dataset discovery and train/val
    splitting (including honoring a resumed run's original split), model/
    optimizer/scheduler construction, the epoch loop (train pass, validation
    pass, metric computation, checkpoint/history/curve/image-panel writes),
    and a final Markdown experiment summary. Takes no arguments and returns
    nothing; all configuration comes from CLI args and the YAML config file.
    """
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    set_seed(int(config["seed"]))
    device = select_device(args.device)
    requested_amp = bool(args.amp or config["training"].get("amp", False))
    use_amp = requested_amp and device.type == "cuda"
    if requested_amp and not use_amp:
        print("AMP requested but disabled: this baseline enables AMP only on CUDA.")
    print(f"Selected device: {device}")
    print(f"AMP: {'enabled (CUDA FP16)' if use_amp else 'disabled'}")

    data_cfg, paths_cfg = config["data"], config["paths"]
    scale_factor = int(data_cfg["scale_factor"])
    all_train_pairs = find_pairs(paths_cfg["train_degraded_dir"], paths_cfg["train_ground_truth_dir"], required_scale=scale_factor)
    excluded_ids = read_excluded_ids(paths_cfg.get("exclude_train_ids_file"))
    unknown_ids = excluded_ids - {pair.sample_id for pair in all_train_pairs}
    if unknown_ids:
        raise ValueError(f"Excluded IDs are absent from training pairs: {sorted(unknown_ids)[:5]}")
    train_pairs = [pair for pair in all_train_pairs if pair.sample_id not in excluded_ids]
    if paths_cfg.get("validation_degraded_dir") and paths_cfg.get("validation_ground_truth_dir"):
        val_pairs = find_pairs(paths_cfg["validation_degraded_dir"], paths_cfg["validation_ground_truth_dir"], required_scale=scale_factor)
        split_description = "explicit validation directories"
    else:
        train_pairs, val_pairs = deterministic_split(train_pairs, float(data_cfg["validation_fraction"]), int(config["seed"]))
        split_description = "deterministic validation split"
    if args.overfit_samples:
        if args.overfit_samples < 2:
            raise ValueError("--overfit-samples must be at least two")
        train_pairs = train_pairs[: args.overfit_samples]
        val_pairs = train_pairs.copy()
        split_description = f"shared {len(train_pairs)}-sample overfit split"
    else:
        train_pairs = deterministic_subset(train_pairs, args.max_train_samples, int(config["seed"]))
        val_pairs = deterministic_subset(val_pairs, args.max_val_samples, int(config["seed"]) + 1)
    # When resuming (and not overriding the split via CLI subset flags), rebuild the
    # exact train/validation split from the checkpoint's saved sample IDs rather than
    # recomputing it, so a resumed run never trains/validates on a different split
    # than the one the checkpoint's metrics and optimizer state were produced from.
    saved_metadata = resume_checkpoint.get("run_metadata") if resume_checkpoint else None
    if (
        saved_metadata
        and not args.overfit_samples
        and args.max_train_samples is None
        and args.max_val_samples is None
        and saved_metadata.get("train_sample_ids")
        and saved_metadata.get("validation_sample_ids")
    ):
        pair_by_id = {pair.sample_id: pair for pair in all_train_pairs}
        pair_by_id.update({pair.sample_id: pair for pair in val_pairs})
        try:
            train_pairs = [pair_by_id[sample_id] for sample_id in saved_metadata["train_sample_ids"]]
            val_pairs = [pair_by_id[sample_id] for sample_id in saved_metadata["validation_sample_ids"]]
        except KeyError as error:
            raise ValueError(f"Resume checkpoint references a missing data sample: {error}") from error
        split_description = str(saved_metadata.get("split_description", split_description))
    if not train_pairs or not val_pairs:
        raise ValueError("Training and validation each require at least one paired sample")
    print(f"Dataset split: {split_description}; train={len(train_pairs)}, validation={len(val_pairs)}")
    run_metadata: dict[str, object] = {
        "train_sample_ids": [pair.sample_id for pair in train_pairs],
        "validation_sample_ids": [pair.sample_id for pair in val_pairs],
        "split_description": split_description,
        "config_path": str(args.config),
    }

    train_dataset = PairedRestorationDataset(
        train_pairs,
        floating_point_scale=data_cfg.get("floating_point_scale"),
        training=True,
        crop_size_lr=data_cfg.get("crop_size_lr"),
        geometric_augment=bool(data_cfg.get("geometric_augment", True)) and not bool(args.overfit_samples),
        texture_aware_sampling=bool(data_cfg.get("texture_aware_sampling", False)),
        texture_candidate_count=int(data_cfg.get("texture_candidate_count", 8)),
    )
    val_dataset = PairedRestorationDataset(val_pairs, data_cfg.get("floating_point_scale"), return_metadata=True)
    batch_size, workers = int(config["training"]["batch_size"]), int(data_cfg.get("num_workers", 0))
    train_loader = make_loader(train_dataset, batch_size, True, device, workers)
    val_loader = make_loader(val_dataset, batch_size, False, device, workers)

    model = NAFNetSR(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"].get("weight_decay", 0.0)))
    epochs = int(args.epochs or config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(config["training"].get("min_learning_rate", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    data_range = float(data_cfg.get("metric_data_range", 1.0))

    loss_cfg = config["loss"]
    loss_kwargs = {
        "lambda_ssim": float(loss_cfg.get("lambda_ssim", 0.1)),
        "data_range": data_range,
        "pixel": str(loss_cfg.get("pixel", "charbonnier")),
        "pixel_weight": float(loss_cfg.get("pixel_weight", 1.0)),
        "charbonnier_epsilon": float(
            loss_cfg.get("charbonnier_epsilon", 1e-3)
        ),
        "hard_pixel_weight": float(loss_cfg.get("hard_pixel_weight", 0.0)),
        "hard_pixel_fraction": float(loss_cfg.get("hard_pixel_fraction", 0.05)),
        "lambda_gradient": float(loss_cfg.get("lambda_gradient", 0.0)),
        "lambda_lpips": float(loss_cfg.get("lambda_lpips", 0.0)),
    }
    if loss_kwargs["lambda_lpips"]:
        # Imported lazily so configs that never set lambda_lpips (i.e. every
        # config before this one) don't pay the import/network-weight-download
        # cost or require the lpips package at all.
        import lpips as lpips_package

        loss_kwargs["lpips_model"] = lpips_package.LPIPS(net="alex").to(device).eval()
        print(f"LPIPS loss term enabled: lambda_lpips={loss_kwargs['lambda_lpips']}")

    checkpoint_dir, output_dir = Path(paths_cfg["checkpoint_dir"]), Path(paths_cfg["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path, curves_path = output_dir / "training_history.csv", output_dir / "training_curves.png"
    start_epoch, best_psnr, best_ssim = 0, float("-inf"), float("-inf")
    history = load_history(history_path) if args.resume else []
    if args.resume:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Checkpoints are always saved with CPU tensors (see cpu_copy) for
        # portability across CUDA/MPS/CPU machines, so optimizer state must be
        # moved back onto the active device before the next optimizer step.
        move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # CosineAnnealingLR stores T_max inside its own state dict, so the line
        # above silently discards the T_max built from the *current* --epochs and
        # restores the horizon the original run was launched with. Resuming with a
        # larger budget then drives the cosine past its period, where it turns back
        # upward: extending a finished 50-epoch run to 100 mirrors the anneal in
        # reverse and finishes at the maximum LR instead of the minimum, leaving the
        # model permanently un-annealed. Re-point the schedule at the real budget and
        # seed the LR with the closed-form value for the resumed position, which is
        # what the recursive step() below expects as its starting point.
        if scheduler.T_max != max(epochs, 1):
            scheduler.T_max = max(epochs, 1)
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = scheduler.eta_min + (base_lr - scheduler.eta_min) * (1 + math.cos(math.pi * scheduler.last_epoch / scheduler.T_max)) / 2
            print(f"Rescaled LR schedule to the resumed budget: T_max={scheduler.T_max}, lr={optimizer.param_groups[0]['lr']:.3e}")
        start_epoch = int(checkpoint["epoch"])
        best_psnr, best_ssim = float(checkpoint["best_psnr"]), float(checkpoint["best_ssim"])
        print(f"Resumed {args.resume} at epoch {start_epoch}; best PSNR={best_psnr:.3f}, SSIM={best_ssim:.4f}")
    print(f"Model parameters: {parameter_count(model):,}")

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        for source, target in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} training", leave=False):
            source = source.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                prediction = model(source)
                loss = reconstruction_loss(
                    prediction,
                    target,
                    **loss_kwargs,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses: list[float] = []
        per_sample: list[dict[str, float | str]] = []
        validation_inference_seconds = 0.0
        with torch.no_grad():
            for source, target, sample_ids in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} validation", leave=False):
                source = source.to(device, non_blocking=device.type == "cuda")
                target = target.to(device, non_blocking=device.type == "cuda")
                synchronize(device)
                start = time.perf_counter()
                with autocast_context(device, use_amp):
                    prediction = model(source)
                    loss = reconstruction_loss(
                        prediction,
                        target,
                        **loss_kwargs,
                    )
                synchronize(device)
                validation_inference_seconds += time.perf_counter() - start
                psnr_values, ssim_values = metric_per_image(prediction.float(), target.float(), data_range=data_range)
                # Per-image std of the raw (unclipped) degraded input is a cheap severity
                # proxy: it's high for heavily corrupted images (e.g. strong speckle) and
                # low for mildly corrupted ones, without requiring any noise-type labels.
                noise_std_values = source.float().std(dim=(1, 2, 3))
                validation_losses.append(float(loss.detach().cpu()))
                per_sample.extend(
                    {"sample_id": sample_id, "psnr": float(psnr_value.cpu()), "ssim": float(ssim_value.cpu()), "noise_std": float(noise_std_value.cpu())}
                    for sample_id, psnr_value, ssim_value, noise_std_value in zip(sample_ids, psnr_values, ssim_values, noise_std_values)
                )
        scheduler.step()
        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(validation_losses))
        mean_psnr = float(np.mean([float(item["psnr"]) for item in per_sample]))
        mean_ssim = float(np.mean([float(item["ssim"]) for item in per_sample]))
        best_five, worst_five = sorted(per_sample, key=lambda item: float(item["psnr"]), reverse=True)[:5], sorted(per_sample, key=lambda item: float(item["psnr"]))[:5]
        severity_summary = bucket_by_severity(per_sample, num_buckets=3)
        write_validation_metrics(output_dir / f"epoch_{epoch:03d}" / "validation_metrics.csv", per_sample)
        write_severity_breakdown(output_dir / f"epoch_{epoch:03d}" / "severity_breakdown.csv", severity_summary)
        save_ranked_examples(output_dir, epoch, best_five, val_dataset, model, device, use_amp, data_range, "best")
        save_ranked_examples(output_dir, epoch, worst_five, val_dataset, model, device, use_amp, data_range, "worst")
        epoch_time = time.perf_counter() - epoch_start
        # The last bucket is the most-noise-corrupted third (buckets are ordered
        # ascending by noise_std); tracking it separately in history/curves surfaces
        # whether the model is actually improving on the hardest cases, not just
        # riding the easy majority to a better dataset-wide mean.
        high_severity = severity_summary[-1]
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "epoch_time": epoch_time,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "val_inference_ms": validation_inference_seconds * 1000.0 / len(per_sample),
            "high_severity_psnr": float(high_severity["mean_psnr"]),
            "high_severity_ssim": float(high_severity["mean_ssim"]),
        }
        history = [existing for existing in history if int(existing["epoch"]) != epoch] + [row]
        history.sort(key=lambda existing: int(existing["epoch"]))
        write_history(history_path, history)
        update_curves(curves_path, history)
        # Track best PSNR and best SSIM independently: the epoch with the highest
        # PSNR need not be the epoch with the highest SSIM, so each metric gets its
        # own checkpoint in addition to the always-overwritten "latest" checkpoint.
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(checkpoint_payload(epoch, model, optimizer, scheduler, best_psnr, best_ssim, config, run_metadata), checkpoint_dir / "best_psnr.pth")
        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            torch.save(checkpoint_payload(epoch, model, optimizer, scheduler, best_psnr, best_ssim, config, run_metadata), checkpoint_dir / "best_ssim.pth")
        torch.save(checkpoint_payload(epoch, model, optimizer, scheduler, best_psnr, best_ssim, config, run_metadata), checkpoint_dir / "latest.pth")
        print(
            f"\nEpoch {epoch}/{epochs}\n"
            f"Train Loss: {mean_train_loss:.5f}\n"
            f"Val Loss:   {mean_val_loss:.5f}\n"
            f"PSNR:       {mean_psnr:.2f} dB\n"
            f"SSIM:       {mean_ssim:.4f}\n"
            f"Best PSNR:  {best_psnr:.2f} dB\n"
            f"Best SSIM:  {best_ssim:.4f}\n"
            f"Epoch Time: {epoch_time:.1f}s\n"
            f"Val inference: {row['val_inference_ms']:.2f} ms/image"
        )
        print_ranked("BEST 5 VALIDATION IMAGES", best_five)
        print_ranked("WORST 5 VALIDATION IMAGES", worst_five)
        print_severity_breakdown(severity_summary)

    if history:
        best_psnr_row = max(history, key=lambda row: float(row["psnr"]))
        best_ssim_row = max(history, key=lambda row: float(row["ssim"]))
        final_row = history[-1]
        summary = [
            "# Experiment Summary",
            "",
            "## Dataset",
            f"- Training samples: {len(train_pairs)}",
            f"- Validation samples: {len(val_pairs)}",
            f"- Split: {split_description}",
            "",
            "## Model and run",
            f"- Parameters: {parameter_count(model):,}",
            f"- Device: {device}",
            f"- Batch size: {batch_size}",
            f"- Epochs completed: {final_row['epoch']}",
            f"- Learning rate at final epoch: {final_row['learning_rate']}",
            "",
            "## Metrics",
            f"- Best PSNR: {best_psnr_row['psnr']:.4f} dB (epoch {best_psnr_row['epoch']})",
            f"- Best SSIM: {best_ssim_row['ssim']:.4f} (epoch {best_ssim_row['epoch']})",
            f"- Final PSNR: {final_row['psnr']:.4f} dB",
            f"- Final SSIM: {final_row['ssim']:.4f}",
            # A final_row loaded from a history file predating severity tracking (e.g.
            # resuming a run with no new epochs to train) may not have these fields.
            f"- Final high-noise-third PSNR: {final_row['high_severity_psnr']:.4f} dB" if "high_severity_psnr" in final_row else "- Final high-noise-third PSNR: not tracked for this epoch",
            f"- Final high-noise-third SSIM: {final_row['high_severity_ssim']:.4f}" if "high_severity_ssim" in final_row else "- Final high-noise-third SSIM: not tracked for this epoch",
            f"- Validation inference time: {final_row['val_inference_ms']:.4f} ms/image",
            "",
            "## Artifacts",
            f"- Best PSNR checkpoint: `{checkpoint_dir / 'best_psnr.pth'}`",
            f"- Best SSIM checkpoint: `{checkpoint_dir / 'best_ssim.pth'}`",
            f"- Latest checkpoint: `{checkpoint_dir / 'latest.pth'}`",
            f"- History: `{history_path}`",
            f"- Curves: `{curves_path}`",
            f"- Latest best panels: `{output_dir / f'epoch_{int(final_row['epoch']):03d}' / 'best'}`",
            f"- Latest worst panels: `{output_dir / f'epoch_{int(final_row['epoch']):03d}' / 'worst'}`",
            "",
        ]
        (output_dir / "experiment_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nFinished. Latest checkpoint: {checkpoint_dir / 'latest.pth'}")


if __name__ == "__main__":
    main()
