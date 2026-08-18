#!/usr/bin/env python3
"""KLA submission entrypoint for offline batch restoration.

Usage:
    python run.py <input-dir> <output-dir>

The script intentionally accepts only NumPy arrays, as required by the
evaluator.  It loads the bundled checkpoint relative to this file, selects an
NVIDIA CUDA device when available, and writes one finite float32 ``.npy``
array per input file using the original filename.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from model import NAFNetSR


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "models" / "final_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore all grayscale .npy files in a directory.")
    parser.add_argument("input_dir", type=Path, help="Directory containing degraded .npy files.")
    parser.add_argument("output_dir", type=Path, help="Directory where restored .npy files will be written.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path (default: models/final_model.pt beside this script).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device; auto selects CUDA when it is available.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device("cuda" if requested == "auto" and cuda_available else requested if requested != "auto" else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int, float]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict and config entries.")
    config = checkpoint["config"]
    try:
        model_config = config["model"]
        scale_factor = int(model_config["scale_factor"])
        floating_point_scale = float(config["data"].get("floating_point_scale", 1.0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Checkpoint has an invalid model/data configuration.") from error
    if scale_factor < 1 or floating_point_scale <= 0:
        raise ValueError("Checkpoint scale factor and floating-point scale must be positive.")

    model = NAFNetSR(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, scale_factor, floating_point_scale


def read_input(path: Path) -> np.ndarray:
    """Read a finite grayscale array, accepting HxW or HxWx1 input layouts."""
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{path.name}: expected a grayscale (H, W) or (H, W, 1) array, got {array.shape}.")
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError(f"{path.name}: expected a numeric array, got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: input contains NaN or Inf values.")
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def restore(
    model: torch.nn.Module,
    source: np.ndarray,
    device: torch.device,
    scale_factor: int,
    floating_point_scale: float,
) -> np.ndarray:
    tensor = torch.from_numpy(source / floating_point_scale).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(tensor)
    output = prediction.squeeze(0).squeeze(0).float().cpu().numpy()
    expected_shape = (source.shape[0] * scale_factor, source.shape[1] * scale_factor)
    if output.shape != expected_shape:
        raise RuntimeError(f"Model produced {output.shape}; expected {expected_shape}.")
    # Guarantee the evaluator's numeric contract at the serialization boundary.
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    input_files = sorted(
        (path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".npy"),
        key=lambda path: path.name,
    )
    if not input_files:
        raise ValueError(f"No .npy files found in {args.input_dir}")

    device = select_device(args.device)
    model, scale_factor, floating_point_scale = load_model(args.checkpoint.resolve(), device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in input_files:
        restored = restore(model, read_input(input_path), device, scale_factor, floating_point_scale)
        np.save(args.output_dir / input_path.name, restored, allow_pickle=False)

    print(f"Restored {len(input_files)} files on {device} into {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
