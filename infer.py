"""Single-image, portable inference for a trained NAFNet-SR checkpoint.

Standalone entrypoint for restoring degraded grayscale images with a
trained NAFNet-SR checkpoint, independent of the training pipeline. It
supports two mutually exclusive modes:

- Single-image mode (``--input``/``--output``): restore one image file.
- Directory mode (``--input_dir``/``--output_dir``): restore every
  supported grayscale image found in a directory, writing each restored
  image alongside an ``inference_manifest.csv`` summarizing per-image
  shapes and runtimes. This is the entrypoint used to produce the
  restored outputs evaluated for submission.

Example invocations:
    python infer.py --checkpoint weights/final_model.pt --input a.png --output b.png
    python infer.py --checkpoint weights/final_model.pt --input_dir data/degraded --output_dir out/

The checkpoint embeds its own model configuration by default, so the
script can be run with only ``--checkpoint`` plus the input/output
arguments; ``--config`` overrides the embedded configuration if needed.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import yaml

from dataset import SUPPORTED_EXTENSIONS, image_to_tensor, list_images, read_grayscale
from model import NAFNetSR
from runtime import select_device, synchronize

# The 8 elements of the D4 dihedral group: an optional horizontal flip
# followed by a rotation of k * 90 degrees. The model is trained with
# random flip/rot90 augmentation (dataset.py:_augment), so it is
# approximately equivariant to all 8, making this a valid self-ensemble.
D4_TRANSFORMS = tuple((flip, k) for flip in (False, True) for k in range(4))


def apply_d4(tensor: torch.Tensor, flip: bool, k: int) -> torch.Tensor:
    """Apply one D4 transform (optional horizontal flip, then rot90 x k) to a [B,C,H,W] tensor."""
    if flip:
        tensor = torch.flip(tensor, dims=[-1])
    return torch.rot90(tensor, k, dims=[-2, -1])


def invert_d4(tensor: torch.Tensor, flip: bool, k: int) -> torch.Tensor:
    """Invert ``apply_d4``: undo the rotation, then the flip, restoring original orientation."""
    tensor = torch.rot90(tensor, -k, dims=[-2, -1])
    if flip:
        tensor = torch.flip(tensor, dims=[-1])
    return tensor


def parse_args() -> argparse.Namespace:
    """Parse single-image or evaluator-ready directory restoration arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments, including whichever of the
        single-image (``input``/``output``) or directory
        (``input_dir``/``output_dir``) mode arguments were supplied. Mode
        selection and validation happen later in ``main``.
    """
    parser = argparse.ArgumentParser(description="Restore one image or an evaluator-ready directory of grayscale images with NAFNet-SR.")
    parser.add_argument("--checkpoint", default="weights/final_model.pt", help="Final checkpoint path; defaults to the submission location.")
    parser.add_argument("--input", help="One degraded image (use with --output).")
    parser.add_argument("--output", help="One restored image path (use with --input).")
    parser.add_argument("--input_dir", "--input-dir", dest="input_dir", help="Directory of degraded test images.")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", help="Directory for restored test images and inference_manifest.csv.")
    parser.add_argument("--config", default=None, help="Optional config; checkpoint configuration is used by default.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--amp", action="store_true", help="Enable CUDA FP16 inference.")
    parser.add_argument("--output_extension", "--output-extension", dest="output_extension", default=None, help="Directory-mode output extension, e.g. .npy or .png. Defaults to each input's extension.")
    parser.add_argument("--tta", action="store_true", help="D4 test-time self-ensemble: average predictions over all 8 flip/rot90 variants of the input. ~8x slower; typically +0.1-0.3 dB and less noise-like artifact since the model was trained with flip/rot90 augmentation. Off by default so the documented submission command is unaffected.")
    return parser.parse_args()


def save_image(path: Path, tensor: torch.Tensor) -> None:
    """Serialize a restored tensor, clipping only at the final clean-image boundary.

    Args:
        path: Destination file path. A ``.npy`` extension saves the raw
            float32 array; any other extension saves an 8-bit grayscale PNG
            (or similar image format inferred from the extension).
        tensor: Restored image tensor (any leading singleton dimensions are
            squeezed out) in the model's [0, 1] floating-point range.

    The array is clipped to [0, 1] once here, at the final output boundary,
    rather than at each intermediate step. ``.npy`` output preserves full
    float32 precision for downstream numeric comparison, while image
    formats are quantized to 8-bit ([0, 255]) since that is what standard
    image viewers/codecs expect.
    """
    array = np.clip(tensor.detach().float().squeeze().cpu().numpy(), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, array.astype(np.float32), allow_pickle=False)
    else:
        Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L").save(path)


def load_model(checkpoint_path: str, config_path: str | None, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load a portable checkpoint and reconstruct its configured NAFNet-SR model.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint containing at least
            ``model_state_dict``, and optionally an embedded ``config``.
        config_path: Optional path to a YAML config to use instead of the
            checkpoint's embedded config. Required if the checkpoint has no
            embedded config.
        device: Device the reconstructed model is moved to.

    Returns:
        tuple[torch.nn.Module, dict]: The loaded model in eval mode, and the
        resolved configuration dictionary used to build it.

    Raises:
        ValueError: If no config is embedded in the checkpoint and
            ``config_path`` is not provided.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config_path:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("Checkpoint has no embedded config; pass --config")
    model = NAFNetSR(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def restore_one(model: torch.nn.Module, config: dict, source_path: Path, destination: Path, device: torch.device, use_amp: bool, tta: bool = False) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Restore one file, validate its x2 shape contract, and return timing metadata.

    Reads a grayscale source image, runs it through the model to produce a
    restored image, verifies the output matches the model's configured
    ``scale_factor`` (e.g. 128x128 -> 256x256), writes it to ``destination``
    via ``save_image``, and times the forward pass.

    Args:
        model: Loaded NAFNet-SR model in eval mode.
        config: Model/data configuration dict (as returned by ``load_model``);
            used for the input scaling convention and expected scale factor.
        source_path: Path to the degraded input image.
        destination: Path the restored image (or ``.npy`` array) is written to.
        device: Device to run inference on.
        use_amp: Whether to run the forward pass under CUDA FP16 autocast.
        tta: If True, run the D4 self-ensemble (8 forward passes over
            flip/rot90 variants of the input, averaged after inverting each
            variant's transform) instead of a single forward pass.

    Returns:
        tuple[tuple[int, int], tuple[int, int], float]: The source image's
        (height, width), the restored image's (height, width), and the
        wall-clock inference time in seconds (all 8 passes, if ``tta``).

    Raises:
        RuntimeError: If the restored output shape does not match the
            expected shape implied by ``config["model"]["scale_factor"]``.
    """
    source, _ = read_grayscale(source_path)
    source_tensor = image_to_tensor(source, config["data"].get("floating_point_scale")).unsqueeze(0).to(device)
    context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad(), context:
        if tta:
            accumulator = None
            for flip, k in D4_TRANSFORMS:
                prediction = invert_d4(model(apply_d4(source_tensor, flip, k)), flip, k)
                accumulator = prediction if accumulator is None else accumulator + prediction
            restored = accumulator / len(D4_TRANSFORMS)
        else:
            restored = model(source_tensor)
    synchronize(device)
    runtime = time.perf_counter() - start
    expected_shape = tuple(dimension * int(config["model"]["scale_factor"]) for dimension in source.shape)
    if tuple(restored.shape[-2:]) != expected_shape:
        raise RuntimeError(f"Model produced {tuple(restored.shape[-2:])}; expected {expected_shape}")
    save_image(destination, restored[0])
    return tuple(source.shape), tuple(restored.shape[-2:]), runtime


def main() -> None:
    """Dispatch validated single-image or directory inference without source edits.

    Parses CLI arguments, validates that exactly one of single-image or
    directory mode was requested with its required arguments, loads the
    model, and then either restores one image (single mode) or restores
    every supported image in a directory and writes an
    ``inference_manifest.csv`` summarizing per-image shapes and runtimes
    (directory mode). In directory mode, duplicate input filename stems are
    rejected up front since they would otherwise silently overwrite one
    another's output file.
    """
    args = parse_args()
    single_mode = bool(args.input or args.output)
    directory_mode = bool(args.input_dir or args.output_dir)
    if single_mode == directory_mode:
        raise SystemExit("Provide exactly one mode: --input/--output or --input_dir/--output_dir.")
    if single_mode and (not args.input or not args.output):
        raise SystemExit("Single-image mode requires both --input and --output.")
    if directory_mode and (not args.input_dir or not args.output_dir):
        raise SystemExit("Directory mode requires both --input_dir and --output_dir.")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Pass --checkpoint or place the final weights there.")
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    model, config = load_model(str(checkpoint_path), args.config, device)
    if args.tta:
        print("D4 test-time self-ensemble enabled (--tta): 8 forward passes per image.")
    if single_mode:
        input_path, output_path = Path(args.input), Path(args.output)
        source_shape, restored_shape, runtime = restore_one(model, config, input_path, output_path, device, use_amp, args.tta)
        print(f"Restored {source_shape} -> {restored_shape} on {device} in {runtime * 1000:.2f} ms: {output_path}")
        return

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    inputs = list_images(input_dir, SUPPORTED_EXTENSIONS)
    if not inputs:
        raise ValueError(f"No supported input images found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = args.output_extension
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    seen_stems: set[str] = set()
    rows = []
    for input_path in inputs:
        if input_path.stem in seen_stems:
            raise ValueError(f"Duplicate input stem would overwrite an output: {input_path.stem}")
        seen_stems.add(input_path.stem)
        output_path = output_dir / f"{input_path.stem}{extension or input_path.suffix.lower()}"
        source_shape, restored_shape, runtime = restore_one(model, config, input_path, output_path, device, use_amp, args.tta)
        rows.append({"sample_id": input_path.stem, "input_path": str(input_path), "output_path": str(output_path), "input_height": source_shape[0], "input_width": source_shape[1], "output_height": restored_shape[0], "output_width": restored_shape[1], "runtime_ms": f"{runtime * 1000:.4f}"})
    manifest_path = output_dir / "inference_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "input_path", "output_path", "input_height", "input_width", "output_height", "output_width", "runtime_ms"))
        writer.writeheader()
        writer.writerows(rows)
    average_ms = sum(float(row["runtime_ms"]) for row in rows) / len(rows)
    print(f"Restored {len(rows)} images on {device}; mean model time {average_ms:.2f} ms/image")
    print(f"Outputs: {output_dir}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
