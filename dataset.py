"""Paired grayscale restoration dataset and deterministic split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler


SUPPORTED_EXTENSIONS = (".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class ImagePair:
    """Validated source/target paths and their native spatial dimensions."""

    sample_id: str
    degraded_path: Path
    ground_truth_path: Path
    degraded_size: tuple[int, int]
    ground_truth_size: tuple[int, int]


def list_images(directory: str | Path, extensions: Sequence[str] = SUPPORTED_EXTENSIONS) -> list[Path]:
    """Return deterministically ordered supported image files from one directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    allowed = {extension.lower() for extension in extensions}
    return sorted((path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed), key=lambda path: path.name)


def read_grayscale(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    """Read a grayscale NumPy or raster image without clipping its values."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
        original_mode = "npy"
    else:
        with Image.open(path) as handle:
            original_mode = handle.mode
            image = np.asarray(handle.convert("L"))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale array at {path}, got {image.shape}")
    if not (np.issubdtype(image.dtype, np.number) or np.issubdtype(image.dtype, np.bool_)):
        raise ValueError(f"Expected numeric values at {path}, got {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError(f"Image contains NaN or infinite values: {path}")
    return image, {"path": str(path), "mode": original_mode, "dtype": str(image.dtype), "shape": tuple(image.shape), "min": float(image.min()), "max": float(image.max())}


def _normalization_scale(image: np.ndarray, floating_point_scale: float | None) -> float:
    """Return the divisor that maps ``image``'s dtype range onto ``[0, 1]``-ish floats.

    Integer images are scaled by their dtype's max representable value and
    boolean images by 1.0. Floating-point images have no implied range (they
    may already be arbitrary sensor units), so ``floating_point_scale`` must
    be supplied explicitly rather than guessed.
    """
    if np.issubdtype(image.dtype, np.integer):
        return float(max(np.iinfo(image.dtype).max, 1))
    if np.issubdtype(image.dtype, np.bool_):
        return 1.0
    if floating_point_scale is None:
        raise ValueError("Floating-point inputs require data.floating_point_scale; inspect the dataset before choosing it")
    if floating_point_scale <= 0:
        raise ValueError("floating_point_scale must be positive")
    return float(floating_point_scale)


def image_to_tensor(image: np.ndarray, floating_point_scale: float | None) -> torch.Tensor:
    """Convert a 2-D image to an unclipped ``[1, H, W]`` float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(image.astype(np.float32) / _normalization_scale(image, floating_point_scale))).unsqueeze(0)


def _index_by_stem(paths: list[Path]) -> dict[str, Path]:
    """Map each path's filename stem to its path, raising on duplicate stems."""
    indexed: dict[str, Path] = {}
    for path in paths:
        if path.stem in indexed:
            raise ValueError(f"Duplicate filename stem: {path.stem}")
        indexed[path.stem] = path
    return indexed


def find_pairs(
    degraded_dir: str | Path,
    ground_truth_dir: str | Path,
    extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
    required_scale: int | None = 2,
) -> list[ImagePair]:
    """Pair source and target files by stem and validate their configured scale."""
    degraded = _index_by_stem(list_images(degraded_dir, extensions))
    targets = _index_by_stem(list_images(ground_truth_dir, extensions))
    missing_targets = sorted(set(degraded) - set(targets))
    missing_degraded = sorted(set(targets) - set(degraded))
    if missing_targets or missing_degraded:
        pieces = []
        if missing_targets:
            pieces.append("missing GT: " + ", ".join(missing_targets[:10]))
        if missing_degraded:
            pieces.append("missing degraded input: " + ", ".join(missing_degraded[:10]))
        raise ValueError("Pairing by filename stem failed; " + "; ".join(pieces))
    pairs: list[ImagePair] = []
    for sample_id in sorted(degraded):
        source, _ = read_grayscale(degraded[sample_id])
        target, _ = read_grayscale(targets[sample_id])
        source_size, target_size = tuple(source.shape), tuple(target.shape)
        if required_scale and target_size != (source_size[0] * required_scale, source_size[1] * required_scale):
            raise ValueError(f"{sample_id}: {source_size} -> {target_size}, expected {required_scale}x")
        pairs.append(ImagePair(sample_id, degraded[sample_id], targets[sample_id], source_size, target_size))
    if not pairs:
        raise ValueError("No paired files found")
    return pairs


class PairedRestorationDataset(Dataset):
    """Returns exactly ``(degraded, ground_truth)`` tensors for each paired image."""

    def __init__(
        self,
        pairs: Sequence[ImagePair],
        floating_point_scale: float | None,
        training: bool = False,
        crop_size_lr: int | None = None,
        geometric_augment: bool = False,
        return_metadata: bool = False,
        texture_aware_sampling: bool = False,
        texture_candidate_count: int = 8,
    ):
        """Store the validated pairs and how to load/augment them.

        Args:
            pairs: Validated source/target file pairs to serve.
            floating_point_scale: Divisor for floating-point images (see
                ``_normalization_scale``); ignored for integer/bool images.
            training: If True, apply random cropping and augmentation.
            crop_size_lr: Random-crop size in the degraded (low-res) image's
                pixel space, or None to skip cropping.
            geometric_augment: If True, apply random flips/rotations during
                training.
            return_metadata: If True, also return each sample's ``sample_id``.
            texture_aware_sampling: If True, bias crop position selection
                toward higher-detail regions instead of choosing purely
                uniformly at random (see ``_choose_crop_position``). ``False``
                (the default) reproduces the original single-uniform-draw
                behavior exactly, so existing configs are unaffected.
            texture_candidate_count: Number of candidate crop positions drawn
                per high-/medium-detail selection when
                ``texture_aware_sampling`` is enabled. Ignored otherwise.
        """
        self.pairs = list(pairs)
        self.floating_point_scale = floating_point_scale
        self.training = training
        self.crop_size_lr = crop_size_lr
        self.geometric_augment = geometric_augment
        self.return_metadata = return_metadata
        self.texture_aware_sampling = texture_aware_sampling
        self.texture_candidate_count = texture_candidate_count

    def __len__(self) -> int:
        """Return the number of pairs in the dataset."""
        return len(self.pairs)

    def _crop(self, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Take a matching crop from ``source`` and its higher-resolution ``target``.

        The crop size is clamped to the source image's dimensions, and the
        corresponding target-space crop is scaled up by the inferred
        source-to-target ratio so the two stay spatially aligned. Crop
        *position* is chosen by ``_choose_crop_position`` (uniform random by
        default, optionally texture-aware).
        """
        if not self.crop_size_lr:
            return source, target
        source_h, source_w = source.shape[-2:]
        crop = min(self.crop_size_lr, source_h, source_w)
        if crop == source_h and crop == source_w:
            return source, target
        scale_h = target.shape[-2] // source_h
        scale_w = target.shape[-1] // source_w
        if scale_h != scale_w:
            raise ValueError("Input and target have incompatible spatial scales")
        top, left = self._choose_crop_position(source, source_h, source_w, crop)
        return (
            source[:, top : top + crop, left : left + crop],
            target[:, top * scale_h : (top + crop) * scale_h, left * scale_h : (left + crop) * scale_h],
        )

    def _choose_crop_position(self, source: torch.Tensor, source_h: int, source_w: int, crop: int) -> tuple[int, int]:
        """Choose the ``(top, left)`` corner of the next low-res crop.

        By default this is a single uniform-random draw -- the same two
        ``random.randint`` calls the original implementation made, so
        ``texture_aware_sampling=False`` (the default) reproduces prior runs
        exactly. When enabled, it mixes in occasional high-/medium-detail
        crops so texture-heavy regions (fine structure, edges) aren't
        outnumbered by flat background in gradient updates: pure random
        cropping can teach the model that smoothing is always a safe bet,
        since most of an image's area is typically flat.

        Args:
            source: The full (uncropped) degraded-image tensor, ``[1, H, W]``.
            source_h: Source image height.
            source_w: Source image width.
            crop: Crop side length (already clamped to fit the source).

        Returns:
            ``(top, left)`` position for the low-res crop.
        """
        if not self.texture_aware_sampling:
            return random.randint(0, source_h - crop), random.randint(0, source_w - crop)

        policy = random.random()
        if policy < 0.5:
            # 50% of draws: plain uniform-random position, same as the default.
            return random.randint(0, source_h - crop), random.randint(0, source_w - crop)

        candidates = [
            (random.randint(0, source_h - crop), random.randint(0, source_w - crop))
            for _ in range(self.texture_candidate_count)
        ]
        ranked = sorted(candidates, key=lambda position: self._detail_score(source, position, crop), reverse=True)
        # 30% of draws (policy in [0.5, 0.8)): the highest-detail candidate.
        # 20% of draws (policy in [0.8, 1.0)): a medium-detail candidate.
        return ranked[0] if policy < 0.8 else ranked[len(ranked) // 2]

    @staticmethod
    def _detail_score(source: torch.Tensor, position: tuple[int, int], crop: int) -> float:
        """Score a candidate crop's local detail as its pixel standard deviation.

        A cheap proxy for a Sobel-magnitude score: flat/background regions
        have low variance, fine structure and edges have high variance.
        """
        top, left = position
        patch = source[:, top : top + crop, left : left + crop]
        return float(patch.std())

    def _augment(self, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same random horizontal/vertical flip and 90-degree rotation to both tensors."""
        if not self.geometric_augment:
            return source, target
        if random.random() < 0.5:
            source, target = torch.flip(source, (-1,)), torch.flip(target, (-1,))
        if random.random() < 0.5:
            source, target = torch.flip(source, (-2,)), torch.flip(target, (-2,))
        turns = random.randrange(4)
        return torch.rot90(source, turns, (-2, -1)), torch.rot90(target, turns, (-2, -1))

    def __getitem__(self, index: int):
        """Load, normalize, and (if training) crop/augment the pair at ``index``.

        Returns:
            ``(degraded, ground_truth)`` tensors, or ``(degraded, ground_truth,
            sample_id)`` when ``return_metadata`` is set.
        """
        pair = self.pairs[index]
        source, _ = read_grayscale(pair.degraded_path)
        target, _ = read_grayscale(pair.ground_truth_path)
        source_tensor = image_to_tensor(source, self.floating_point_scale)
        target_tensor = image_to_tensor(target, self.floating_point_scale)
        if self.training:
            source_tensor, target_tensor = self._crop(source_tensor, target_tensor)
            source_tensor, target_tensor = self._augment(source_tensor, target_tensor)
        if self.return_metadata:
            return source_tensor, target_tensor, pair.sample_id
        return source_tensor, target_tensor


def deterministic_split(pairs: Sequence[ImagePair], validation_fraction: float, seed: int) -> tuple[list[ImagePair], list[ImagePair]]:
    """Create a reproducible non-overlapping training/validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(pairs) < 2:
        raise ValueError("At least two pairs are required to create a split")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    validation_count = min(max(1, round(len(pairs) * validation_fraction)), len(pairs) - 1)
    validation_indices = set(indices[:validation_count])
    return ([pair for i, pair in enumerate(pairs) if i not in validation_indices], [pair for i, pair in enumerate(pairs) if i in validation_indices])


def deterministic_subset(pairs: Sequence[ImagePair], limit: int | None, seed: int) -> list[ImagePair]:
    """Choose a reproducible subset without biasing a smoke run toward filename order."""
    if limit is None or limit >= len(pairs):
        return list(pairs)
    if limit < 1:
        raise ValueError("Subset size must be positive")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    selected = set(indices[:limit])
    return [pair for index, pair in enumerate(pairs) if index in selected]


class ShapeBatchSampler(Sampler[list[int]]):
    """Avoid mixing image dimensions when no crop makes them uniform."""

    def __init__(self, dataset: PairedRestorationDataset, batch_size: int, shuffle: bool):
        """Group ``dataset``'s sample indices by degraded-image size to form batches.

        Args:
            dataset: Dataset whose pairs are grouped by ``degraded_size``.
            batch_size: Maximum number of samples per yielded batch.
            shuffle: If True, shuffle indices within each size group and the
                resulting batch order.
        """
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.groups: dict[tuple[int, int], list[int]] = {}
        for index, pair in enumerate(dataset.pairs):
            self.groups.setdefault(pair.degraded_size, []).append(index)

    def __iter__(self):
        """Yield batches of indices, each batch drawn from a single size group."""
        batches = []
        for group in self.groups.values():
            indices = group.copy()
            if self.shuffle:
                random.shuffle(indices)
            batches.extend(indices[offset : offset + self.batch_size] for offset in range(0, len(indices), self.batch_size))
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        """Return the total number of batches across all size groups."""
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups.values())
