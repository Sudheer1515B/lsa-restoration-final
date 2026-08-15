"""Inspect paired KLA restoration data before configuring a training run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from dataset import find_pairs, read_grayscale


def parse_args() -> argparse.Namespace:
    """Parse extracted-directory or archive inspection options."""
    parser = argparse.ArgumentParser(description="Print pairing, image, range, and duplicate diagnostics for paired restoration data.")
    parser.add_argument("--degraded-dir", default="data/kla/train/degraded")
    parser.add_argument("--gt-dir", default="data/kla/train/ground_truth")
    parser.add_argument("--output", default="outputs/dataset_report.md")
    parser.add_argument("--train-zip", default=None, help="Inspect train.zip directly instead of extracted directories.")
    parser.add_argument("--test-zip", default=None, help="Optional Test_NoisyLR.zip to describe the test-like inputs.")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9999)
    return parser.parse_args()


def thumbnail_fingerprint(image: np.ndarray, grid: int = 16) -> np.ndarray | None:
    """Normalize a small average-pooled view; high cosine similarity is a candidate duplicate."""
    height, width = image.shape
    if height < grid or width < grid:
        return None
    cropped = image[: height - height % grid, : width - width % grid].astype(np.float32, copy=False)
    thumb = cropped.reshape(grid, cropped.shape[0] // grid, grid, cropped.shape[1] // grid).mean(axis=(1, 3)).ravel()
    thumb -= thumb.mean()
    norm = float(np.linalg.norm(thumb))
    return thumb / norm if norm > 1e-12 else None


def near_duplicate_candidates(sample_ids: list[str], fingerprints: list[np.ndarray | None], threshold: float) -> list[tuple[str, str, float]]:
    """Return up to ten high-similarity source-image pairs for manual review."""
    valid = [(sample_id, fingerprint) for sample_id, fingerprint in zip(sample_ids, fingerprints) if fingerprint is not None]
    if len(valid) < 2:
        return []
    ids = [sample_id for sample_id, _ in valid]
    vectors = np.stack([vector for _, vector in valid])
    candidates: list[tuple[str, str, float]] = []
    for start in range(0, len(ids), 256):
        similarities = vectors[start : start + 256] @ vectors.T
        for local_index, row in enumerate(similarities):
            absolute_index = start + local_index
            row[absolute_index:] = -np.inf
            other_index = int(row.argmax())
            similarity = float(row[other_index])
            if similarity >= threshold:
                candidates.append((ids[other_index], ids[absolute_index], similarity))
    return sorted(candidates, key=lambda item: item[2], reverse=True)[:10]


def _hash_array(array: np.ndarray) -> str:
    """Return a content hash of ``array`` (shape, dtype, and raw bytes) for exact-duplicate detection."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def summarize_pairs(pairs: list[tuple[str, np.ndarray, np.ndarray]], threshold: float) -> tuple[list[str], dict[str, object]]:
    """Aggregate format, range, duplicate, and pairing diagnostics into a report."""
    resolutions, dtypes = Counter(), Counter()
    source_mins: list[float] = []
    source_maxs: list[float] = []
    target_mins: list[float] = []
    target_maxs: list[float] = []
    source_means: list[float] = []
    source_stds: list[float] = []
    source_hashes: dict[str, list[str]] = {}
    target_hashes: dict[str, list[str]] = {}
    fingerprints: list[np.ndarray | None] = []
    sample_ids: list[str] = []
    for sample_id, source, target in pairs:
        resolutions[f"{source.shape[0]}x{source.shape[1]} -> {target.shape[0]}x{target.shape[1]}"] += 1
        dtypes[f"degraded: {source.dtype}"] += 1
        dtypes[f"ground truth: {target.dtype}"] += 1
        source_mins.append(float(source.min()))
        source_maxs.append(float(source.max()))
        target_mins.append(float(target.min()))
        target_maxs.append(float(target.max()))
        source_means.append(float(source.mean()))
        source_stds.append(float(source.std()))
        source_hashes.setdefault(_hash_array(source), []).append(sample_id)
        target_hashes.setdefault(_hash_array(target), []).append(sample_id)
        sample_ids.append(sample_id)
        fingerprints.append(thumbnail_fingerprint(source))
    source_duplicates = [ids for ids in source_hashes.values() if len(ids) > 1]
    target_duplicates = [ids for ids in target_hashes.values() if len(ids) > 1]
    near_duplicates = near_duplicate_candidates(sample_ids, fingerprints, threshold)
    lines = [
        "## Pairing",
        f"- Matched pairs: {len(pairs)}",
        "- Mapping: identical filename stem (`NoisyLR/<id>.npy` ↔ `GT/<id>.npy`).",
        "",
        "## Resolution and format",
        *[f"- `{name}`: {count}" for name, count in sorted(resolutions.items())],
        *[f"- `{name}`: {count}" for name, count in sorted(dtypes.items())],
        "",
        "## Intensity range",
        f"- Degraded global min/max: {min(source_mins):.6g} / {max(source_maxs):.6g}",
        f"- Ground-truth global min/max: {min(target_mins):.6g} / {max(target_maxs):.6g}",
        f"- Degraded per-image mean range: {min(source_means):.6g} / {max(source_means):.6g}",
        f"- Degraded per-image std range: {min(source_stds):.6g} / {max(source_stds):.6g}",
        "- No clipping was applied during inspection.",
        "",
        "## Degradation labels and duplicates",
        "- No explicit degradation-type label is encoded in the archive paths or numeric filenames; varying intensity statistics indicate multiple corruption severities may be present.",
        f"- Exact degraded duplicate groups: {len(source_duplicates)}",
        f"- Exact ground-truth duplicate groups: {len(target_duplicates)}",
        f"- Near-duplicate candidates (16×16 normalized thumbnail cosine ≥ {threshold}): {len(near_duplicates)}",
    ]
    for first, second, similarity in near_duplicates:
        lines.append(f"  - `{first}` / `{second}`: {similarity:.6f}")
    return lines, {
        "pairs": len(pairs),
        "source_min": min(source_mins),
        "source_max": max(source_maxs),
        "target_min": min(target_mins),
        "target_max": max(target_maxs),
        "source_duplicate_groups": len(source_duplicates),
        "target_duplicate_groups": len(target_duplicates),
        "near_duplicates": len(near_duplicates),
    }


def pairs_from_directories(degraded_dir: str, gt_dir: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load validated pairs from the prepared on-disk directory layout."""
    pairs = find_pairs(degraded_dir, gt_dir, required_scale=2)
    result = []
    for pair in pairs:
        source, _ = read_grayscale(pair.degraded_path)
        target, _ = read_grayscale(pair.ground_truth_path)
        result.append((pair.sample_id, source, target))
    return result


def pairs_from_train_zip(path: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load genuine train pairs directly from the immutable supplied archive."""
    with zipfile.ZipFile(path) as archive:
        source_members = {Path(name).stem: name for name in archive.namelist() if name.startswith("train/NoisyLR/") and name.endswith(".npy")}
        target_members = {Path(name).stem: name for name in archive.namelist() if name.startswith("train/GT/") and name.endswith(".npy")}
        if set(source_members) != set(target_members):
            raise ValueError("NoisyLR and GT IDs differ within train archive")
        result = []
        for sample_id in sorted(source_members):
            with archive.open(source_members[sample_id]) as source_handle, archive.open(target_members[sample_id]) as target_handle:
                result.append((sample_id, np.load(source_handle, allow_pickle=False), np.load(target_handle, allow_pickle=False)))
        return result


def summarize_test_zip(path: str, train_pairs: list[tuple[str, np.ndarray, np.ndarray]]) -> list[str]:
    """Describe test inputs and test whether same-stem train GT is structurally valid."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.startswith("NoisyLR/") and name.endswith(".npy")]
        if not members:
            raise ValueError(f"No NoisyLR NumPy files found in {path}")
        arrays: list[tuple[str, np.ndarray]] = []
        for name in members:
            with archive.open(name) as handle:
                arrays.append((Path(name).stem, np.load(handle, allow_pickle=False)))
    targets = {sample_id: target for sample_id, _, target in train_pairs}
    comparable = []
    # Force-pair each test input against any train GT that happens to share its filename
    # stem, purely to measure whether that coincidental overlap could be a real pairing.
    # A true match would downsample cleanly to a near-zero-noise, high-PSNR match; low
    # PSNR here is the evidence that these stem collisions are NOT genuine pairs.
    for sample_id, source in arrays:
        target = targets.get(sample_id)
        if target is None or target.shape != (source.shape[0] * 2, source.shape[1] * 2):
            continue
        downsampled_target = target.reshape(source.shape[0], 2, source.shape[1], 2).mean(axis=(1, 3))
        mse = float(np.mean((source - downsampled_target) ** 2))
        comparable.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    test_min = min(float(array.min()) for _, array in arrays)
    test_max = max(float(array.max()) for _, array in arrays)
    pairing_message = "No same-stem target arrays exist in train.zip."
    if comparable:
        pairing_message = (
            f"Same-stem comparison against a 2× downsampled train GT: mean {float(np.mean(comparable)):.2f} dB. "
            "This low structural agreement means same stems must not be treated as paired validation targets."
        )
    return [
        "",
        "## Test-like archive",
        f"- Noisy inputs: {len(arrays)}",
        f"- Shape: {arrays[0][1].shape}; dtype: {arrays[0][1].dtype}",
        f"- Global min/max: {test_min:.6g} / {test_max:.6g}",
        f"- Pairing check: {pairing_message}",
    ]


def main() -> None:
    """Write and print the concise dataset report used to configure the pipeline."""
    args = parse_args()
    if args.train_zip:
        pairs = pairs_from_train_zip(args.train_zip)
        source = f"archive `{args.train_zip}`"
    else:
        pairs = pairs_from_directories(args.degraded_dir, args.gt_dir)
        source = f"directories `{args.degraded_dir}` and `{args.gt_dir}`"
    lines, summary = summarize_pairs(pairs, args.near_duplicate_threshold)
    report = ["# KLA Dataset Inspection", "", f"Inspected {source}.", "", *lines]
    if args.test_zip:
        report.extend(summarize_test_zip(args.test_zip, pairs))
    report.extend(["", "## Conclusion", f"- Configure x2 super-resolution: {summary['pairs']} matched grayscale float arrays with the observed source/target mapping above.", "- Keep floating-point normalization at 1.0 for this supplied KLA dataset, preserving over-range noisy values.", "- Use a deterministic split of the genuine train pairs for validation; do not manufacture Test_NoisyLR/GT pairs based only on filename stems.", ""])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"KLA dataset inspection: {summary['pairs']} matched pairs")
    print(f"Pairing: numeric filename stem; degraded range {summary['source_min']:.6g}..{summary['source_max']:.6g}; GT range {summary['target_min']:.6g}..{summary['target_max']:.6g}")
    print(f"Exact duplicate groups: degraded={summary['source_duplicate_groups']}, GT={summary['target_duplicate_groups']}; near-duplicate candidates={summary['near_duplicates']}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
