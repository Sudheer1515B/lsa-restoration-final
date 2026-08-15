"""Prepare the supplied KLA archives without inventing test/GT pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def parse_args() -> argparse.Namespace:
    """Parse source archive and destination paths without modifying the archives."""
    parser = argparse.ArgumentParser(description="Create the local KLA train and held-out data layout from supplied ZIP archives.")
    parser.add_argument("--train_zip", default="train.zip")
    parser.add_argument("--test_zip", default="Test_NoisyLR.zip")
    parser.add_argument("--output_root", default="data/kla")
    return parser.parse_args()


def members_by_stem(archive: zipfile.ZipFile, prefix: str) -> dict[str, str]:
    """Index NumPy members under one archive prefix by their unique filename stem."""
    members = {}
    for name in archive.namelist():
        if not name.startswith(prefix) or not name.endswith(".npy"):
            continue
        stem = Path(name).stem
        if stem in members:
            raise ValueError(f"Duplicate array stem '{stem}' under {prefix} in {archive.filename}")
        members[stem] = name
    if not members:
        raise ValueError(f"No NumPy arrays found under {prefix} in {archive.filename}")
    return members


def write_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    """Extract one member idempotently, refusing to overwrite conflicting data."""
    payload = archive.read(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(payload).digest():
            raise FileExistsError(f"Refusing to overwrite different data at {destination}")
        return
    destination.write_bytes(payload)


def main() -> None:
    """Create paired train files and explicitly unpaired test-input files."""
    args = parse_args()
    train_zip, test_zip, root = Path(args.train_zip), Path(args.test_zip), Path(args.output_root)
    if not train_zip.is_file() or not test_zip.is_file():
        raise FileNotFoundError("Both --train_zip and --test_zip must point to existing files")
    with zipfile.ZipFile(train_zip) as train_archive, zipfile.ZipFile(test_zip) as test_archive:
        train_degraded = members_by_stem(train_archive, "train/NoisyLR/")
        train_ground_truth = members_by_stem(train_archive, "train/GT/")
        test_degraded = members_by_stem(test_archive, "NoisyLR/")
        if set(train_degraded) != set(train_ground_truth):
            raise ValueError("Training NoisyLR and GT archives do not have identical sample IDs")
        # Test_NoisyLR stems only coincidentally overlap with train GT stems; there is no
        # image-alignment evidence they refer to the same underlying samples, so the test
        # inputs are copied out as unpaired inference data rather than matched to any GT.
        heldout_ids = sorted(test_degraded)
        for sample_id, member in train_degraded.items():
            write_member(train_archive, member, root / "train" / "degraded" / f"{sample_id}.npy")
        for sample_id, member in train_ground_truth.items():
            write_member(train_archive, member, root / "train" / "ground_truth" / f"{sample_id}.npy")
        for sample_id, member in test_degraded.items():
            write_member(test_archive, member, root / "test" / "degraded" / f"{sample_id}.npy")
    split_path = root / "splits" / "test_like_ids.txt"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(heldout_ids) + "\n", encoding="utf-8")
    manifest = {
        "train_pairs": len(train_degraded),
        "test_inputs": len(heldout_ids),
        "test_policy": "Test_NoisyLR is unpaired inference data. Matching filename stems in train.zip are not treated as matching clean targets without image-alignment evidence.",
        "train_zip": str(train_zip),
        "test_zip": str(test_zip),
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {manifest['train_pairs']} training pairs and {manifest['test_inputs']} unpaired test inputs under {root}")


if __name__ == "__main__":
    main()
