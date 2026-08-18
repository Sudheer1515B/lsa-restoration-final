# KLA Semiconductor Image Restoration

## Submission quick start

The evaluator-facing submission consists of `run.py`, `requirements.txt`, and
the self-contained checkpoint in `models/final_model.pt`. From the submission
root, install the two pinned runtime dependencies and run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py <input-dir> <output-dir>
```

After activating that environment (or on an evaluator where `python` already
points to the installed interpreter), the required command is equivalently:

```bash
python run.py <input-dir> <output-dir>
```

`run.py` reads every `.npy` file directly inside `<input-dir>`, creates
`<output-dir>` when needed, and writes one `float32` grayscale `.npy` array
with the same filename for each input. The model restores at the checkpoint's
configured 2× resolution (for KLA inputs: 128×128 → 256×256) and clamps every
output to finite values in `[0, 1]`. It automatically uses an NVIDIA CUDA GPU
when a CUDA-enabled PyTorch installation is available; otherwise it runs on
CPU. No network access, API keys, downloads, prompts, or manual configuration
are required after dependencies are installed.

The bundled checkpoint contains its architecture configuration and weights.
Use `python run.py --help` only for the optional `--checkpoint` and `--device`
overrides; neither is required by the evaluator command.

Submission for the SEMICON India Hackathon restoration task: a lightweight, single-image NAFNet-style model that restores grayscale semiconductor inspection images at 2× spatial resolution. The final model (`weights/final_model.pt`) scores **PSNR 28.99 dB / SSIM 0.8247 / LPIPS 0.1341** on a genuine held-out validation split (never trained on) — see [`docs/methodology.md`](docs/methodology.md) for the full methodology and measurements.

## Observed dataset contract

`train.zip` contains 3,200 paired `float32` NumPy arrays:

```text
train/NoisyLR/000298.npy  (128, 128), noisy values can be outside [0, 1]
train/GT/000298.npy       (256, 256), clean values are in [0, 1]
```

The numeric filename stem is the pairing key within `train.zip`. `Test_NoisyLR.zip` contains 400 additional noisy captures with the same `128×128` shape. Although some stems overlap with training GT filenames, direct image comparison shows these are not valid noisy/GT pairs; it is unpaired inference data, not validation data. The baseline therefore makes a seeded validation split from genuine train pairs.

Do not clip the noisy input before it reaches the model. The baseline uses `floating_point_scale: 1.0`, and clips only a final serialized output.

## Setup and inspection

Place the supplied `train.zip` and `Test_NoisyLR.zip` in the repo root (not included in this repository), then create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

python prepare_kla_data.py
.venv/bin/python inspect_dataset.py \
  --degraded-dir data/kla/train/degraded \
  --gt-dir data/kla/train/ground_truth \
  --test-zip Test_NoisyLR.zip \
  --output outputs/kla_dataset_report.md
```
`inspect_dataset.py` reports paths, filename-stem pairing, resolutions, dtype/range, exact duplicates, and near-duplicate candidates. It can also read the training archive directly with `--train-zip train.zip`.

## NAFNet-SR model and training path

[`model/nafnet_sr.py`](model/nafnet_sr.py) implements a standard-PyTorch NAFNet-style encoder, bottleneck, skip-connected decoder, and residual PixelShuffle ×2 reconstruction head. It accepts `[B, 1, H, W]` and returns `[B, 1, 2H, 2W]`, including non-divisible input dimensions. Only paired geometric flips/90° rotations are used for augmentation; no synthetic degradation is added.

[`configs/baseline.yaml`](configs/baseline.yaml) is the smallest working config (`L1 + 0.2 × SSIM` loss) — a good smoke-test starting point, but **not** the final model.

The final model, [`configs/medium_nafnet_a.yaml`](configs/medium_nafnet_a.yaml) (width 32, `enc_blocks=[1,2,2]`, `middle_blocks=2`, `dec_blocks=[2,2,1]`, ~1.9M params), was trained in two stages:

1. 80 epochs with `Charbonnier + 0.10×SSIM + 0.05×Sobel-gradient` loss (`loss.lambda_ssim`/`loss.lambda_gradient` in the config).
2. A 25-epoch fine-tune adding a small `0.05×LPIPS` term ([`configs/medium_nafnet_a_lpips_ft.yaml`](configs/medium_nafnet_a_lpips_ft.yaml)), which cut LPIPS by 46% (0.2506 → 0.1341) for a 0.15 dB PSNR cost. See [`docs/methodology.md`](docs/methodology.md) for the full rationale and measurements.

`weights/final_model.pt` is the result of stage 2.

## Verify before a full experiment

Run the tiny shared-sample overfit check first. It should steadily improve its validation (same-as-training) PSNR/SSIM, write ranked panels, curves, history, and portable checkpoints.

```bash
.venv/bin/python train.py \
  --config configs/overfit.yaml \
  --overfit-samples 8
```

Then start a practical five-epoch subset baseline before moving to the full training set:

```bash
.venv/bin/python train.py \
  --config configs/baseline.yaml \
  --max-train-samples 128 \
  --max-val-samples 32
```

For the real baseline, omit the subset flags. Device selection is automatic: CUDA → MPS → CPU. Override it with `--device cuda`, `--device mps`, or `--device cpu`; enable mixed precision only on CUDA using `--amp`.

Resume safely across CPU/MPS/CUDA with:

```bash
.venv/bin/python train.py \
  --config configs/baseline.yaml \
  --resume checkpoints/baseline/latest.pth
```

Each epoch creates:

```text
checkpoints/baseline/latest.pth
checkpoints/baseline/best_psnr.pth
checkpoints/baseline/best_ssim.pth
outputs/baseline/training_history.csv
outputs/baseline/training_curves.png
outputs/baseline/epoch_001/validation_metrics.csv
outputs/baseline/epoch_001/severity_breakdown.csv
outputs/baseline/epoch_001/best/01_<sample-id>.png
outputs/baseline/epoch_001/worst/01_<sample-id>.png
```

The panels contain `INPUT | RESTORED | GROUND TRUTH` and the sample ID, PSNR, and SSIM. Checkpoints embed model configuration and CPU-normalized state dictionaries so they can be moved between MPS and CUDA.

`validation_metrics.csv` includes each validation image's `noise_std` (the standard deviation of its raw degraded input, a cheap per-image corruption-severity proxy). `severity_breakdown.csv` groups that epoch's validation images into three equal-count severity buckets (least to most corrupted) and reports each bucket's mean PSNR/SSIM, so a model that only improves on mildly-corrupted images doesn't hide behind a better dataset-wide average; `training_history.csv` and `training_curves.png` also track the most-corrupted third's PSNR/SSIM (`high_severity_psnr`/`high_severity_ssim`) across epochs. See [`configs/charbonnier_ssim_hard.yaml`](configs/charbonnier_ssim_hard.yaml) for an optional `loss.hard_pixel_weight` term (see [`losses.py`](losses.py)'s `hard_pixel_loss`) that specifically penalizes each image's highest-residual pixels, for when the pixel/SSIM terms alone under-correct sparse high-magnitude noise such as speckle.

## Submission inference (directory evaluator)

`infer.py` is the standalone evaluation entrypoint. It accepts an input directory and output directory without source edits, restores every supported grayscale image, preserves input stems, and writes `inference_manifest.csv` with image dimensions and per-image runtime.

`weights/final_model.pt` is the committed submission checkpoint; pass `--checkpoint` to point at a different one:

```bash
python infer.py \
  --checkpoint weights/final_model.pt \
  --input_dir <test-images-directory> \
  --output_dir restored_test_outputs
```

If test inputs are KLA NumPy arrays, outputs normally remain `.npy`; pass `--output_extension .png` only when PNG delivery is required. `restored_test_outputs/` in this repository already contains the actual final-test run's `.npy` outputs plus `inference_manifest.csv`, generated from `weights/final_model.pt`; re-run the command above if you regenerate the model.

Add `--tta` to either `infer.py` or `evaluate.py` for an optional D4 self-ensemble (8× slower). It's measured but **not** used for the committed `restored_test_outputs/` — see [`docs/methodology.md`](docs/methodology.md) for why (it trades back part of the LPIPS gain above for a small PSNR/SSIM bump).

### Single-image check

Restore one image using the embedded checkpoint configuration:

```bash
.venv/bin/python infer.py \
  --checkpoint weights/final_model.pt \
  --input data/kla/test/degraded/000298.npy \
  --output outputs/restored_000298.npy
```

## Metric evaluation

Evaluate paired images (`<data>/degraded` and `<data>/ground_truth`):

```bash
.venv/bin/python evaluate.py \
  --checkpoint weights/final_model.pt \
  --data <a genuine paired validation directory> \
  --output outputs/heldout_metrics.csv
```

Or reconstruct the model's own genuine held-out validation split (the one it was actually scored on above) directly from the checkpoint's embedded config, instead of pointing at a directory by hand:

```bash
.venv/bin/python evaluate.py \
  --checkpoint weights/final_model.pt \
  --val-split \
  --output outputs/genuine_val_metrics.csv
```

Both report mean PSNR, SSIM, optional LPIPS, and inference time. LPIPS is evaluation-only; if its package or pretrained weights are unavailable, evaluation still reports the primary metrics. See [`docs/methodology.md`](docs/methodology.md) for the evaluation protocol.
