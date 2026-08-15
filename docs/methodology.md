# Methodology

## Problem

Restore 128×128 degraded grayscale semiconductor inspection images (speckle
and Gaussian noise, downsampled from a higher-resolution source) to clean
256×256 images matching ground truth. Noise pushes some input pixel values
outside the `[0, 1]` range the clean images live in; the model must handle
that without the input being clipped first.

## Architecture

[`model/nafnet_sr.py`](../model/nafnet_sr.py) implements a NAFNet-style
encoder / bottleneck / skip-connected decoder with a residual PixelShuffle
×2 reconstruction head. The network predicts a residual correction on top of
a bicubic upsample of the input, rather than reconstructing the image from
scratch — a well-behaved formulation that only requires learning the delta
between the cheap upsample and the true clean image.

Capacity was scaled in controlled, isolated steps rather than guessed: a
~291K-parameter model was trained first as a control point, then capacity
was increased in two steps — a width-only bump (~646K params) and a combined
width-and-depth increase (**medium_nafnet_a**, ~1.9M params: `width=32`,
`enc_blocks=[1,2,2]`, `middle_blocks=2`, `dec_blocks=[2,2,1]`) — comparing
PSNR/SSIM, worst-case visual panels, and per-image inference time at each
step. medium_nafnet_a is the architecture behind the final submission.

## Loss function design

The training objective combines four terms, each isolated and validated on
its own before being added:

```text
L = Charbonnier(pred, gt)
  + λ_ssim    * (1 - SSIM(pred, gt))
  + λ_grad    * SobelL1(pred, gt)
  + λ_lpips   * LPIPS(pred, gt)
```

- **Charbonnier** (`sqrt((pred-gt)² + ε²)`) replaces raw L1/L2 for its smooth
  gradient near zero and robustness to the outlier pixel values speckle
  noise produces.
- **SSIM** (weight 0.10) rewards local structural agreement, not just
  pixel-wise closeness.
- **Sobel gradient** (weight 0.05) directly compares image derivatives,
  rewarding correct edge location and strength rather than relying on the
  pixel/SSIM terms to reproduce edges as a side effect.
- **LPIPS** (weight 0.05, added in a second fine-tuning stage) directly
  optimizes the perceptual metric the challenge is graded on, instead of
  treating it as a spectator metric measured only at the end.

### Why the LPIPS term

PSNR and SSIM both reward the statistically safe, averaged answer, which
biases a pixel/structural loss toward over-smoothing. Measured on a genuine
held-out split, the pre-LPIPS model scored **PSNR 29.14 dB / SSIM 0.8283**
but only **LPIPS 0.2506** — inside the over-smoothed range (perceptually
good restorations are typically well under 0.15). Visually this showed up as
faint real signal being suppressed alongside noise: in a starfield test
image, the model kept the brightest points but visibly smoothed away several
fainter ones that were genuinely present in the input — a costly failure
mode for a task where the goal is preserving faint real detail, not just
denoising.

A small LPIPS term was added and the model fine-tuned for 25 epochs from the
fully-converged checkpoint, with a deliberately gentle learning-rate
re-warm (~7% of peak) so the perceptual term could move the weights without
undoing the existing pixel/structural convergence. This improved LPIPS by
46% (0.2506 → 0.1341) for a 0.15 dB PSNR cost — a favorable trade given LPIPS
is one of the three graded metrics. This is the model shipped as
`weights/final_model.pt`.

## Training infrastructure

- **Correct schedule resumption.** `CosineAnnealingLR`'s `state_dict()`
  stores its original `T_max`; naively resuming with a new `--epochs` value
  would silently keep the original horizon rather than the intended one,
  driving the schedule past its period. Resuming instead rescales `T_max` to
  the new budget and reseeds the learning rate at the closed-form cosine
  value for the resumed position, so extending a finished run's schedule
  anneals correctly.
- **Deterministic, checkpoint-embedded splits.** Every checkpoint embeds the
  exact train/validation sample IDs it was produced from
  (`dataset.deterministic_split`, seeded), so a resumed run never trains or
  validates on a different split than the one its saved metrics came from.
- **Device-portable checkpoints.** Checkpoints store CPU-normalized state
  dictionaries and move freely between CUDA, Apple MPS, and CPU; device
  selection is automatic (CUDA → MPS → CPU) with mixed precision enabled
  only on CUDA.

## Evaluation protocol

All reported numbers use a deterministic, seeded 85/15 train/validation
split built from the genuine paired training data. `evaluate.py --val-split`
reconstructs this split automatically from a checkpoint's own embedded
configuration, so evaluation always runs against images the model provably
never trained on, without depending on a hand-maintained validation
directory.

## Test-time self-ensembling

A D4 self-ensemble (averaging predictions over the 8 flip/rotation-equivariant
transforms of the input — the network is trained with matching augmentation,
so it is approximately equivariant to them) is implemented and available via
`--tta` on `infer.py`/`evaluate.py`. Measured against the final model, it
improves PSNR/SSIM (28.99 → 29.15 dB, 0.8247 → 0.8285) but *worsens* LPIPS
(0.1341 → 0.1603) — it partially undoes the fine-tuning result above, since
averaging multiple predictions is itself a smoothing operation. It is
implemented and correct but not used for the submitted results, since LPIPS
is one of the three graded metrics and the trade isn't clearly favorable.

## Results

Final model (`weights/final_model.pt`: medium_nafnet_a + LPIPS fine-tune),
measured on the genuine held-out validation split:

| Metric | Value |
|---|---|
| PSNR | 28.99 dB |
| SSIM | 0.8247 |
| LPIPS | 0.1341 |
| Inference time | ~13 ms/image (Apple MPS) |
