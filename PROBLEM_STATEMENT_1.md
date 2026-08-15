# Problem Statement 1: AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Background

In semiconductor manufacturing, microscopic inspection images are used to measure and verify chip quality at every stage of production. These images must be extremely sharp and clean because a single pixel of noise or a small loss of detail can hide a defect that causes a chip to fail.

In practice, inspection images are often degraded by two types of signal loss:

- **Speckle Noise** — random pixel-level noise that makes the image look ‘grainy.’ This noise can actually push pixel values beyond the true image range, meaning some pixels appear brighter or darker than they actually are in reality.
- **Spatial Resolution Reduction** — the image has been ‘shrunk’ (downsampled), losing fine detail. A 512×512 pixel ground truth image becomes a blurry 256×256 pixel image, or a 256×256 becomes 128×128. Details that were visible at full resolution are gone.

Currently, engineers work with these degraded images and make do with the noise and lost detail. AI-powered restoration can recover information that appears lost removing noise while sharpening detail back to the original resolution.

## Description

You will receive a training dataset of paired images: for each sample, you get a degraded image (noisy + low resolution) and the corresponding ground truth image (clean + full resolution). Your job is to train an AI model that learns to reverse the degradation taking a bad image as input and producing a restored image that matches the ground truth as closely as possible.

### The degradation types your model must handle

| Degradation Type | What It Looks Like | What Your Model Must Fix |
| --- | --- | --- |
| Speckle Noise | Random pixel-level noise, image looks grainy. Some pixel values are pushed beyond the true range of the image. | Remove the grain while preserving the real image details underneath. Do not blur the image to remove noise, that destroys useful information. |
| Gaussian Noise | Image appears soft and hazy edges and fine structures lose sharpness. | Restore edge sharpness and contrast without introducing artificial patterns or ringing. |
| Spatial Resolution Reduction (Super-Resolution) | Image has been downsampled: 512×512 → 256×256, or 256×256 → 128×128. Fine details are lost. | Upscale the image back to the original resolution (512×512 or 256×256) while reconstructing the fine details that were lost during downsampling. |

- Your model must handle **ALL** degradation types simultaneously a single image may have speckle noise **AND** reduced resolution at the same time.
- The test set will include images from different sources than the training data (out-of-distribution). Your model must generalize not just memorize the training examples.
- Speed matters. Your model will be benchmarked on inference time. A model that produces great results but takes 10 minutes per image is less useful than one that produces good results in 10 seconds.

### Training Data — What You Get

KLA will provide a paired training dataset. For each sample, you receive:

| What You Get | Resolution | Description |
| --- | --- | --- |
| Ground Truth Image (clean, full resolution) | 512×512 pixels or 256×256 pixels | The ‘correct answer’ — the image as it should look. High resolution, high signal-to-noise ratio. This is what your model's output should match. |
| Degraded Image (noisy, low resolution) | 256×256 pixels or 128×128 pixels | The input to your model — noisy and downsampled. Your model takes this as input and must produce an output that matches the ground truth. |

**Important data notes:**

- The degraded image intensity range may **EXCEED** the ground truth range this is expected behaviour caused by speckle noise pushing pixel values beyond the original signal. Your model must handle this.
- The images come from diverse data origins different types of semiconductor structures. Your model should generalize across these variations, not overfit to one type.
- Images are grayscale (single channel). Colour images are **NOT** part of this challenge.

### Test Data — What Comes Later

After the training phase, KLA will release a test dataset. The test set contains:

- **In-distribution samples:** images similar to what you trained on. Tests accuracy.
- **Out-of-distribution samples:** images from different sources than the training data. Tests generalization and robustness whether your model can handle image types it has never seen.

## Submission Requirements

All submissions are made through the i4C hackathon portal. You must submit a PPT/PDF using the provided Idea Submission Template **AND** a GitHub repository link. Here is exactly what each must contain:

### Component 1 — PPT/PDF Submission (Using Idea Submission Template)

Use the provided Hackathon Idea Submission Template. Fill in each slide as follows:

| Template Slide | What to Fill In |
| --- | --- |
| Slide 1: Team Details | Team name, member names, roles, college name, contact details. |
| Slide 2: Problem Statement Addressed | Select ‘AI-Based Restoration of Degraded Images.’ Describe in your own words why this problem matters in semiconductor manufacturing. |
| Slide 3: Idea Description | Your key concept and approach – what type of AI model did you choose? Why? How does it address all 3 degradation types (speckle, Gaussian, super-resolution)? |
| Slide 4: Proposed Solution | Detailed solution – model architecture, training strategy, loss function design, data augmentation approach. Include a system/pipeline diagram. |
| Slide 5: Innovation & Uniqueness | What makes your approach different? Did you design a novel loss function? A unique data augmentation strategy? A faster inference pipeline? |
| Slide 6: Results | SSIM, pSNR, LPIPS scores on your test split. Before/after image comparisons (degraded input → your restored output → ground truth). Confusion-free visual evidence that your model works. |
| Slide 7: Technology & Feasibility | Tech stack used (PyTorch/TensorFlow/other), hardware used for training (GPU type, cloud platform), training time, model size, inference time per image. |
| Slide 8: GitHub & Video Link | GitHub repository link (mandatory). Video link showing your model running (optional but recommended). |
| Slide 9: References | Research papers, datasets, tools referenced. |

**FILE FORMAT:** Save as PDF before uploading. File naming convention: `TeamName_KLA_PS01` (e.g., `VisionForge_KLA_PS01.pdf`). Maximum 8-9 slides. Remove the instruction slide.

### Component 2 — GitHub Repository (Mandatory)

Your GitHub repository must be public and must contain:

| # | Repository Content | Details |
| --- | --- | --- |
| 1 | `README.md` | Complete setup instructions. A reviewer must be able to clone your repo and run inference from the README without contacting you. |
| 2 | Evaluation Script (standalone `.py`) | A Python script (**NOT** a Jupyter notebook) that accepts: (a) path to test images directory, (b) path to output directory. It loads your trained model, runs inference on all input images, and writes restored outputs to the specified directory. Must run without manual edits. |
| 3 | Training Script | Python script or Jupyter notebook that reproduces your training process from scratch. |
| 4 | Trained Model Weights | Your final trained model file (in a format your evaluation script can load — `.pt`, `.onnx`, `.h5`, etc.). Must be downloadable (use Git LFS or link to Google Drive/HuggingFace if file is large). |
| 5 | Restored Test Outputs | A folder containing your model's output on the test set — the actual restored images your model produced. |
| 6 | `requirements.txt` | Complete pip freeze output from your training environment. Required for reproducibility. |

**CRITICAL:** The evaluation script is the most important file in your repository. It will be used **AS-IS** by KLA's benchmarking team to measure your model's quality scores and inference time on the H100 GPU. If your script does not run without manual edits, your submission cannot be benchmarked and unscored submissions cannot win. Test your script on a fresh machine before submitting.

## Supporting Materials

- [Detailed Problem Statement Explanation PPT](https://i4c.in/wp-content/uploads/2026/08/7b675083-e081-47d3-8c55-fde76a77b673.pptx)
- [Download Dataset](https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=drive_link)
- [Know More about KLA](https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=drive_link)
