# Final model weights

`final_model.pt` is the submitted checkpoint (medium_nafnet_a, LPIPS-fine-tuned; see [`docs/methodology.md`](../docs/methodology.md)). It's committed directly — 7.3MB, well under GitHub's single-file limit, so no Git LFS or external hosting is needed.

`infer.py` defaults to this path; pass `--checkpoint` to point at a different one.
