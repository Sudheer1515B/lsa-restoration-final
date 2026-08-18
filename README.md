# KLA AI-Based Image Restoration Submission

## Submission contents

```text
team_name/
├── run.py
├── requirements.txt
├── README.md
└── models/
    ├── final_model.pt
    ├── nafnet_sr.py
    └── __init__.py
```

`models/final_model.pt` contains the trained weights and model configuration.
The accompanying source files reconstruct the network locally, so no model,
weight, API key, or internet download is needed at inference time.

## Setup

Install the pinned dependencies once on the evaluation machine:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, create and use the environment with:

```bat
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation on an NVIDIA evaluation machine. The
script selects CUDA automatically when it is available, otherwise it runs on
CPU.

## Run

```bash
python run.py <input-dir> <output-dir>
```

If `python` is not on the path, use the environment interpreter instead:

```bash
.venv/bin/python run.py <input-dir> <output-dir>
```

`run.py` reads every `.npy` file directly in `<input-dir>`, creates
`<output-dir>` if necessary, and writes one identically named `.npy` file per
input. Outputs are finite grayscale `float32` arrays in `[0, 1]` at the
checkpoint's 2× target resolution (KLA inputs: 128×128 to 256×256).
