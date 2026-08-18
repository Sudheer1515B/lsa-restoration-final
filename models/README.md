# Submission model files

`final_model.pt` is the self-contained NAFNet-SR checkpoint used by
`../run.py`. It includes both the trained weights and the model configuration.
`nafnet_sr.py` is the small model definition required to reconstruct those
weights. Together, these files ensure the submission runs without the
development-only `model/` directory or any training files.
