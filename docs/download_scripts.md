# Download Scripts Documentation

The `src/download` directory contains several utility scripts for setting up your environment and acquiring data.

## Model Download
### `src/download/download_models.py`

This script is the primary tool for downloading Stable Diffusion 3.5 weights from Hugging Face.

**Usage:**
```bash
python src/download/download_models.py
```

**Features:**
*   Checks for the `huggingface_hub` library.
*   Downloads `sd3.5_large.safetensors` by default.
*   Can be configured to download other variants (Medium, Turbo) if modified or extended.
*   Handles authentication tokens if required (ensure you are logged in via `huggingface-cli login`).

## Dataset Downloads

The other scripts in this folder are designed to download specific datasets commonly used in deepfake detection and image generation research.

*   **`download_genimage.py`**: Downloads the GenImage dataset (ImageNet, VQDM, SD, etc.).
*   **`download_echodataset.py`**: Helper for discharging EchoDataset.
*   **`download_tiny_genimage.py`**: Downloads a smaller subset of GenImage for quick testing.
*   **`download_unbiased_genimage.py`**: Downloads the "Unbiased" version of the GenImage dataset.
*   **`download_nanobanana.py`**: Specific script for the NanoBanana dataset.

These scripts generally require internet access and may require valid URLs or dataset permissions depending on the specific source.
