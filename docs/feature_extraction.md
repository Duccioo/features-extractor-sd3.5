# Feature Extraction Documentation

This document explains how the feature extraction pipeline works, specifically focusing on the `src/run_feature_extraction.py` script and the underlying `src/extract_features.py` library.

## Overview

The goal of this tool is to process two sets of images (Real and Fake/Generated) through the Stable Diffusion 3.5 model and extract internal representations (features) for analysis. These features can be used to study the differences between real and generated images.

## Scripts

### `src/run_feature_extraction.py`

This is the main Command Line Interface (CLI) entry point. It orchestrates the entire process:
1.  Sets up logging (saving output to `execution.log`).
2.  Loads the SD3.5 model using `utils.model`.
3.  Iterates through the provided image directories.
4.  Calls the feature extraction logic.
5.  Saves metadata about the experiment (arguments, timestamp, system info) to `experiment_config.json`.

#### Usage

```bash
python src/run_feature_extraction.py \
    --model_path <path_to_safetensors> \
    --real_images_path <path_to_real> \
    --fake_images_path <path_to_fake> \
    --output_path <output_dir>
```

#### Key Arguments

*   `--model_path`: Absolute path to the SD3.5 checkpoint file (e.g., `sd3.5_large.safetensors`).
*   `--real_images_path`: Directory containing real images (JPG/PNG).
*   `--fake_images_path`: Directory containing generated images (JPG/PNG).
*   `--output_path`: Directory where features and logs will be saved.
*   `--image_size`: Size to resize images to before processing (default: 512).
*   `--extract_attention`: Flag. If present, attention maps will also be saved. **Note:** This consumes a lot of storage.
*   `--preprocessing_mode`: Preprocessing strategy (default: `imagenet_style`).
*   `--jpeg_aug_real` / `--jpeg_aug_fake`: Apply JPEG compression augmentation to input images (0 for off, 1 for on).
*   `--mean_pooling_only`: Flag. If present, applies spatial mean pooling to reduce feature tensor size from `[1, seq_len, dim]` to `[1, dim]`. This significantly reduces disk space usage while preserving the main feature representation.

### `src/extract_features.py`

This is the core library responsible for the actual extraction logic. It:
1.  Hooks into the SD3.5 model layers (specifically the MM-DiT blocks).
2.  Captures `hidden_states` (the latent representation of the image).
3.  Captures `context` (text embedding interactions, though often empty for unconditional generation).
4.  Optionally captures `attention` weights (specifically layers 0, 1, 2, 12, 23, 35, 36, 37).

## Output Structure

The output directory will be organized as follows:

```
output_path/
├── execution.log           # Full log of the run
├── experiment_config.json  # Metadata and arguments
├── hidden_x/              # Image latent features
│   ├── real/
│   └── fake/
├── hidden_context/        # Context features
│   ├── real/
│   └── fake/
└── attention/             # (Optional) Attention maps
    ├── real/
    └── fake/
```

The features are typically saved as PyTorch tensors (`.pt` files).

## Feature Tensor Shapes

By default, feature tensors have shape `[1, seq_len, dim]` where:
- `seq_len` is the sequence length (e.g., 1024 for 512x512 images)
- `dim` is the hidden dimension (e.g., 1536 for SD3.5 medium)

With `--mean_pooling_only`, features are reduced to `[1, dim]` by averaging over the spatial dimension. This reduces disk usage by approximately 1000x while retaining the global feature representation.
