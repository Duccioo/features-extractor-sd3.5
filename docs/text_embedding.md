# Text Embedding Extraction Documentation

This document describes the `src/extract_text_embedding.py` script, which is used to pre-compute text embeddings from Stable Diffusion 3.5's text encoders (CLIP L, CLIP G, and T5).

## Purpose

Stable Diffusion 3.5 uses three different text encoders to understand prompts. Loading all these models into memory just to extract features from images can be resource-intensive or unnecessary if you are using a fixed prompt (or no prompt/empty string) for all images.

This script allows you to extract the embeddings once and save them to a `.pt` file. You can then load this file during feature extraction instead of running the full text encoding pipeline every time.

## Usage

### Basic Usage

To extract embeddings for default prompts (often just an empty string or simple labels):

```bash
python src/extract_text_embedding.py \
    --model_path models/sd3.5_large.safetensors \
    --output text_embeddings.pt
```

### Custom Prompts

You can specify a list of prompts to encode:

```bash
python src/extract_text_embedding.py \
    --model_path models/sd3.5_large.safetensors \
    --output my_embeddings.pt \
    --prompts "a photo of a cat" "scenery"
```

### Reduced Feature Size (Mean Pooling)

To save disk space, you can average the sequence embedding over the sequence dimension (154 tokens) using `--mean_pooling_only`. This reduces the `c_crossattn` tensor from `[1, 154, 4096]` to `[1, 4096]`.

```bash
python src/extract_text_embedding.py \
    --model_path models/sd3.5_large.safetensors \
    --output text_embeddings_pooled.pt \
    --mean_pooling_only
```

## How It Works

1.  **Loads SD3Inferencer**: It initializes the Stable Diffusion 3.5 inference pipeline, which loads the text encoders (CLIP-L/G, T5).
2.  **Encodes Prompts**: It passes the strings through the encoders to get the pooled and sequence embeddings.
3.  **Saves to Disk**: The result is a dictionary containing the tensors, saved via `torch.save`.

## Integration

These embeddings is used by the feature extractor to condition the model. If you provide a path to these embeddings (via `--text_embedding_path` in `run_feature_extraction.py`), the system skips loading the text encoders and uses these pre-computed tensors directly.
