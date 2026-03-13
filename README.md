<div align="center">

![Features Extractor Cover](assets/immagine_copertina.png)

# 🎨 Features Extractor for Stable Diffusion 3.5

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stable Diffusion 3.5](https://img.shields.io/badge/SD-3.5-purple.svg?logo=stability-ai&logoColor=white)](https://stability.ai/)

*🔬 Extract internal features (hidden states and attention maps) from Stable Diffusion 3.5's MM-DiT for deepfake detection and diffusion model research.*

</div>

---

## 🧠 What is this?

This library hooks into the joint transformer blocks (MM-DiT) of Stable Diffusion 3.5 and extracts the **hidden states** and **attention maps** produced when an image is passed through the model. These internal representations capture rich structural and semantic information that can be used for:

-   **AI-generated image detection** (real vs. fake classification)
-   **Generative model attribution** (identifying which model generated an image)
-   **Diffusion model interpretability** research

The core idea: encode any image into SD3.5's latent space, run a forward pass at a chosen diffusion timestep, and capture the intermediate features from each transformer block.

### Key Features

| Feature | Description |
|---------|-------------|
| **Flexible layer selection** | Extract from any subset of the 38 joint blocks (image branch, text branch, or both) |
| **Timestep control** | Choose the diffusion timestep (0–1000) to control noise level during extraction |
| **Text conditioning** | Use precomputed text embeddings or empty prompts as conditioning |
| **Mean pooling** | Optionally reduce spatial dimensions for lightweight feature vectors |
| **Batch processing** | Multi-image batching with configurable `batch_size` and `num_workers` |
| **Safetensors output** | Single `.safetensors` file per image, namespaced keys (`hidden_x__*`, `hidden_context__*`, `attention__*`) |
| **Multiple preprocessing modes** | `imagenet_style`, `brutal_resize`, `crop_100_then_resize`, `none` |
| **JPEG augmentation** | On-the-fly random JPEG compression to mitigate format bias |
| **Corrupted image handling** | Skips broken/truncated images gracefully |

---

## 📋 Setup

### 1. Clone with Submodules

The official [Stable Diffusion 3.5](https://github.com/Stability-AI/sd3.5) code is included as a git submodule:

```bash
git clone --recurse-submodules https://github.com/Duccioo/features-extractor-sd3.5.git
cd features-extractor-sd3.5
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

### 2. Install

**Option A — pip install (recommended):**
```bash
pip install .
```
This installs the package as `sd35-feature-extractor` and provides the `sd35-extract` CLI command.

**Option B — editable install (for development):**
```bash
pip install -e .
```

**Option C — requirements only:**
```bash
pip install -r requirements.txt
```
Then install [PyTorch](https://pytorch.org/get-started/locally/) separately.

### 3. Download Models

Download SD3.5 model weights from HuggingFace using the included helper script:

```bash
# Interactive TUI (recommended — guides you through model selection)
python src/download/download_models.py

# Or directly download a specific model
python src/download/download_models.py --model large          # SD3.5 Large (8B params)
python src/download/download_models.py --model medium         # SD3.5 Medium (2.5B params)
python src/download/download_models.py --model large-turbo    # SD3.5 Large Turbo

# Download multiple models + ControlNets
python src/download/download_models.py --model large medium --controlnets blur canny depth

# Download only the text encoders (CLIP-L, CLIP-G, T5-XXL)
python src/download/download_models.py --only-encoders

# List all available models
python src/download/download_models.py --list
```

> **Note:** You need a HuggingFace token. Set it via `HUGGINGFACE_TOKEN` env variable or pass `--token`.  
> After download, text encoders are automatically moved to the root `models/` folder and `t5xxl_fp16` is renamed to `t5xxl`.

---

## 🚀 Feature Extraction

### The `extract_features()` API

The core function in [`src/extract_features.py`](src/extract_features.py) gives full control over how features are extracted. You can use it **programmatically from any Python project**:

```python
from sd35_extractor.extract_features import extract_features

# Minimal usage — extract from a folder of images
stats = extract_features(
    images_dir="path/to/images",
    output_dir="output_features",
    category="real",            # subfolder name in output
    model_path="models/sd3.5_large.safetensors",
)
```

#### Full parameter reference

```python
stats = extract_features(
    # === Required ===
    images_dir="path/to/images",
    output_dir="output_features",

    # === Model (provide model+vae OR model_path) ===
    model=None,                          # pre-loaded model object
    vae=None,                            # pre-loaded VAE object
    model_path="models/sd3.5_large.safetensors",  # auto-loads if model/vae not given

    # === Output ===
    category="real",                     # output subfolder name (e.g. "real", "fake", "sd3.5")

    # === Diffusion parameters ===
    timestep=0,                          # 0 = clean latents, 100 = light noise, 500 = heavy noise
    text_embedding_path=None,            # path to precomputed text embeddings (.pt)
    text_embedding_prompt="",            # which prompt key to use from the embedding file

    # === Layer selection (use -1 for last available layer) ===
    selected_layers_x=[0, -1],           # image branch layers (0–37 for Large model)
    selected_layers_context=[0, -1],     # text branch layers (0–36, last block is pre_only)
    selected_layers_attention=[0, -1],   # attention layers (only if extract_attention=True)

    # === Feature options ===
    extract_attention=False,             # also extract attention weights (⚠️ large files!)
    apply_mean=True,                     # save the mean across all layers (middle_avg)
    mean_pooling_only=False,             # [1, seq_len, dim] → [1, dim] (huge disk savings)
    skip_context=False,                  # skip text branch features (~50% savings)

    # === Preprocessing ===
    image_size=512,
    preprocessing_mode="imagenet_style", # "imagenet_style" | "brutal_resize" | "crop_100_then_resize" | "none"
    jpeg_aug=False,                      # on-the-fly JPEG compression (bias mitigation)

    # === Performance ===
    batch_size=1,                        # increase for faster processing (uses more VRAM)
    num_workers=4,                       # DataLoader workers
    torch_compile=False,                 # torch.compile for faster inference
    num_images=-1,                       # -1 = all, or limit to N images

    # === Device ===
    device=None,                         # auto-detect CUDA/CPU
    dtype=torch.float16,
)
```

#### Reusing a loaded model across multiple calls

```python
from sd35_extractor.utils.model import load_sd35_model, validate_model_loading
import torch

device = torch.device("cuda")
model, vae = load_sd35_model("models/sd3.5_large.safetensors", device, torch.float16)
validate_model_loading(model.diffusion_model, "MM-DiT")
validate_model_loading(vae, "VAE")

# Extract for different datasets, reusing the same model
for folder, cat in [("data/real", "real"), ("data/fake_sd", "fake_sd"), ("data/fake_mj", "fake_mj")]:
    extract_features(
        images_dir=folder,
        output_dir="features_out",
        category=cat,
        model=model,
        vae=vae,
        timestep=100,
        mean_pooling_only=True,
        selected_layers_x=[0, -1],
        selected_layers_context=[0, -1],
    )
```

### CLI Usage

After `pip install .`:

```bash
sd35-extract \
    --model_path models/sd3.5_large.safetensors \
    --real_images_path path/to/real/images \
    --fake_images_path path/to/fake/images \
    --output_path output_features \
    --timestep 100 \
    --mean_pooling_only \
    --image_size 512
```

Or directly with Python:

```bash
python src/run_feature_extraction.py \
    --model_path models/sd3.5_large.safetensors \
    --real_images_path path/to/real/images \
    --output_path output_features
```

> **Tip:** At least one of `--real_images_path` or `--fake_images_path` must be provided. The CLI also saves an `experiment_config.json` and `execution.log` in the output folder.

---

## 📦 Output Format

Each image produces a single `.safetensors` file with namespaced keys:

```
output_features/
├── real/
│   ├── image001.safetensors
│   ├── image002.safetensors
│   └── ...
└── fake/
    ├── image001.safetensors
    └── ...
```

**Loading extracted features:**

```python
from sd35_extractor.extract_features import load_features

features = load_features("output_features/real/image001.safetensors")

# Access feature groups
hidden_x_avg = features["hidden_x"]["middle_avg"]       # mean across all layers
hidden_x_l0  = features["hidden_x"]["layer_0"]          # first block
hidden_x_l37 = features["hidden_x"]["layer_37"]         # last block

ctx_avg      = features["hidden_context"]["middle_avg"]  # text branch mean
attn_joint   = features["attention"]["layer_0_joint"]    # attention (if extracted)
```

**Tensor shapes** (SD3.5 Large at 512×512, without mean pooling):

| Key | Shape | Description |
|-----|-------|-------------|
| `hidden_x__middle_avg` | `[1, 1024, 1536]` | Mean of image branch across all 38 layers |
| `hidden_x__layer_0` | `[1, 1024, 1536]` | Image branch at block 0 |
| `hidden_context__middle_avg` | `[1, 154, 1536]` | Mean of text branch across 37 layers |
| `hidden_context__layer_0` | `[1, 154, 1536]` | Text branch at block 0 |

With `mean_pooling_only=True`, spatial dims are averaged: `[1, 1024, 1536]` → `[1, 1536]`.

---

## 📝 Text Embeddings

Pre-compute text embeddings to use as conditioning during feature extraction:

```bash
# Default: encodes "" (empty) and "a photo"
python src/extract_text_embedding.py \
    --model_path models/sd3.5_large.safetensors \
    --output text_embeddings.pt

# Custom prompts
python src/extract_text_embedding.py \
    --model_path models/sd3.5_large.safetensors \
    --prompts "" "a photo" "a painting of a landscape" \
    --output text_embeddings.pt

# With mean pooling
python src/extract_text_embedding.py \
    --model_path models/sd3.5_large.safetensors \
    --mean_pooling_only \
    --output text_embeddings_pooled.pt
```

**Using pre-computed embeddings in feature extraction:**

```python
stats = extract_features(
    images_dir="path/to/images",
    output_dir="output",
    category="real",
    model_path="models/sd3.5_large.safetensors",
    text_embedding_path="text_embeddings.pt",
    text_embedding_prompt="a photo",       # must match a prompt encoded in the file
)
```

**Loading embeddings directly:**

```python
from sd35_extractor.extract_text_embedding import load_text_embeddings

cond = load_text_embeddings("text_embeddings.pt", "a photo", device="cuda")
# cond["c_crossattn"]  →  [1, 154, 4096]
# cond["y"]            →  [1, 2048]
```

---

## 📊 Dataset Download Scripts

The `src/download/dataset/` folder contains ready-to-use scripts for downloading popular AI-generated image detection benchmarks.

### Unified TUI

```bash
# Interactive menu to download models and datasets
python src/download/dataset/download_tui.py
```

### Individual Dataset Scripts

| Script | Dataset | Source | Size |
|--------|---------|--------|------|
| `download_genimage.py` | [GenImage](https://github.com/GenImage-Dataset/GenImage) (1M+ images, 8 generators) | Google Drive | ~60–100 GB |
| `download_tiny_genimage.py` | [Tiny GenImage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) (5K/generator) | Kaggle | ~8 GB |
| `download_unbiased_genimage.py` | [Unbiased GenImage](https://github.com/gendetection/UnbiasedGenImage) (with metadata.csv) | Harvard Dataverse | ~500 GB |
| `download_openfake.py` | [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (20+ generators, 1.9M images) | HuggingFace | ~1 TB |
| `download_echodataset.py` | [Echo-4o-Image](https://huggingface.co/datasets/Yejy53/Echo-4o-Image) | HuggingFace | — |
| `download_nanobanana.py` | [NanoBanana](https://huggingface.co/datasets/bitmind/nano-banana) | HuggingFace | — |

**Quick examples:**

```bash
# Tiny GenImage (good for quick experiments)
python src/download/dataset/download_tiny_genimage.py --output ./data/tiny-genimage

# OpenFake — small subset for testing
python src/download/dataset/download_openfake.py -o data/OpenFake-mini \
    --limit 100 --limit-per-model 20

# OpenFake — only Stable Diffusion 3.5 fakes
python src/download/dataset/download_openfake.py -o data/OpenFake \
    --models "Stable Diffusion 3.5" --limit-per-model 500

# GenImage — show info and download instructions
python src/download/dataset/download_genimage.py --info

# Unbiased GenImage — download only metadata.csv
python src/download/dataset/download_unbiased_genimage.py --output ./data --metadata-only
```

> **Note:** Some scripts require additional credentials:
> - **HuggingFace** scripts: set `HUGGINGFACE_TOKEN` env variable
> - **Kaggle** scripts: configure `~/.kaggle/kaggle.json` ([instructions](https://www.kaggle.com/docs/api))

---

## 🗂️ Project Structure

```
features-extractor-sd3.5/
├── src/
│   ├── extract_features.py          # Core extraction API (extract_features, load_features)
│   ├── run_feature_extraction.py    # CLI wrapper for batch real/fake extraction
│   ├── extract_text_embedding.py    # Text embedding extraction tool
│   ├── utils/
│   │   ├── model.py                 # Model loading (load_sd35_model)
│   │   ├── features.py              # Feature hooks (create_dual_feature_hook, AttentionCapture)
│   │   ├── conditioning.py          # Empty conditioning creation
│   │   ├── preprocessing.py         # Image preprocessing (StandardPreprocessor)
│   │   ├── system.py                # Logging and system info
│   │   └── ...
│   └── download/
│       ├── download_models.py       # SD3.5 model downloader (HuggingFace)
│       └── dataset/                 # Dataset download scripts
│           ├── download_tui.py      # Unified interactive TUI
│           ├── download_genimage.py
│           ├── download_tiny_genimage.py
│           ├── download_openfake.py
│           ├── download_unbiased_genimage.py
│           ├── download_echodataset.py
│           └── download_nanobanana.py
├── sd3.5/                           # Official SD3.5 code (git submodule)
├── models/                          # Model weights (after download)
├── pyproject.toml                   # Package config (sd35-feature-extractor)
└── requirements.txt
```

---

## 📚 Documentation

For more detailed guides, see the [`docs/`](docs/) folder:

-   [Feature Extraction Guide](docs/feature_extraction.md) — deep dive into the extraction pipeline and output format
-   [Text Embedding Guide](docs/text_embedding.md) — how to pre-compute and use text embeddings
-   [Download Scripts](docs/download_scripts.md) — details on model and dataset downloaders

## 📄 License

This project is licensed under the [MIT License](LICENSE).
