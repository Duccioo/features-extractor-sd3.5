# Features Extractor for Stable Diffusion 3.5

This repository contains tools and scripts to extract internal features (hidden states and attention maps) from Stable Diffusion 3.5 models. It is designed to facilitate analysis and experiments involving real and generated images.

## Prerequisites

Before using this code, you must set up the environment and download the necessary dependencies.

### 1. Download Stable Diffusion 3.5 Code
This repository depends on the official Stable Diffusion 3.5 implementation. You must download the code from Stability AI.

1.  Go to the root of this repository.
2.  Clone the `sd3.5` repository:
    ```bash
    git clone https://github.com/Stability-AI/sd3.5
    ```
    **Note:** The scripts expect the `sd3.5` folder to be located in the root directory of this project (alongside `src`, `LICENSE`, etc.).

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install torch torchvision pillow tqdm huggingface_hub python-dotenv
```

## Setup & Models

### Download Models
You need to download the Stable Diffusion 3.5 model weights (e.g., `sd3.5_large.safetensors`). A helper script is provided in the `src/download` folder.

Run the download script:
```bash
python src/download/download_models.py
```
*Note: This script uses Hugging Face. You may need to set up your Hugging Face token if the models require authentication.*

## Usage

### Feature Extraction
The main entry point for extracting features is `src/run_feature_extraction.py`. This script processes two directories of images (Real and Fake) and extracts features for analysis.

**Basic Usage:**

```bash
python src/run_feature_extraction.py \
    --model_path models/sd3.5_large.safetensors \
    --real_images_path path/to/real/images \
    --fake_images_path path/to/fake/images \
    --output_path output_features
```

**Key Arguments:**
-   `--model_path`: Path to the downloaded model checkpoint (`.safetensors`).
-   `--real_images_path`: Folder containing the real images.
-   `--fake_images_path`: Folder containing the fake/generated images.
-   `--output_path`: Destination folder for extracted features.
-   `--extract_attention`: Optional. Add this flag to extract attention maps (WARNING: uses significant disk space).
-   `--image_size`: Resolution to resize images (default: 512).
-   `--num_images`: Limit the number of images to process (-1 for all).

### Other Tools

#### Extract Text Embeddings
You can pre-compute text embeddings for use in conditioning:
```bash
python src/extract_text_embedding.py --model_path models/sd3.5_large.safetensors --output text_embeddings.pt
```

#### Download Datasets
The `src/download/` folder contains scripts to help you download various datasets used for training or testing, such as:
-   `download_genimage.py`
-   `download_echodataset.py`
-   `download_tiny_genimage.py`

## Documentation

For more detailed information on how the scripts work, please refer to the `docs/` folder:

-   [Feature Extraction Guide](docs/feature_extraction.md): Details on `run_feature_extraction.py` and output format.
-   [Text Embedding Guide](docs/text_embedding.md): How to pre-compute text embeddings.
-   [Download Scripts](docs/download_scripts.md): Information about model and dataset downloaders.
