"""
SD3.5 Feature Extraction Library

Extracts internal features (hidden states and attention) from Stable Diffusion 3.5.
Exposes 'extract_features' function for folder processing.
"""

import os
import sys
from glob import glob
import torch
from PIL import Image, ImageFile
from tqdm import tqdm


# Allow loading truncated images (some images in large datasets may be corrupted)
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Add the sd3.5 directory to the path BEFORE importing modules that depend on it
# sd3.5 is in features-extractor-sd3.5/sd3.5
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_current_dir)  # features-extractor-sd3.5
sys.path.insert(0, os.path.join(_repo_root, "sd3.5"))

# ---
# Load precomputed text embeddings when provided
from extract_text_embedding import load_text_embeddings

from sd3_impls import SD3LatentFormat, ModelSamplingDiscreteFlow
from utils.model import load_sd35_model, validate_model_loading
from utils.features import create_dual_feature_hook, AttentionCapture
from utils.conditioning import create_empty_conditioning
from utils.preprocessing import StandardPreprocessor


# Default attention layers to save (if not specified)
# SD3.5 has 38 joint blocks (layers 0-37)
ATTENTION_LAYERS_TO_SAVE = [0, 1, 2, 12, 23, 35, 36, 37]


def setup_output_directories_for_category(
    output_path, category, extract_attention=False, skip_context=False
):
    """Create output directory structure for features specific to a category."""
    os.makedirs(output_path, exist_ok=True)

    # Hidden features (x) - image branch
    os.makedirs(os.path.join(output_path, "hidden_x", category), exist_ok=True)
    # Hidden features (context) - text branch
    if not skip_context:
        os.makedirs(
            os.path.join(output_path, "hidden_context", category), exist_ok=True
        )
    # Attention weights
    if extract_attention:
        os.makedirs(os.path.join(output_path, "attention", category), exist_ok=True)


def collect_images(images_path, num_images=-1):
    """Collect all image files from a directory."""
    extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    images = []
    for ext in extensions:
        images.extend(glob(os.path.join(images_path, ext)))

    if num_images > 0:
        images = sorted(images)[:num_images]
    return images


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for efficient batch loading of images."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor, image_path, True  # tensor, path, success
        except (OSError, IOError) as e:
            # Return a dummy tensor for failed images
            dummy = torch.zeros(3, 512, 512)
            return dummy, image_path, False


def collate_fn_skip_errors(batch):
    """Custom collate function that filters out failed images."""
    valid_items = [(img, path) for img, path, success in batch if success]
    failed_paths = [path for _, path, success in batch if not success]

    if not valid_items:
        return None, [], failed_paths

    images = torch.stack([item[0] for item in valid_items])
    paths = [item[1] for item in valid_items]
    return images, paths, failed_paths


def extract_features_from_batch(
    image_tensors,
    image_paths,
    model,
    vae,
    sigma,
    device,
    dtype,
    features_x,
    features_context,
    timestep=0,
    extract_attention=False,
    conditioning=None,
):
    """Extract features from a batch of images.

    Returns:
        List of (image_path, agg_x, agg_ctx, attn_weights) tuples
    """
    features_x.clear()
    features_context.clear()

    batch_size = image_tensors.shape[0]
    image_tensors = image_tensors.to(device, dtype)

    attention_weights = {}

    with torch.no_grad():
        latents = vae.encode(image_tensors)
        latents = SD3LatentFormat().process_in(latents)

        if timestep > 0:
            noise = torch.randn_like(latents)
            noised_latents = sigma * noise + (1.0 - sigma) * latents
        else:
            noised_latents = latents

        if conditioning is not None:
            cond_context, cond_y = conditioning
            context = cond_context.to(device=device, dtype=dtype)
            y = cond_y.to(device=device, dtype=dtype)
            if context.shape[0] == 1 and batch_size > 1:
                context = context.expand(batch_size, *context.shape[1:])
            if y.shape[0] == 1 and batch_size > 1:
                y = y.expand(batch_size, *y.shape[1:])
        else:
            context, y = create_empty_conditioning(
                batch_size, device, dtype, model=model
            )

        _ = model.apply_model(
            noised_latents,
            sigma.expand(batch_size),
            c_crossattn=context,
            y=y,
        )

        if extract_attention:
            with AttentionCapture(model) as attn_capture:
                _ = model.apply_model(
                    noised_latents,
                    sigma.expand(batch_size),
                    c_crossattn=context,
                    y=y,
                )
                attention_weights.update(attn_capture.attention_weights)

    # Split features by batch index - features are stored with batch dimension
    results = []
    for batch_idx, image_path in enumerate(image_paths):
        # Extract per-image features from batched features
        per_image_x = {k: v[batch_idx : batch_idx + 1] for k, v in features_x.items()}
        per_image_ctx = {
            k: v[batch_idx : batch_idx + 1] for k, v in features_context.items()
        }

        per_image_attn = {}
        if extract_attention:
            per_image_attn = {
                k: v[batch_idx : batch_idx + 1] for k, v in attention_weights.items()
            }

        results.append((image_path, per_image_x, per_image_ctx, per_image_attn))

    return results


def extract_features_from_image(
    image_path,
    model,
    vae,
    transform,
    sigma,
    device,
    dtype,
    features_x,
    features_context,
    timestep=0,
    extract_attention=False,
    conditioning=None,
):
    """Extract features from a single image."""
    features_x.clear()
    features_context.clear()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device, dtype)

    attention_weights = {}

    with torch.no_grad():
        latents = vae.encode(image_tensor)
        latents = SD3LatentFormat().process_in(latents)

        if timestep > 0:
            noise = torch.randn_like(latents)
            noised_latents = sigma * noise + (1.0 - sigma) * latents
        else:
            noised_latents = latents

        batch_size = noised_latents.shape[0]
        if conditioning is not None:
            cond_context, cond_y = conditioning
            context = cond_context.to(device=device, dtype=dtype)
            y = cond_y.to(device=device, dtype=dtype)
            if context.shape[0] == 1 and batch_size > 1:
                context = context.expand(batch_size, *context.shape[1:])
            if y.shape[0] == 1 and batch_size > 1:
                y = y.expand(batch_size, *y.shape[1:])
        else:
            context, y = create_empty_conditioning(
                batch_size, device, dtype, model=model
            )

        _ = model.apply_model(
            noised_latents,
            sigma.expand(batch_size),
            c_crossattn=context,
            y=y,
        )

        if extract_attention:
            with AttentionCapture(model) as attn_capture:
                _ = model.apply_model(
                    noised_latents,
                    sigma.expand(batch_size),
                    c_crossattn=context,
                    y=y,
                )
                attention_weights.update(attn_capture.attention_weights)

    return attention_weights


def aggregate_layer_features(
    features_dict, mean_pooling_only=False, last_layer_only=False
):
    """Aggregate layer features: keep first, last, and average of all layers.

    Args:
        features_dict: Dictionary of layer_name -> tensor.
        mean_pooling_only: If True, apply spatial mean pooling to reduce tensor size.
                          Reduces [1, seq_len, dim] to [1, dim].
        last_layer_only: If True, return only the last layer's features.
                        This significantly reduces disk space (~66% savings).
    """
    if not features_dict:
        return {}

    sorted_keys = sorted(features_dict.keys(), key=lambda x: int(x.split("_")[1]))
    num_layers = len(sorted_keys)

    if num_layers == 0:
        return {}

    def apply_pooling(tensor):
        """Apply spatial mean pooling if enabled."""
        if mean_pooling_only and tensor.dim() == 3:
            # [batch, seq_len, dim] -> [batch, dim]
            return tensor.mean(dim=1)
        return tensor

    aggregated = {}

    # If last_layer_only, return only the last layer
    if last_layer_only:
        aggregated["last"] = apply_pooling(features_dict[sorted_keys[-1]])
        return aggregated

    aggregated["first"] = apply_pooling(features_dict[sorted_keys[0]])
    if num_layers >= 2:
        aggregated["last"] = apply_pooling(features_dict[sorted_keys[-1]])

    all_tensors = [features_dict[k] for k in sorted_keys]
    stacked = torch.stack(all_tensors, dim=0)
    aggregated["middle_avg"] = apply_pooling(stacked.mean(dim=0))

    return aggregated


def aggregate_attention_features(
    attention_dict, layers_to_save=None, mean_pooling_only=False
):
    """Aggregate attention features.

    Args:
        attention_dict: Dictionary of attention weights.
        layers_to_save: List of layer indices to save.
        mean_pooling_only: If True, apply mean pooling to reduce tensor size.
    """
    if not attention_dict:
        return {}

    joint_keys = sorted(
        [k for k in attention_dict if k.endswith("_joint")],
        key=lambda x: int(x.split("_")[1]),
    )
    self_keys = sorted(
        [k for k in attention_dict if k.endswith("_self")],
        key=lambda x: int(x.split("_")[1]),
    )

    aggregated = {}

    def apply_pooling(tensor):
        """Apply mean pooling if enabled. For attention [batch, heads, seq, seq] -> [batch, heads]."""
        if mean_pooling_only and tensor.dim() >= 3:
            # Average over spatial dimensions
            while tensor.dim() > 2:
                tensor = tensor.mean(dim=-1)
        return tensor

    def aggregate_attention_type(keys, attn_type):
        if not keys:
            return

        selected_keys = keys
        if layers_to_save is not None:
            wanted = set(layers_to_save)
            selected_keys = [k for k in keys if int(k.split("_")[1]) in wanted]

        for key in selected_keys:
            layer_num = key.split("_")[1]
            aggregated[f"layer_{layer_num}_{attn_type}"] = apply_pooling(
                attention_dict[key]
            )

        if len(keys) > 1:
            all_tensors = [attention_dict[k] for k in keys]
            stacked = torch.stack(all_tensors, dim=0)
            aggregated[f"middle_avg_{attn_type}"] = apply_pooling(stacked.mean(dim=0))

    aggregate_attention_type(joint_keys, "joint")
    aggregate_attention_type(self_keys, "self")

    return aggregated


def save_features(features_dict, output_path, feature_type, category, image_name):
    """Save extracted features to disk."""
    for name, tensor in features_dict.items():
        save_path = os.path.join(
            output_path, feature_type, category, f"{image_name}_{name}.pt"
        )
        torch.save(tensor, save_path)


def extract_features(
    images_dir,
    output_dir,
    category="data",
    model=None,
    vae=None,
    model_path=None,
    timestep=0,
    layers_to_save=ATTENTION_LAYERS_TO_SAVE,
    extract_attention=False,
    num_images=-1,
    image_size=512,
    simulate_low_res=False,
    text_embedding_path=None,
    text_embedding_prompt="",
    apply_mean=True,
    preprocessing_mode="genimage_resize",
    jpeg_aug=True,
    mean_pooling_only=False,
    last_layer_only=False,
    skip_context=False,
    batch_size=1,
    num_workers=4,
    torch_compile=False,
    device=None,
    dtype=torch.float16,
):
    """
    Extract features from all images in a folder.

    Args:
        images_dir: Input folder with images.
        output_dir: Base folder for output.
        category: Category name (e.g. "real", "fake") used for output subfolder.
        model: Loaded SD3.5 model. If None, loaded from model_path.
        vae: Loaded VAE.
        model_path: Path to model checkpoint if model not provided.
        timestep: Diffusion timestep (0-1000).
        layers_to_save: List of attention layers to save.
        extract_attention: Boolean to extract attention weights.
        num_images: Max images to process.
        image_size: Target image size.
        simulate_low_res: Whether to simulate low res.
        text_embedding_path: Path to text embeddings.
        text_embedding_prompt: Prompt key.
        apply_mean: Whether to compute/save aggregated mean features.
        preprocessing_mode: Mode for processing ('imagenet_style', 'brutal_resize', etc).
        jpeg_aug: Whether to apply JPEG augmentation/compression.
        mean_pooling_only: If True, apply spatial mean pooling to features before saving.
                          This reduces [1, seq_len, dim] to [1, dim], saving disk space.
        last_layer_only: If True, save only the last layer's features (~66% disk savings).
        skip_context: If True, skip saving hidden_context features (~50% additional savings).
        batch_size: Number of images to process per batch (default: 1).
                   Higher values speed up processing but use more VRAM.
        num_workers: Number of workers for DataLoader (default: 4).
        torch_compile: If True, use torch.compile for faster inference.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Dictionary with statistics (count, shapes).
    """

    # 1. Setup Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model if needed
    if model is None or vae is None:
        if model_path is None:
            raise ValueError(
                "Must provide 'model' and 'vae' objects, OR 'model_path' string."
            )
        print(f"Loading model from {model_path}...")
        model, vae = load_sd35_model(
            model_path, device, dtype, verbose=False
        )  # verbose=False to avoid noise
        validate_model_loading(model.diffusion_model, "MM-DiT")
        validate_model_loading(vae, "VAE")

    # 2.5 Apply torch.compile if requested
    if torch_compile:
        print(
            "Compiling model with torch.compile (this may take a minute on first run)..."
        )
        model.diffusion_model = torch.compile(
            model.diffusion_model, mode="reduce-overhead"
        )
        vae = torch.compile(vae, mode="reduce-overhead")

    # 3. Setup Hooks
    features_x = {}
    features_context = {}
    hooks = []

    # Attach hooks to all blocks
    for i, block in enumerate(model.diffusion_model.joint_blocks):
        hook_fn = create_dual_feature_hook(features_x, features_context, f"block_{i}")
        h = block.register_forward_hook(hook_fn)
        hooks.append(h)

    stats = {"count": 0, "shapes": None}

    try:
        # 4. Preparation
        setup_output_directories_for_category(
            output_dir, category, extract_attention, skip_context
        )

        model_sampling = ModelSamplingDiscreteFlow(shift=3.0)

        if simulate_low_res:
            print(
                "WARNING: --simulate_low_res is deprecated. Using standard 'genimage_resize'."
            )

        preprocessor = StandardPreprocessor(
            image_size=image_size, mode=preprocessing_mode, jpeg_aug=jpeg_aug
        )
        transform = preprocessor.transform

        ts_val = torch.tensor([timestep], device=device, dtype=dtype)
        sigma = model_sampling.sigma(ts_val)

        conditioning = None
        if text_embedding_path:
            cond = load_text_embeddings(
                text_embedding_path, text_embedding_prompt, device=device, dtype=dtype
            )
            conditioning = (cond["c_crossattn"], cond["y"])

        # 5. Collect Images
        images = collect_images(images_dir, num_images)
        print(
            f"Processing {len(images)} images from {images_dir} for category '{category}'..."
        )
        if batch_size > 1:
            print(f"Using batch size: {batch_size}, num_workers: {num_workers}")
        stats["count"] = len(images)

        first_image_shapes = None
        skipped_images = []

        # 6. Process images - batch or single
        if batch_size > 1:
            # Use DataLoader for batch processing
            dataset = ImageDataset(images, transform)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn_skip_errors,
            )

            pbar = tqdm(dataloader, desc=f"Extracting {category} (batch={batch_size})")
            for image_tensors, image_paths, failed_paths in pbar:
                # Track failed images
                for path in failed_paths:
                    skipped_images.append((path, "Failed to load image"))

                if image_tensors is None or len(image_paths) == 0:
                    continue

                try:
                    # Extract features from batch
                    batch_results = extract_features_from_batch(
                        image_tensors,
                        image_paths,
                        model,
                        vae,
                        sigma,
                        device,
                        dtype,
                        features_x,
                        features_context,
                        timestep,
                        extract_attention,
                        conditioning,
                    )

                    # Process each image's results
                    for (
                        image_path,
                        per_image_x,
                        per_image_ctx,
                        per_image_attn,
                    ) in batch_results:
                        # Aggregate features
                        agg_x = aggregate_layer_features(
                            per_image_x,
                            mean_pooling_only=mean_pooling_only,
                            last_layer_only=last_layer_only,
                        )
                        agg_ctx = (
                            {}
                            if skip_context
                            else aggregate_layer_features(
                                per_image_ctx,
                                mean_pooling_only=mean_pooling_only,
                                last_layer_only=last_layer_only,
                            )
                        )

                        # Filter if apply_mean is False
                        if not apply_mean:
                            agg_x.pop("middle_avg", None)
                            agg_ctx.pop("middle_avg", None)

                        image_name = os.path.basename(image_path).split(".")[0]
                        save_features(
                            agg_x, output_dir, "hidden_x", category, image_name
                        )
                        if not skip_context:
                            save_features(
                                agg_ctx,
                                output_dir,
                                "hidden_context",
                                category,
                                image_name,
                            )

                        agg_attn = None
                        if extract_attention:
                            agg_attn = aggregate_attention_features(
                                per_image_attn,
                                layers_to_save,
                                mean_pooling_only=mean_pooling_only,
                            )
                            if not apply_mean:
                                agg_attn = {
                                    k: v
                                    for k, v in agg_attn.items()
                                    if "middle_avg" not in k
                                }
                            save_features(
                                agg_attn, output_dir, "attention", category, image_name
                            )

                        # Capture shapes from first image
                        if first_image_shapes is None:
                            first_image_shapes = {
                                "hidden_x": {
                                    k: list(v.shape) for k, v in agg_x.items()
                                },
                                "hidden_context": {
                                    k: list(v.shape) for k, v in agg_ctx.items()
                                },
                                "attention": {},
                            }
                            if agg_attn:
                                first_image_shapes["attention"] = {
                                    k: list(v.shape) for k, v in agg_attn.items()
                                }
                            stats["shapes"] = first_image_shapes

                except Exception as e:
                    # Log batch error and continue
                    for path in image_paths:
                        skipped_images.append((path, f"Batch error: {str(e)}"))
                    continue
        else:
            # Original single-image processing
            for image_path in tqdm(images, desc=f"Extracting {category}"):
                try:
                    attn_weights = extract_features_from_image(
                        image_path,
                        model,
                        vae,
                        transform,
                        sigma,
                        device,
                        dtype,
                        features_x,
                        features_context,
                        timestep,
                        extract_attention,
                        conditioning,
                    )

                    # Aggregate
                    agg_x = aggregate_layer_features(
                        features_x,
                        mean_pooling_only=mean_pooling_only,
                        last_layer_only=last_layer_only,
                    )
                    agg_ctx = (
                        {}
                        if skip_context
                        else aggregate_layer_features(
                            features_context,
                            mean_pooling_only=mean_pooling_only,
                            last_layer_only=last_layer_only,
                        )
                    )

                    # Filter if apply_mean is False
                    if not apply_mean:
                        agg_x.pop("middle_avg", None)
                        agg_ctx.pop("middle_avg", None)

                    image_name = os.path.basename(image_path).split(".")[0]
                    save_features(agg_x, output_dir, "hidden_x", category, image_name)
                    if not skip_context:
                        save_features(
                            agg_ctx, output_dir, "hidden_context", category, image_name
                        )

                    agg_attn = None
                    if extract_attention:
                        agg_attn = aggregate_attention_features(
                            attn_weights,
                            layers_to_save,
                            mean_pooling_only=mean_pooling_only,
                        )
                        if not apply_mean:
                            # Remove middle_avg keys
                            agg_attn = {
                                k: v
                                for k, v in agg_attn.items()
                                if "middle_avg" not in k
                            }
                        save_features(
                            agg_attn, output_dir, "attention", category, image_name
                        )

                    # Capture shapes from first image
                    if first_image_shapes is None:
                        first_image_shapes = {
                            "hidden_x": {k: list(v.shape) for k, v in agg_x.items()},
                            "hidden_context": {
                                k: list(v.shape) for k, v in agg_ctx.items()
                            },
                            "attention": {},
                        }
                        if agg_attn:
                            first_image_shapes["attention"] = {
                                k: list(v.shape) for k, v in agg_attn.items()
                            }
                        stats["shapes"] = first_image_shapes

                except (OSError, IOError) as e:
                    # Handle corrupted/truncated images gracefully
                    skipped_images.append((image_path, str(e)))
                    continue
                except Exception as e:
                    # Log unexpected errors but continue processing
                    skipped_images.append((image_path, f"Unexpected error: {str(e)}"))
                    continue

        # Report skipped images
        if skipped_images:
            stats["skipped_count"] = len(skipped_images)
            stats["skipped_images"] = skipped_images
            print(
                f"\nWarning: Skipped {len(skipped_images)} corrupted/problematic images:"
            )
            for path, error in skipped_images[:10]:  # Show first 10
                print(f"  - {os.path.basename(path)}: {error}")
            if len(skipped_images) > 10:
                print(f"  ... and {len(skipped_images) - 10} more")

    finally:
        # Guarantee hooks are removed so model is clean
        for h in hooks:
            h.remove()

    return stats
