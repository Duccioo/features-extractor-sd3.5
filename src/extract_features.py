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
import gc
import traceback
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load


# Do NOT allow loading truncated images - fail fast on corrupted files
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


# Default layers to save: first (0) and last layer (-1 = auto-detect from model)
# The number of joint blocks varies by model (e.g., large=38, medium=24)
# Use -1 to automatically save the last available layer

# Image branch features (hidden_x)
SELECTED_LAYERS_X = [0, -1]

# Text branch features (hidden_context)
# Note: The last joint block has pre_only=True, so context has one fewer layer
SELECTED_LAYERS_CONTEXT = [0, -1]

# Attention weights - uses same layer indexing as hidden_x
# These only take effect if --extract_attention is enabled
SELECTED_LAYERS_ATTENTION = [0, -1]



def setup_output_directories_for_category(
    output_path, category, extract_attention=False, skip_context=False
):
    """Create output directory structure for features specific to a category.
    
    New format: single directory per category, one .safetensors file per image.
    """
    os.makedirs(os.path.join(output_path, category), exist_ok=True)


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
            # Return a dummy tensor for failed images (empty, will be filtered)
            dummy = torch.empty(0)
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
    enabled_flag=None,
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
            # Disable hooks for attention pass to prevent overwriting features
            if enabled_flag is not None:
                enabled_flag[0] = False
                
            try:
                with AttentionCapture(model) as attn_capture:
                    _ = model.apply_model(
                        noised_latents,
                        sigma.expand(batch_size),
                        c_crossattn=context,
                        y=y,
                    )
                    attention_weights.update(attn_capture.attention_weights)
            finally:
                # Re-enable hooks
                if enabled_flag is not None:
                    enabled_flag[0] = True

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
    enabled_flag=None,
):
    """Extract features from a single image.
    
    Returns:
        Tuple of (attention_weights, features_x, features_context)
    """
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
            # Disable hooks for attention pass
            if enabled_flag is not None:
                enabled_flag[0] = False
                
            try:
                with AttentionCapture(model) as attn_capture:
                    _ = model.apply_model(
                        noised_latents,
                        sigma.expand(batch_size),
                        c_crossattn=context,
                        y=y,
                    )
                    attention_weights.update(attn_capture.attention_weights)
            finally:
                # Re-enable hooks
                if enabled_flag is not None:
                    enabled_flag[0] = True

    return attention_weights, features_x, features_context


def resolve_layer_indices(selected_layers, num_blocks, is_context=False):
    """Resolve -1 sentinel values to actual last layer index.
    
    Args:
        selected_layers: List of layer indices, where -1 means 'last available layer'.
        num_blocks: Total number of joint blocks in the model.
        is_context: If True, context branch has one fewer layer (last block is pre_only).
    
    Returns:
        List of resolved layer indices.
    """
    if selected_layers is None:
        return None
    
    last_idx = (num_blocks - 2) if is_context else (num_blocks - 1)
    resolved = []
    for idx in selected_layers:
        if idx == -1:
            resolved.append(last_idx)
        else:
            resolved.append(idx)
    return resolved


def aggregate_layer_features(
    features_dict, mean_pooling_only=False, selected_layers=None
):
    """Aggregate layer features: keep average of all layers and selected layers.

    Args:
        features_dict: Dictionary of layer_name -> tensor.
        mean_pooling_only: If True, apply spatial mean pooling to reduce tensor size.
                          Reduces [1, seq_len, dim] to [1, dim].
        selected_layers: Optional list of layer indices to save (e.g., [0, 23] for first/last).
                        Use -1 for last available layer (resolved before calling this).
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

    # Always save mean of all layers
    all_tensors = [features_dict[k] for k in sorted_keys]
    stacked = torch.stack(all_tensors, dim=0)
    aggregated["middle_avg"] = apply_pooling(stacked.mean(dim=0))

    # Add selected layers if specified
    if selected_layers is not None:
        for layer_idx in selected_layers:
            key = f"block_{layer_idx}"
            if key in features_dict:
                aggregated[f"layer_{layer_idx}"] = apply_pooling(features_dict[key])
            else:
                print(f"  Warning: layer block_{layer_idx} not found in features (available: {list(features_dict.keys())[:5]}...)")

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


def save_all_features(agg_x, agg_ctx, agg_attn, output_path, category, image_name):
    """Save all extracted features for one image into a single .safetensors file.
    
    Keys are namespaced: 'hidden_x__middle_avg', 'hidden_context__layer_0', 
    'attention__layer_0_joint', etc.
    """
    all_tensors = {}
    
    # Hidden X features
    for name, tensor in agg_x.items():
        all_tensors[f"hidden_x__{name}"] = tensor.contiguous()
    
    # Hidden Context features
    if agg_ctx:
        for name, tensor in agg_ctx.items():
            all_tensors[f"hidden_context__{name}"] = tensor.contiguous()
    
    # Attention features
    if agg_attn:
        for name, tensor in agg_attn.items():
            all_tensors[f"attention__{name}"] = tensor.contiguous()
    
    if not all_tensors:
        return
    
    save_path = os.path.join(output_path, category, f"{image_name}.safetensors")
    safetensors_save(all_tensors, save_path)


def load_features(filepath):
    """Load features from a .safetensors file.
    
    Returns a dict with three sub-dicts: 'hidden_x', 'hidden_context', 'attention'.
    Each sub-dict maps feature names (e.g. 'middle_avg', 'layer_0') to tensors.
    
    Example:
        features = load_features("output/real/image001.safetensors")
        hidden_x_avg = features['hidden_x']['middle_avg']  # [1, 1024, 1536]
        ctx_layer_0 = features['hidden_context']['layer_0']  # [1, 154, 1536]
    """
    raw = safetensors_load(filepath)
    
    result = {'hidden_x': {}, 'hidden_context': {}, 'attention': {}}
    for key, tensor in raw.items():
        # Keys are like 'hidden_x__middle_avg', 'attention__layer_0_joint'
        parts = key.split('__', 1)
        if len(parts) == 2:
            group, name = parts
            if group in result:
                result[group][name] = tensor
            else:
                result[group] = {name: tensor}
        else:
            # Fallback for unexpected keys
            result[key] = tensor
    
    return result


def extract_features(
    images_dir,
    output_dir,
    category="data",
    model=None,
    vae=None,
    model_path=None,
    timestep=0,
    extract_attention=False,
    num_images=-1,
    image_size=512,
    text_embedding_path=None,
    text_embedding_prompt="",
    apply_mean=True,
    preprocessing_mode="imagenet_style",
    jpeg_aug=False,
    mean_pooling_only=False,
    skip_context=False,
    selected_layers_x=None,
    selected_layers_context=None,
    selected_layers_attention=None,
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
        skip_context: If True, skip saving hidden_context features (~50% additional savings).
        selected_layers_x: Optional list of layer indices for hidden_x (image branch, 0-37).
        selected_layers_context: Optional list of layer indices for hidden_context (text branch, 0-36).
        selected_layers_attention: Optional list of layer indices for attention weights (0-37).
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
    num_blocks = len(model.diffusion_model.joint_blocks)
    print(f"Model has {num_blocks} joint blocks (layers 0-{num_blocks-1})")
    for i, block in enumerate(model.diffusion_model.joint_blocks):
        # We need the enabled_flag from register_feature_hooks. 
        # But here we are calling create_dual_feature_hook manually?
        # Ah, the user code earlier used register_feature_hooks in utils/features.py but here it seems to implement it inline?
        # Wait, line 33 imports create_dual_feature_hook.
        # Let's use register_feature_hooks from utils.features instead of this manual loop if possible, 
        # OR just update this manual loop to match the new API.
        # The file content at line 511 iterates over blocks. 
        # I should just update this part to use the new create_dual_feature_hook signature.
        pass
    
    # Actually, I should use the proper register_feature_hooks from utils if I can, but the import at line 33 is:
    # from utils.features import create_dual_feature_hook, AttentionCapture
    # It does NOT import register_feature_hooks. 
    # To minimize changes, I will just update the manual registration here.
    
    enabled_flag = [True]
    for i, block in enumerate(model.diffusion_model.joint_blocks):
        hook_fn = create_dual_feature_hook(features_x, features_context, f"block_{i}", enabled_flag)
        h = block.register_forward_hook(hook_fn)
        hooks.append(h)

    # Resolve -1 sentinel to actual last layer index
    if selected_layers_x is not None:
        selected_layers_x = resolve_layer_indices(selected_layers_x, num_blocks, is_context=False)
        print(f"Selected layers (hidden_x): {selected_layers_x}")
    if selected_layers_context is not None:
        selected_layers_context = resolve_layer_indices(selected_layers_context, num_blocks, is_context=True)
        print(f"Selected layers (hidden_context): {selected_layers_context}")
    if selected_layers_attention is not None:
        selected_layers_attention = resolve_layer_indices(selected_layers_attention, num_blocks, is_context=False)
        print(f"Selected layers (attention): {selected_layers_attention}")

    stats = {"count": 0, "shapes": None}

    try:
        # 4. Preparation
        setup_output_directories_for_category(
            output_dir, category, extract_attention, skip_context
        )

        model_sampling = ModelSamplingDiscreteFlow(shift=3.0)

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
                        enabled_flag,
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
                            selected_layers=selected_layers_x,
                        )
                        agg_ctx = (
                            {}
                            if skip_context
                            else aggregate_layer_features(
                                per_image_ctx,
                                mean_pooling_only=mean_pooling_only,
                                selected_layers=selected_layers_context,
                            )
                        )

                        # Filter if apply_mean is False
                        if not apply_mean:
                            agg_x.pop("middle_avg", None)
                            agg_ctx.pop("middle_avg", None)

                        image_name = os.path.splitext(os.path.basename(image_path))[0]

                        agg_attn = None
                        if extract_attention:
                            attn_layers = selected_layers_attention
                            agg_attn = aggregate_attention_features(
                                per_image_attn,
                                attn_layers,
                                mean_pooling_only=mean_pooling_only,
                            )
                            if not apply_mean:
                                agg_attn = {
                                    k: v
                                    for k, v in agg_attn.items()
                                    if "middle_avg" not in k
                                }

                        # Save all features in a single .safetensors file
                        save_all_features(
                            agg_x, agg_ctx, agg_attn,
                            output_dir, category, image_name
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

                    # Aggressive memory cleanup after each batch
                    features_x.clear()
                    features_context.clear()
                    gc.collect()
                    torch.cuda.empty_cache()
                
                except Exception as e:
                    # Log batch error and continue
                    print(f"\n[DEBUG] Batch error: {str(e)}")
                    traceback.print_exc()
                    # Extra cleanup on error
                    try:
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    for path in image_paths:
                        skipped_images.append((path, f"Batch error: {str(e)}"))
                    continue
        else:
            # Original single-image processing
            for image_path in tqdm(images, desc=f"Extracting {category}"):
                try:
                    attn_weights, _, _ = extract_features_from_image(
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
                        enabled_flag,
                    )

                    # Aggregate
                    agg_x = aggregate_layer_features(
                        features_x,
                        mean_pooling_only=mean_pooling_only,
                        selected_layers=selected_layers_x,
                    )
                    agg_ctx = (
                        {}
                        if skip_context
                        else aggregate_layer_features(
                            features_context,
                            mean_pooling_only=mean_pooling_only,
                            selected_layers=selected_layers_context,
                        )
                    )

                    # Filter if apply_mean is False
                    if not apply_mean:
                        agg_x.pop("middle_avg", None)
                        agg_ctx.pop("middle_avg", None)

                    image_name = os.path.splitext(os.path.basename(image_path))[0]

                    agg_attn = None
                    if extract_attention:
                        attn_layers = selected_layers_attention
                        agg_attn = aggregate_attention_features(
                            attn_weights,
                            attn_layers,
                            mean_pooling_only=mean_pooling_only,
                        )
                        if not apply_mean:
                            agg_attn = {
                                k: v
                                for k, v in agg_attn.items()
                                if "middle_avg" not in k
                            }

                    # Save all features in a single .safetensors file
                    save_all_features(
                        agg_x, agg_ctx, agg_attn,
                        output_dir, category, image_name
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
