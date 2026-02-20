"""
Script di esecuzione per l'estrazione delle features (CLI).
Chiama la libreria src/extract_features.py per processare immagini reali e fake.
"""

import argparse
import os
import sys
import datetime
import json
import torch

# ---
# Aggiungi la directory corrente e sd3.5 al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sd3.5"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.system import DualLogger, get_system_info
from utils.model import load_sd35_model, validate_model_loading
from extract_features import (
    extract_features,
    SELECTED_LAYERS_X,
    SELECTED_LAYERS_CONTEXT,
    SELECTED_LAYERS_ATTENTION,
)


def setup_logging(output_path):
    """Setup dual logging to terminal and file."""
    log_file = os.path.join(output_path, "execution.log")
    sys.stdout = DualLogger(log_file)
    sys.stderr = sys.stdout


def create_experiment_data(args):
    """Create initial experiment metadata."""
    return {
        "start_time": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "system_info": get_system_info(),
        "feature_shapes": {"hidden_x": {}, "hidden_context": {}, "attention": {}},
    }


def finalize_experiment(experiment_data, output_path):
    """Finalize and save experiment metadata."""
    experiment_data["end_time"] = datetime.datetime.now().isoformat()
    start = datetime.datetime.fromisoformat(experiment_data["start_time"])
    end = datetime.datetime.fromisoformat(experiment_data["end_time"])
    experiment_data["duration_seconds"] = (end - start).total_seconds()

    json_path = os.path.join(output_path, "experiment_config.json")
    try:
        with open(json_path, "w") as f:
            json.dump(experiment_data, f, indent=4)
        print(f"Experiment metadata saved to {json_path}")
    except Exception as e:
        print(f"Error saving experiment metadata: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run feature extraction for Real/Fake images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/meconcelli/homeRepo/test-diffusion-model/models/sd3.5_large.safetensors",
    )
    parser.add_argument("--real_images_path", type=str, default=None, help="Path to real images (optional)")
    parser.add_argument("--fake_images_path", type=str, default=None, help="Path to fake images (optional)")
    parser.add_argument("--output_path", type=str, default="features")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--text_embedding_path", type=str, default=None)
    parser.add_argument("--text_embedding_prompt", type=str, default="")
    parser.add_argument("--extract_attention", action="store_true")
    parser.add_argument("--num_images", type=int, default=-1)
    parser.add_argument("--simulate_low_res", action="store_true")
    parser.add_argument(
        "--apply_mean",
        action="store_true",
        default=True,
        help="Apply mean aggregation to features",
    )
    parser.add_argument(
        "--preprocessing_mode",
        type=str,
        default="imagenet_style",
        help="Preprocessing mode (imagenet_style, brutal_resize, crop_100_then_resize)",
    )
    # JPEG augmentation controls (using int 0/1 for explicit control)
    parser.add_argument(
        "--jpeg_aug_real",
        type=int,
        default=0,
        help="Enable JPEG augmentation for real images (0 or 1)",
    )
    parser.add_argument(
        "--jpeg_aug_fake",
        type=int,
        default=0,
        help="Enable JPEG augmentation for fake images (0 or 1)",
    )
    parser.add_argument(
        "--mean_pooling_only",
        action="store_true",
        default=False,
        help="Apply spatial mean pooling to reduce feature size from [1, seq_len, dim] to [1, dim]",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate: at least one path must be provided
    if not args.real_images_path and not args.fake_images_path:
        print("Error: At least one of --real_images_path or --fake_images_path must be provided.")
        return

    # Seeding
    seed = 69
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    os.makedirs(args.output_path, exist_ok=True)
    setup_logging(args.output_path)
    experiment_data = create_experiment_data(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    print(f"Loading model from {args.model_path}...")
    model, vae = load_sd35_model(args.model_path, device, dtype, verbose=True)
    validate_model_loading(model.diffusion_model, "MM-DiT")
    validate_model_loading(vae, "VAE")

    real_stats = {}
    fake_stats = {}

    # Extract Real (only if path provided)
    if args.real_images_path:
        print(f"\nStarting extraction for REAL images (JPEG={bool(args.jpeg_aug_real)})...")
        real_stats = extract_features(
            images_dir=args.real_images_path,
            output_dir=args.output_path,
            category="real",
            model=model,
            vae=vae,
            timestep=args.timestep,
            selected_layers_x=SELECTED_LAYERS_X,
            selected_layers_context=SELECTED_LAYERS_CONTEXT,
            selected_layers_attention=SELECTED_LAYERS_ATTENTION,
            extract_attention=args.extract_attention,
            num_images=args.num_images,
            image_size=args.image_size,
            simulate_low_res=args.simulate_low_res,
            text_embedding_path=args.text_embedding_path,
            text_embedding_prompt=args.text_embedding_prompt,
            apply_mean=args.apply_mean,
            preprocessing_mode=args.preprocessing_mode,
            jpeg_aug=bool(args.jpeg_aug_real),
            mean_pooling_only=args.mean_pooling_only,
            device=device,
            dtype=dtype,
        )
    else:
        print("\nSkipping REAL images (no path provided)")

    # Extract Fake (only if path provided)
    if args.fake_images_path:
        print(f"\nStarting extraction for FAKE images (JPEG={bool(args.jpeg_aug_fake)})...")
        fake_stats = extract_features(
            images_dir=args.fake_images_path,
            output_dir=args.output_path,
            category="fake",
            model=model,
            vae=vae,
            timestep=args.timestep,
            selected_layers_x=SELECTED_LAYERS_X,
            selected_layers_context=SELECTED_LAYERS_CONTEXT,
            selected_layers_attention=SELECTED_LAYERS_ATTENTION,
            extract_attention=args.extract_attention,
            num_images=args.num_images,
            image_size=args.image_size,
            simulate_low_res=args.simulate_low_res,
            text_embedding_path=args.text_embedding_path,
            text_embedding_prompt=args.text_embedding_prompt,
            apply_mean=args.apply_mean,
            preprocessing_mode=args.preprocessing_mode,
            jpeg_aug=bool(args.jpeg_aug_fake),
            mean_pooling_only=args.mean_pooling_only,
            device=device,
            dtype=dtype,
        )
    else:
        print("\nSkipping FAKE images (no path provided)")

    # Update experiment data with shapes (taken from real stats if available, else fake)
    shapes = real_stats.get("shapes") or fake_stats.get("shapes")
    if shapes:
        experiment_data["feature_shapes"] = shapes

    finalize_experiment(experiment_data, args.output_path)


if __name__ == "__main__":
    main()
