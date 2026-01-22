"""
Script to extract and save text embeddings from SD3.5 for later use.

This script uses SD3Inferencer to load text encoders and compute embeddings,
exactly like the official implementation. The embeddings are saved to a .pt file
that can be easily loaded and used as conditioning input for SD3.5 inference.

Usage:
    python extract_text_embedding.py --model_path ../models/sd3.5_large.safetensors --output text_embeddings.pt

    # Custom prompts
    python extract_text_embedding.py --model_path ../models/sd3.5_large.safetensors --prompts "" "a photo" "a cat"
"""

import argparse
import datetime
import os
import sys
import torch


# ---
# Add the sd3.5 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sd3.5"))

from sd3_infer import SD3Inferencer


#################################################################################################
### Text Embedding Extraction
#################################################################################################


class SD3TextEmbeddingExtractor:
    """Extract text embeddings compatible with SD3.5 conditioning."""

    def __init__(
        self,
        model_path: str,
        model_folder: str = None,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """
        Initialize the extractor using SD3Inferencer.

        Args:
            model_path: Path to the SD3.5 model file (.safetensors)
            model_folder: Path to folder containing clip_l, clip_g, t5xxl (default: same as model)
            device: Device for text encoders
            verbose: Print loading progress
        """
        self.device = device
        self.verbose = verbose

        # Use SD3Inferencer to load everything
        self.inferencer = SD3Inferencer()

        # Determine model folder (defaults to parent dir of model file)
        if model_folder is None:
            model_folder = os.path.dirname(model_path)

        if verbose:
            print(f"Loading SD3.5 from: {model_path}")
            print(f"Model folder: {model_folder}")

        # Load only text encoders (we don't need the full model for embedding extraction)
        self.inferencer.load(
            model=model_path,
            shift=3.0,
            model_folder=model_folder,
            text_encoder_device=device,
            verbose=verbose,
            load_tokenizers=True,
        )

        if verbose:
            print("Text encoders loaded successfully.")

    def get_embedding(self, prompt: str) -> dict:
        """
        Compute text embedding for a single prompt using the official get_cond method.

        Returns a dict with:
            - 'c_crossattn': torch.Tensor of shape [1, 154, 4096]
            - 'y': torch.Tensor of shape [1, 2048] (pooled embedding)
        """
        with torch.no_grad():
            c_crossattn, y = self.inferencer.get_cond(prompt)

        return {
            "c_crossattn": c_crossattn,  # [1, 154, 4096]
            "y": y,  # [1, 2048]
        }

    def extract_embeddings(self, prompts: list) -> dict:
        """
        Extract embeddings for multiple prompts.

        Returns a dict with structure:
        {
            "prompts": ["", "a photo", ...],
            "embeddings": {
                "": {"c_crossattn": tensor, "y": tensor},
                "a photo": {"c_crossattn": tensor, "y": tensor},
                ...
            },
            "metadata": {...}
        }
        """
        embeddings = {}

        for prompt in prompts:
            if self.verbose:
                print(f"  Encoding: {repr(prompt)}")

            emb = self.get_embedding(prompt)
            embeddings[prompt] = emb

            if self.verbose:
                print(
                    f"    c_crossattn: {emb['c_crossattn'].shape}, y: {emb['y'].shape}"
                )

        return {
            "prompts": prompts,
            "embeddings": embeddings,
            "metadata": {
                "dtype": str(embeddings[prompts[0]]["c_crossattn"].dtype),
                "created_at": datetime.datetime.now().isoformat(),
                "description": "SD3.5 text embeddings for conditioning",
            },
        }


def load_text_embeddings(
    filepath: str, prompt: str = None, device: str = "cuda", dtype=torch.float16
):
    """
    Utility function to load precomputed text embeddings.

    Args:
        filepath: Path to the .pt file with embeddings
        prompt: Specific prompt to retrieve (if None, returns all embeddings)
        device: Device to move tensors to
        dtype: Data type for tensors

    Returns:
        If prompt is specified: dict with 'c_crossattn' and 'y' tensors ready for SD3.5
        If prompt is None: the full embeddings dict

    Example:
        # Load and use with SD3.5
        cond = load_text_embeddings("text_embeddings.pt", "a photo")
        model.apply_model(x, sigma, c_crossattn=cond['c_crossattn'], y=cond['y'])
    """
    data = torch.load(filepath, weights_only=False)

    if prompt is None:
        return data

    if prompt not in data["embeddings"]:
        available = list(data["embeddings"].keys())
        raise KeyError(f"Prompt {repr(prompt)} not found. Available: {available}")

    emb = data["embeddings"][prompt]
    return {
        "c_crossattn": emb["c_crossattn"].to(device=device, dtype=dtype),
        "y": emb["y"].to(device=device, dtype=dtype),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract text embeddings from SD3.5 for later use."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the SD3.5 model file (.safetensors)",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        default=None,
        help="Path to folder containing clip_l, clip_g, t5xxl (default: same as model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="text_embeddings.pt",
        help="Output file path for the embeddings (default: text_embeddings.pt)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["", "a photo"],
        help='Prompts to encode (default: "" and "a photo")',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for text encoders (default: cpu)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    # Extract embeddings
    print("=" * 60)
    print("SD3.5 Text Embedding Extraction")
    print("=" * 60)

    extractor = SD3TextEmbeddingExtractor(
        model_path=args.model_path,
        model_folder=args.model_folder,
        device=args.device,
        verbose=not args.quiet,
    )

    print(f"\nExtracting embeddings for {len(args.prompts)} prompts...")
    result = extractor.extract_embeddings(args.prompts)

    # Save to file
    torch.save(result, args.output)
    print(f"\nEmbeddings saved to: {args.output}")

    # Print summary
    print("\nSummary:")
    print("-" * 40)
    for prompt in result["prompts"]:
        emb = result["embeddings"][prompt]
        print(
            f"  {repr(prompt):20s} -> c_crossattn: {list(emb['c_crossattn'].shape)}, y: {list(emb['y'].shape)}"
        )

    print("\n" + "=" * 60)
    print("Usage example:")
    print("-" * 40)
    print(
        """
from extract_text_embedding import load_text_embeddings

# Load embeddings for a specific prompt
cond = load_text_embeddings("text_embeddings.pt", "a photo", device="cuda")

# Use with SD3.5 model
model.apply_model(x, sigma, c_crossattn=cond['c_crossattn'], y=cond['y'])
"""
    )


if __name__ == "__main__":
    main()
