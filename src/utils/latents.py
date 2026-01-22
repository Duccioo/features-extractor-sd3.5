"""
Latent Space Utilities
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add the sd3.5 directory to the path (relative to src/utils/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sd3.5"))

try:
    from sd3_impls import SD3LatentFormat
except ImportError:
    pass


def get_noise(shape, generator, device, dtype):
    """Generates a tensor of noise."""
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def add_noise(latents, noise, sigmas):
    """Adds noise to the latents according to the Rectified Flow formula."""
    return sigmas * noise + (1.0 - sigmas) * latents


def scale_latents(latents):
    """Scales the latents to the expected format."""
    return SD3LatentFormat().process_in(latents)


def latent_to_image(vae, latents):
    """Decodes latents into an image."""
    latents = SD3LatentFormat().process_out(latents)
    image = vae.decode(latents)
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    image = image.cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((image * 255).astype(np.uint8))


def encode_image_to_latent(vae, image_tensor):
    """Encode an image tensor to latent space.

    Args:
        vae: The VAE encoder
        image_tensor: Tensor of shape [B, C, H, W] in range [-1, 1]

    Returns:
        Latents scaled for SD3.5
    """
    with torch.no_grad():
        latents = vae.encode(image_tensor)
        latents = SD3LatentFormat().process_in(latents)
        # Keep dtype/device consistent with caller (e.g., bfloat16 on GPU)
        latents = latents.to(device=image_tensor.device, dtype=image_tensor.dtype)
    return latents
