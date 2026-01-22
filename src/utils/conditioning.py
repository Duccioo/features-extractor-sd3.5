"""
Conditioning Utilities
"""

import torch


def get_context_dimensions(model):
    """Extract context dimensions from a loaded SD3.5 model.

    The context embedder's Linear layer output dim defines context_dim.
    The default values are based on SD3.5 architecture.

    Args:
        model: The loaded SD3.5 BaseModel

    Returns:
        Tuple of (context_dim, context_len, pooled_dim)
    """
    context_dim = 4096  # SD3.5 default: T5-XXL (4096) projected
    context_len = 154  # SD3.5 default: 77 (CLIP) + 77 (T5) = 154 (may vary)
    pooled_dim = 2048  # SD3.5 default: CLIP-L (768) + CLIP-G (1280) = 2048

    if hasattr(model, "diffusion_model"):
        dm = model.diffusion_model
        # Try to get context_dim from context_embedder
        if hasattr(dm, "context_embedder"):
            ce = dm.context_embedder
            # For Linear, the expected input dim is in_features
            if hasattr(ce, "in_features"):
                context_dim = ce.in_features
            elif hasattr(ce, "out_features"):
                context_dim = ce.out_features

        if hasattr(dm, "y_embedder") and hasattr(dm.y_embedder, "in_features"):
            pooled_dim = dm.y_embedder.in_features

    return context_dim, context_len, pooled_dim


def create_empty_conditioning(
    batch_size,
    device,
    dtype,
    context_dim=None,
    context_len=None,
    pooled_dim=None,
    model=None,
):
    """Create empty/null conditioning tensors for unconditional inference.

    Args:
        batch_size: Batch size
        device: Device for tensors
        dtype: Data type for tensors
        context_dim: Dimension of context embeddings (default: 4096 for SD3.5, or auto-detect from model)
        context_len: Sequence length of context (default: 154 for SD3.5, or auto-detect from model)
        pooled_dim: Dimension of pooled embeddings (default: 2048 for SD3.5, or auto-detect from model)
        model: Optional model to auto-detect dimensions from

    Returns:
        Tuple of (context, pooled_embedding)
    """
    # Auto-detect from model if provided and dimensions not specified
    if model is not None:
        auto_context_dim, auto_context_len, auto_pooled_dim = get_context_dimensions(
            model
        )
        context_dim = context_dim or auto_context_dim
        context_len = context_len or auto_context_len
        pooled_dim = pooled_dim or auto_pooled_dim
    else:
        # SD3.5 defaults
        context_dim = context_dim or 4096
        context_len = context_len or 154
        pooled_dim = pooled_dim or 2048

    context = torch.zeros(
        batch_size, context_len, context_dim, device=device, dtype=dtype
    )
    y = torch.zeros(batch_size, pooled_dim, device=device, dtype=dtype)
    return context, y
