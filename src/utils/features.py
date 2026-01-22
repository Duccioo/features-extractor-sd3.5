"""
Feature Capture Utilities
"""

import os
import sys
import torch

# Add the sd3.5 directory to the path (relative to src/utils/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sd3.5"))


def create_feature_hook(features_dict, name, output_index=1):
    """Create a hook function to capture features from a layer.

    Args:
        features_dict: Dictionary to store captured features
        name: Name/key for this feature
        output_index: Index in the output tuple to capture (default: 1 for image features)

    Returns:
        Hook function
    """

    def hook(model, input, output):
        # For JointBlock: output is (context, x)
        # output[1] = x = image features
        # output[0] = context = text features
        features_dict[name] = output[output_index].detach().cpu()

    return hook


def create_dual_feature_hook(features_x, features_context, name):
    """Create a hook that captures both image (x) and context features.

    Args:
        features_x: Dictionary for image features
        features_context: Dictionary for context features
        name: Base name for the features

    Returns:
        Hook function
    """

    def hook(model, input, output):
        # The output of a JointBlock is a tuple (context, x)
        # output[1] = x = image features
        features_x[name] = output[1].detach().cpu()
        # Also save context if available
        if output[0] is not None:
            features_context[name] = output[0].detach().cpu()

    return hook


def register_feature_hooks(model, features_x, features_context):
    """Register hooks on all JointBlocks to capture features.

    Args:
        model: The SD3.5 model
        features_x: Dictionary to store image features
        features_context: Dictionary to store context features

    Returns:
        List of hook handles
    """
    handles = []
    print("Registering hooks on JointBlocks...")
    for i, block in enumerate(model.diffusion_model.joint_blocks):
        hook_fn = create_dual_feature_hook(features_x, features_context, f"block_{i}")
        handle = block.register_forward_hook(hook_fn)
        handles.append(handle)
    return handles


class AttentionCapture:
    """Context manager to capture attention weights during forward pass.

    Follows the official SD3.5 mmditx.py implementation:
    - Each JointBlock contains a context_block and x_block (both DismantledBlock)
    - block_mixing() performs joint attention with concatenated Q,K,V from both blocks
    - If x_block.x_block_self_attn is True, a second attention call occurs (self-attn on x only)

    Captured attention types:
    - 'block_{i}_joint': Joint attention over (context + x) tokens
      Shape: (B, context_len + x_len, context_len + x_len), e.g. (1, 4250, 4250)
    - 'block_{i}_self': X-block self-attention (only if x_block_self_attn=True)
      Shape: (B, x_len, x_len), e.g. (1, 4096, 4096)
    """

    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.original_forwards = {}
        self.blocks_with_self_attn = set()

    def _detect_self_attn_blocks(self):
        """Detect which blocks have x_block_self_attn enabled."""
        if not hasattr(self.model, "diffusion_model"):
            return
        dm = self.model.diffusion_model
        for i, block in enumerate(dm.joint_blocks):
            # Check if x_block has self-attention enabled
            if hasattr(block, "x_block") and hasattr(
                block.x_block, "x_block_self_attn"
            ):
                if block.x_block.x_block_self_attn:
                    self.blocks_with_self_attn.add(i)

    def _wrap_block_forward(self, block, block_idx):
        """Wrap a JointBlock's forward to capture attention weights.

        Based on mmditx.py block_mixing():
        1. First attention() call: joint attention (context + x concatenated)
        2. Second attention() call (if x_block_self_attn): self-attention on x only
        """
        original_forward = block.forward
        capture_dict = self.attention_weights
        has_self_attn = block_idx in self.blocks_with_self_attn

        def wrapped_forward(context, x, c):
            import other_impls
            import mmditx

            original_attention = other_impls.attention
            captured_attentions = []

            def capturing_attention(q, k, v, heads, mask=None):
                b, seq_len, total_dim = q.shape
                dim_head = total_dim // heads

                q_reshaped = q.view(b, seq_len, heads, dim_head).transpose(1, 2)
                k_reshaped = k.view(b, -1, heads, dim_head).transpose(1, 2)
                v_reshaped = v.view(b, -1, heads, dim_head).transpose(1, 2)

                # Compute attention weights
                scale = dim_head**-0.5
                attn_weights = (
                    torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * scale
                )
                if mask is not None:
                    attn_weights = attn_weights + mask

                attn_weights_softmax = torch.softmax(attn_weights, dim=-1)

                if torch.isnan(attn_weights_softmax).any():
                    print(
                        "[AttentionCapture] Warning: NaNs detected in attention weights"
                    )

                # Store mean attention over heads (head averaging saves memory)
                captured_attentions.append(
                    {
                        "weights": attn_weights_softmax.mean(dim=1).detach().cpu(),
                        "seq_len": seq_len,
                    }
                )

                # Compute output normally (matching official implementation)
                out = torch.matmul(attn_weights_softmax, v_reshaped)
                return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

            # Temporarily patch attention for this block
            other_impls.attention = capturing_attention
            mmditx.attention = capturing_attention

            try:
                result = original_forward(context, x, c=c)
            finally:
                # Restore original attention function immediately
                other_impls.attention = original_attention
                mmditx.attention = original_attention

            # Store captured attentions with proper naming based on call order
            # Per mmditx.py block_mixing():
            # - First call: joint attention (context + x)
            # - Second call (if present): self-attention on x
            if len(captured_attentions) >= 1:
                capture_dict[f"block_{block_idx}_joint"] = captured_attentions[0][
                    "weights"
                ]
            # FIX: Always save self-attention if there are 2+ attention calls,
            # regardless of has_self_attn flag (which may not be detected correctly)
            if len(captured_attentions) >= 2:
                capture_dict[f"block_{block_idx}_self"] = captured_attentions[1][
                    "weights"
                ]

            return result

        return wrapped_forward

    def __enter__(self):
        # First detect which blocks have self-attention
        self._detect_self_attn_blocks()

        if hasattr(self.model, "diffusion_model"):
            dm = self.model.diffusion_model
            for i, block in enumerate(dm.joint_blocks):
                # Store original forward
                self.original_forwards[i] = block.forward
                # Replace with wrapped version
                block.forward = self._wrap_block_forward(block, i)

        return self

    def __exit__(self, *args):
        # Restore all original forwards
        if hasattr(self.model, "diffusion_model"):
            dm = self.model.diffusion_model
            for i, block in enumerate(dm.joint_blocks):
                if i in self.original_forwards:
                    block.forward = self.original_forwards[i]

        self.original_forwards.clear()
        self.blocks_with_self_attn.clear()

    def get_self_attn_layer_indices(self):
        """Return list of layer indices that have self-attention enabled."""
        return sorted(self.blocks_with_self_attn)
