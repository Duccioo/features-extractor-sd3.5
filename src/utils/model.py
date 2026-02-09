"""
Model Loading Utilities
"""

import os
import sys
import torch
from safetensors import safe_open

# Add the sd3.5 directory to the path (relative to src/utils/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sd3.5"))

try:
    from sd3_impls import SDVAE, BaseModel
except ImportError:
    # Fallback if running from a context where path is already set differently
    # or if the folder structure is different.
    pass


def load_into(ckpt, model, prefix, device, dtype=None):
    """Load weights from a safetensors file into a pytorch module with diagnostics."""
    missing_attrs = []
    shape_mismatches = []
    loaded = 0

    for key in ckpt.keys():
        model_key = key
        # Handle prefixes
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model

            for p in path:
                if obj is None:
                    break
                # Handle lists, ModuleList, and other indexable containers
                if p.isdigit():
                    try:
                        obj = obj[int(p)]
                    except (IndexError, KeyError, TypeError):
                        obj = getattr(obj, p, None)
                else:
                    obj = getattr(obj, p, None)

            if obj is None:
                missing_attrs.append(model_key)
                continue

            try:
                tensor = ckpt.get_tensor(key)
                target_dtype = (
                    dtype
                    if (dtype and tensor.dtype.is_floating_point)
                    else tensor.dtype
                )
                tensor = tensor.to(device=device, dtype=target_dtype)
                if not hasattr(obj, "shape") or obj.shape != tensor.shape:
                    shape_mismatches.append(model_key)
                    continue
                # Use no_grad to allow in-place operation on nn.Parameter
                with torch.no_grad():
                    obj.copy_(tensor)
                    if hasattr(obj, "requires_grad"):
                        obj.requires_grad = False
                loaded += 1
            except (AttributeError, RuntimeError, TypeError) as e:
                missing_attrs.append(f"{model_key} ({type(e).__name__})")
                continue

    if missing_attrs:
        sample = ", ".join(missing_attrs[:5])
        print(
            f"[load_into] Missing attributes for {len(missing_attrs)} keys (showing up to 5): {sample}"
        )
    if shape_mismatches:
        sample = ", ".join(shape_mismatches[:5])
        print(
            f"[load_into] Shape mismatches for {len(shape_mismatches)} keys (showing up to 5): {sample}"
        )
    print(f"[load_into] Loaded {loaded} tensors with prefix '{prefix}'")


def load_sd35_model(model_path, device, dtype=torch.float16, verbose=True):
    """Load SD3.5 diffusion model and VAE from a safetensors checkpoint.

    Args:
        model_path: Path to the safetensors checkpoint file
        device: Device to load the model to
        dtype: Data type for model weights (default: float16)
        verbose: Whether to print loading progress

    Returns:
        Tuple of (model, vae)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model from {model_path}...")

    # Load diffusion model
    with safe_open(model_path, framework="pt", device="cpu") as f:
        model = BaseModel(
            shift=3.0,
            file=f,
            prefix="model.diffusion_model.",
            device=device,
            dtype=dtype,
            verbose=verbose,
        ).eval()
        load_into(f, model, "model.", device, dtype)

    # Load VAE
    print("Loading VAE...")
    vae = SDVAE(device="cpu", dtype=dtype).eval()
    with safe_open(model_path, framework="pt", device="cpu") as f:
        prefix = ""
        if any(k.startswith("first_stage_model.") for k in f.keys()):
            prefix = "first_stage_model."
        else:
            print(
                "[load_sd35_model] Warning: no 'first_stage_model.' keys found; VAE weights may be missing"
            )
        load_into(f, vae, prefix, "cpu", dtype)
    vae = vae.to(device)

    return model, vae


def validate_model_loading(model, model_name="DiT"):
    """
    Verifica se ci sono parametri nel modello che sono rimasti 'vergini'
    (ovvero non toccati dal caricamento).
    """
    print(f"\n--- Validating {model_name} Weights ---")

    # 1. Recuperiamo tutti i parametri e buffer del modello instanziato
    wrong_dtype_params = []

    for name, param in model.named_parameters():
        # Verifica Dtype: se hai chiesto float16 e trovi float32, probabilmente non è stato caricato
        # (A meno che tu non abbia castato tutto il modello a float16 prima)
        if param.dtype == torch.float32:
            # Nota: alcuni bias o scalar potrebbero legittimamente essere float32,
            # ma in un modello caricato 'pure fp16', questo è un forte indicatore di warning.
            wrong_dtype_params.append(f"{name} ({param.dtype})")

    # Verifica buffer (come pos_embed)
    for name, buf in model.named_buffers():
        if buf.dtype == torch.float32:
            # Attenzione: alcune maschere o buffer di indici potrebbero essere float32/int64
            # Qui stiamo cercando buffer "pesanti"
            if buf.numel() > 100:  # euristica semplice
                wrong_dtype_params.append(f"Buffer: {name} ({buf.dtype})")

    if not wrong_dtype_params:
        print(
            f"✅ [SUCCESS] Tutti i parametri di {model_name} sembrano avere il dtype corretto (float16/bfloat16)."
        )
        print(
            "   Questo è un ottimo indicatore che sono stati sovrascritti dal checkpoint."
        )
    else:
        print(
            f"⚠️  [WARNING] Trovati {len(wrong_dtype_params)} parametri in float32 (potrebbero non essere stati caricati):"
        )
        for p in wrong_dtype_params[:5]:
            print(f"   - {p}")

    # Conteggio Statistico
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Totale Parametri: {total_params / 1e9:.2f} B")

    return len(wrong_dtype_params) == 0
