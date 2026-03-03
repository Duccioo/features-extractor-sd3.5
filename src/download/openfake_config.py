"""
Configurazione e costanti per il dataset OpenFake.

Contiene le costanti del dataset, la lista dei modelli noti,
e funzioni helper condivise tra i moduli.
"""

from __future__ import annotations

import re

# ============================================================================
# CONFIGURAZIONE DEL DATASET
# ============================================================================

HF_DATASET_ID = "ComplexDataLab/OpenFake"
DATASET_SIZE = "~1.06 TB"
PAPER_URL = "https://arxiv.org/abs/2509.09495"

TRAIN_EXAMPLES = 1_870_684
TEST_EXAMPLES = 59_658

# Modelli noti nel dataset (lista non esaustiva)
KNOWN_MODELS = [
    "sd-1.5",
    "sd-1.5-dreamshaper",
    "sd-1.5-epicdream",
    "sd-2.1",
    "sdxl",
    "sdxl-epic-realism",
    "sdxl-juggernaut",
    "sdxl-realvis-v5",
    "sdxl-touchofrealism",
    "sd-3.5",
    "flux.1-dev",
    "flux.1-schnell",
    "flux-1.1-pro",
    "flux-realism",
    "flux-amateursnapshotphotos",
    "flux-mvc5000",
    "midjourney-6",
    "midjourney-7",
    "dalle-3",
    "imagen-3.0-002",
    "imagen-4.0",
    "gpt-image-1",
    "ideogram-3.0",
    "grok-2-image-1212",
    "hidream-i1-full",
    "recraft-v3",
    "mystic",
    "chroma",
]


def sanitize_dirname(name: str) -> str:
    """Converte un nome modello in un nome di directory sicuro."""
    sanitized = re.sub(r"[^\w\-.]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_").lower()
