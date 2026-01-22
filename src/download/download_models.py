#!/usr/bin/env python3
"""
Script per scaricare i modelli Stable Diffusion 3.5 da HuggingFace.

Modelli supportati:
- SD3.5 Large
- SD3.5 Large Turbo
- SD3.5 Medium
- SD3 Medium (legacy)

Documentazione ufficiale: https://github.com/Stability-AI/sd3.5
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

try:
    from huggingface_hub import hf_hub_download, HfFolder
except ImportError:
    print(
        "Errore: manca il pacchetto 'huggingface-hub'. Installalo con: pip install huggingface-hub",
        file=sys.stderr,
    )
    sys.exit(1)

# Carica .env se disponibile
try:
    from dotenv import load_dotenv

    load_dotenv()
    script_dir_env = Path(__file__).resolve().parent / ".env"
    if script_dir_env.exists():
        load_dotenv(script_dir_env)
except Exception:
    pass


# ============================================================================
# CONFIGURAZIONE MODELLI
# ============================================================================

@dataclass
class ModelConfig:
    """Configurazione per un modello SD3.5."""
    name: str
    description: str
    repo_id: str
    model_file: str
    text_encoder_repo: str = "stabilityai/stable-diffusion-3.5-large"  # Repo per text encoders
    text_encoders: List[str] = field(default_factory=lambda: [
        "text_encoders/clip_l.safetensors",
        "text_encoders/clip_g.safetensors",
        "text_encoders/t5xxl_fp16.safetensors",
    ])


# Modelli disponibili
MODELS: Dict[str, ModelConfig] = {
    "large": ModelConfig(
        name="SD3.5 Large",
        description="Modello completo 8B parametri, qualità massima",
        repo_id="stabilityai/stable-diffusion-3.5-large",
        model_file="sd3.5_large.safetensors",
    ),
    "large-turbo": ModelConfig(
        name="SD3.5 Large Turbo",
        description="Versione veloce del Large, meno step di inferenza",
        repo_id="stabilityai/stable-diffusion-3.5-large-turbo",
        model_file="sd3.5_large_turbo.safetensors",
    ),
    "medium": ModelConfig(
        name="SD3.5 Medium",
        description="Modello 2.5B parametri, bilanciamento qualità/velocità",
        repo_id="stabilityai/stable-diffusion-3.5-medium",
        model_file="sd3.5_medium.safetensors",
    ),
    "sd3-medium": ModelConfig(
        name="SD3 Medium (Legacy)",
        description="Versione precedente SD3 Medium",
        repo_id="stabilityai/stable-diffusion-3-medium",
        model_file="sd3_medium.safetensors",
    ),
}

# ControlNets disponibili (solo per Large)
CONTROLNETS: Dict[str, str] = {
    "blur": "sd3.5_large_controlnet_blur.safetensors",
    "canny": "sd3.5_large_controlnet_canny.safetensors",
    "depth": "sd3.5_large_controlnet_depth.safetensors",
}
CONTROLNET_REPO = "stabilityai/stable-diffusion-3.5-controlnets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scarica modelli SD3.5 e text encoders da HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Scarica SD3.5 Large (default)
  python download_models.py

  # Scarica SD3.5 Medium
  python download_models.py --model medium

  # Scarica SD3.5 Large Turbo
  python download_models.py --model large-turbo

  # Scarica più modelli
  python download_models.py --model large medium

  # Scarica con ControlNets
  python download_models.py --model large --controlnets blur canny depth

  # Scarica solo text encoders
  python download_models.py --only-encoders

  # Lista modelli disponibili
  python download_models.py --list
        """,
    )
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        choices=list(MODELS.keys()),
        default=["large"],
        help="Modello/i da scaricare (default: large)",
    )
    parser.add_argument(
        "--dest", "-d",
        default="models",
        help="Directory di destinazione (default: models)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Riscarica anche se il file esiste già",
    )
    parser.add_argument(
        "--token",
        help="Token HuggingFace (può essere settato anche via HUGGINGFACE_TOKEN env)",
    )
    parser.add_argument(
        "--controlnets", "-c",
        nargs="*",
        choices=list(CONTROLNETS.keys()),
        help="ControlNets da scaricare (solo per Large): blur, canny, depth",
    )
    parser.add_argument(
        "--only-encoders",
        action="store_true",
        help="Scarica solo i text encoders, senza il modello principale",
    )
    parser.add_argument(
        "--skip-encoders",
        action="store_true",
        help="Non scaricare i text encoders (solo modello principale)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Mostra modelli disponibili ed esci",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Non eseguire il postprocessing (spostamento/rinomina file)",
    )
    return parser.parse_args()


def list_models() -> None:
    """Stampa la lista dei modelli disponibili."""
    print("\n📦 MODELLI DISPONIBILI:\n")
    print(f"{'Alias':<15} {'Nome':<25} {'Descrizione'}")
    print("=" * 80)
    for alias, config in MODELS.items():
        print(f"{alias:<15} {config.name:<25} {config.description}")
    
    print("\n🎛️  CONTROLNETS (solo per Large):\n")
    for name, filename in CONTROLNETS.items():
        print(f"  - {name}: {filename}")
    print()


def ensure_token(token_arg: Optional[str]) -> None:
    """Configura il token HuggingFace."""
    if token_arg:
        HfFolder.save_token(token_arg.strip())
        return
    env_token = os.getenv("HUGGINGFACE_TOKEN")
    if env_token:
        HfFolder.save_token(env_token.strip())
        return


def download_file(
    repo_id: str,
    filename: str,
    dest: Path,
    overwrite: bool = False,
) -> bool:
    """
    Scarica un singolo file da HuggingFace.
    
    Returns:
        True se il download è riuscito, False altrimenti.
    """
    # Determina il nome locale del file
    local_name = Path(filename).name
    local_target = dest / local_name
    
    if local_target.exists() and not overwrite:
        print(f"[SKIP] {local_target} esiste già")
        return True
    
    try:
        print(f"[DOWNLOADING] {repo_id}:{filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True
    except Exception as e:
        print(f"[ERROR] Impossibile scaricare {filename}: {e}", file=sys.stderr)
        return False


def download_model(
    model_key: str,
    dest: Path,
    overwrite: bool = False,
    skip_encoders: bool = False,
    only_encoders: bool = False,
) -> None:
    """Scarica un modello specifico con i suoi text encoders."""
    config = MODELS[model_key]
    
    print(f"\n{'='*60}")
    print(f"📥 {config.name}")
    print(f"   {config.description}")
    print(f"   Repo: {config.repo_id}")
    print(f"{'='*60}\n")
    
    dest.mkdir(parents=True, exist_ok=True)
    
    # Scarica il modello principale
    if not only_encoders:
        download_file(config.repo_id, config.model_file, dest, overwrite)
    
    # Scarica i text encoders
    if not skip_encoders:
        print("\n📝 Text Encoders:")
        for encoder in config.text_encoders:
            download_file(config.text_encoder_repo, encoder, dest, overwrite)


def download_controlnets(
    controlnets: List[str],
    dest: Path,
    overwrite: bool = False,
) -> None:
    """Scarica i ControlNets specificati."""
    print(f"\n{'='*60}")
    print("🎛️  ControlNets")
    print(f"{'='*60}\n")
    
    for cn_name in controlnets:
        filename = CONTROLNETS[cn_name]
        download_file(CONTROLNET_REPO, filename, dest, overwrite)


def postprocess(dest: Path) -> None:
    """
    Postprocessing dopo il download:
    - Sposta i file da text_encoders/ alla root
    - Rinomina t5xxl_fp16 -> t5xxl
    """
    # Sposta text encoders
    text_enc_dir = dest / "text_encoders"
    if text_enc_dir.exists() and text_enc_dir.is_dir():
        for item in text_enc_dir.iterdir():
            if item.is_file():
                target = dest / item.name
                if target.exists():
                    print(f"[POST] Skip: {item.name} esiste già in root")
                else:
                    try:
                        item.rename(target)
                        print(f"[POST] Spostato: {item.name}")
                    except Exception as e:
                        print(f"[POST][ERROR] {item}: {e}", file=sys.stderr)
        
        # Rimuovi directory se vuota
        try:
            text_enc_dir.rmdir()
            print("[POST] Rimossa directory text_encoders/")
        except OSError:
            pass
    
    # Rinomina t5xxl
    old_t5 = dest / "t5xxl_fp16.safetensors"
    new_t5 = dest / "t5xxl.safetensors"
    if old_t5.exists() and not new_t5.exists():
        try:
            old_t5.rename(new_t5)
            print(f"[POST] Rinominato: t5xxl_fp16 -> t5xxl")
        except Exception as e:
            print(f"[POST][ERROR] Rinomina t5xxl: {e}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    
    # Lista modelli
    if args.list:
        list_models()
        return 0
    
    # Setup
    ensure_token(args.token)
    dest = Path(args.dest).expanduser().resolve()
    
    print("\n" + "=" * 60)
    print("🚀 STABLE DIFFUSION 3.5 MODEL DOWNLOADER")
    print("=" * 60)
    print(f"\n📁 Destinazione: {dest}")
    print(f"📦 Modelli selezionati: {', '.join(args.model)}")
    
    if args.controlnets:
        print(f"🎛️  ControlNets: {', '.join(args.controlnets)}")
    
    # Scarica ogni modello selezionato
    for model_key in args.model:
        download_model(
            model_key=model_key,
            dest=dest,
            overwrite=args.overwrite,
            skip_encoders=args.skip_encoders,
            only_encoders=args.only_encoders,
        )
    
    # Scarica ControlNets
    if args.controlnets:
        download_controlnets(args.controlnets, dest, args.overwrite)
    
    # Postprocessing
    if not args.no_postprocess:
        print("\n📋 Postprocessing...")
        postprocess(dest)
    
    print(f"\n✅ Completato! File salvati in: {dest}")
    print("\nFile scaricati:")
    for f in sorted(dest.glob("*.safetensors")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.1f} MB)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
