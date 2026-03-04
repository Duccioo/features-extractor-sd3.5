#!/usr/bin/env python3
"""
Script per scaricare il dataset OpenFake da HuggingFace.

OpenFake è un dataset per la valutazione di deepfake detection e attribuzione,
contenente immagini reali e sintetiche generate da 20+ modelli generativi.

Dimensione totale: ~1.06 TB (1.87M train + 59.6K test)
Poiché il dataset è enorme, questo script usa **streaming** per scaricare
solo le immagini desiderate senza scaricare l'intero dataset.

Strategie per il download parziale:
  - Filtraggio per modello generativo (--models)
  - Filtraggio per label real/fake (--labels)
  - Limite globale per split (--limit)
  - Limite per singolo modello (--limit-per-model)

Link ufficiali:
  - HuggingFace: https://huggingface.co/datasets/ComplexDataLab/OpenFake
  - Paper: https://arxiv.org/abs/2509.09495

Struttura output:
  ├── {split}/
  │   ├── real/
  │   │   ├── img_000000.png
  │   │   └── ...
  │   ├── fake/
  │   │   ├── {model_name}/
  │   │   │   ├── img_000000.png
  │   │   │   └── ...
  │   │   └── ...
  ├── metadata_{split}.csv

Modelli coperti (20+ in totale):
  - Stable Diffusion 1.5, 2.1, XL, 3.5
  - Flux 1.0-dev, 1.1-Pro, 1.0-Schnell
  - Midjourney v6, v7
  - DALL·E 3, Imagen 3, Imagen 4
  - GPT Image 1, Ideogram 3.0, Grok-2, HiDream-I1, Recraft v3, Chroma
  - Plus community LoRA/finetuned variants
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from openfake_utils.openfake_config import (
        DATASET_SIZE, PAPER_URL, TRAIN_EXAMPLES, TEST_EXAMPLES, KNOWN_MODELS,
    )
    from openfake_utils.openfake_download import download_openfake
    from openfake_utils.openfake_verify import verify_dataset, list_available_models
    from openfake_utils.openfake_tui import check_tui_deps, interactive_mode
except ImportError:
    try:
        from .openfake_utils.openfake_config import (
            DATASET_SIZE, PAPER_URL, TRAIN_EXAMPLES, TEST_EXAMPLES, KNOWN_MODELS,
        )
        from .openfake_download import download_openfake
        from .openfake_utils.openfake_verify import verify_dataset, list_available_models
        from .openfake_utils.openfake_tui import check_tui_deps, interactive_mode
    except ImportError:
        # Fallback for when running directly and imports are flat (e.g. if files were moved)
        from openfake_config import (
            DATASET_SIZE, PAPER_URL, TRAIN_EXAMPLES, TEST_EXAMPLES, KNOWN_MODELS,
        )
        from openfake_download import download_openfake
        from openfake_verify import verify_dataset, list_available_models
        from openfake_tui import check_tui_deps, interactive_mode


def print_download_instructions() -> None:
    """Stampa le istruzioni per il download."""
    print(
        f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 OPENFAKE DATASET - DOWNLOAD INSTRUCTIONS                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  OpenFake: Deepfake detection & attribution dataset                        ║
║  Dimensione totale: {DATASET_SIZE} ({TRAIN_EXAMPLES + TEST_EXAMPLES:,} immagini)                  ║
║  Modelli: 20+ generatori (SD 1.5-3.5, Flux, Midjourney, DALL·E 3, ...)   ║
║                                                                            ║
║  MODALITÀ DI DOWNLOAD:                                                     ║
║                                                                            ║
║  A. TUI interattiva (consigliata):                                         ║
║     python download_openfake.py --interactive                              ║
║                                                                            ║
║  B. Mini subset per testing (~50MB):                                       ║
║     python download_openfake.py -o data/OpenFake-mini \\                    ║
║       --limit 100 --limit-per-model 20                                     ║
║                                                                            ║
║  C. Solo un modello specifico:                                             ║
║     python download_openfake.py -o data/OpenFake \\                         ║
║       --models "Stable Diffusion 3.5" --limit-per-model 500               ║
║                                                                            ║
║  D. Dataset completo (ATTENZIONE: ~1.06 TB!):                             ║
║     python download_openfake.py -o data/OpenFake --quiet                   ║
║                                                                            ║
║  OPZIONI DI FILTRAGGIO:                                                    ║
║    --models     Filtra per modelli specifici                               ║
║    --labels     Filtra per label (real/fake)                               ║
║    --limit      Limite globale per split                                   ║
║    --limit-per-model  Limite per singolo modello                           ║
║                                                                            ║
║  📄 Paper: {PAPER_URL}                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Script per scaricare il dataset OpenFake (streaming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Esempi di utilizzo:

  # TUI interattiva
  python download_openfake.py --interactive

  # Mostra informazioni sul dataset
  python download_openfake.py --info

  # Elenca i modelli disponibili nel dataset
  python download_openfake.py --list-models

  # Mini download per test (100 immagini)
  python download_openfake.py -o data/OpenFake-mini --limit 100

  # Scarica solo SD 3.5 fake (max 500 per modello)
  python download_openfake.py -o data/OpenFake \\
    --models "Stable Diffusion 3.5" --labels fake --limit-per-model 500

  # Scarica solo il test set
  python download_openfake.py -o data/OpenFake --split test

  # Verifica un dataset esistente
  python download_openfake.py --verify data/OpenFake
        """,
    )

    parser.add_argument(
        "--output", "-o", type=str, default="./data/OpenFake",
        help="Directory di destinazione (default: ./data/OpenFake)",
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "test", "both"], default="both",
        help="Split da scaricare (default: both)",
    )
    parser.add_argument(
        "--models", nargs="+", type=str, default=None,
        help="Filtra per modelli specifici (es. --models 'Stable Diffusion 3.5' 'Midjourney v6')",
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, choices=["real", "fake"], default=None,
        help="Filtra per label (es. --labels fake)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limite globale di immagini per split",
    )
    parser.add_argument(
        "--limit-per-model", type=int, default=None,
        help="Limite di immagini per singolo modello per split",
    )
    parser.add_argument(
        "--limit-real", type=int, default=None,
        help="Limite di immagini reali per split (se omesso usa --limit-per-model)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed per shuffling riproducibile (default: 42)",
    )
    parser.add_argument(
        "--format", dest="img_format", type=str, choices=["PNG", "JPEG"], default="PNG",
        help="Formato immagini output (default: PNG)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Avvia la TUI interattiva",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Mostra informazioni sul dataset ed esce",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Elenca i modelli disponibili nel dataset (richiede streaming)",
    )
    parser.add_argument(
        "--list-models-scan", type=int, default=50_000,
        help="Numero di esempi da scansionare per --list-models (default: 50000)",
    )
    parser.add_argument(
        "--verify", type=str, metavar="PATH",
        help="Verifica un dataset esistente",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Modalità silenziosa",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Disabilita lo shuffle (usa ordine originale del dataset)",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Forza lo shuffle anche con limit piccoli",
    )

    args = parser.parse_args()

    # ── Info ───────────────────────────────────────────────────
    if args.info:
        print_download_instructions()
        print("Modelli noti nel dataset:")
        for i, model in enumerate(KNOWN_MODELS, 1):
            print(f"  {i:2d}. {model}")
        return 0

    # ── List models ────────────────────────────────────────────
    if args.list_models:
        models = list_available_models(split="train", max_scan=args.list_models_scan)
        print("\n📋 MODELLI TROVATI:")
        for i, model in enumerate(models, 1):
            known_marker = " ✓" if model in KNOWN_MODELS else " [new]"
            print(f"  {i:2d}. {model}{known_marker}")
        return 0

    # ── Verify ─────────────────────────────────────────────────
    if args.verify:
        verify_dataset(Path(args.verify))
        return 0

    # ── Interactive TUI ────────────────────────────────────────
    if args.interactive:
        if not check_tui_deps():
            print(
                "❌ La TUI interattiva richiede pacchetti aggiuntivi.\n"
                "   Installa con: pip install rich questionary",
                file=sys.stderr,
            )
            return 1
        result = interactive_mode()
        return 0 if result is not None else 1

    # ── Download diretto ───────────────────────────────────────
    splits = ["train", "test"] if args.split == "both" else [args.split]

    if not args.quiet:
        print_download_instructions()

    # Conferma utente per download grandi
    if not args.quiet and not args.limit and not args.limit_per_model:
        response = input(
            f"\n⚠️  Stai per scaricare OpenFake senza limiti ({DATASET_SIZE}).\n"
            "   Vuoi continuare? [y/N]: "
        )
        if response.lower() not in ["y", "yes", "si", "s"]:
            print("❌ Download annullato.")
            return 1

    # Determina se shuffle è disabilitato
    no_shuffle = args.no_shuffle
    if args.shuffle:
        no_shuffle = False

    try:
        stats = download_openfake(
            output_dir=Path(args.output),
            splits=splits,
            models_filter=args.models,
            labels_filter=args.labels,
            limit=args.limit,
            limit_per_model=args.limit_per_model,
            limit_real=args.limit_real,
            seed=args.seed,
            img_format=args.img_format,
            quiet=args.quiet,
            no_shuffle=no_shuffle,
        )

        total_saved = sum(s["saved"] for s in stats.values())
        total_errors = sum(s["skipped_error"] for s in stats.values())

        if total_errors > 0:
            print(f"\n⚠️  {total_errors} immagini con errori.")

        if total_saved == 0:
            print("\n⚠️  Nessuna immagine scaricata. Controlla i filtri.")
            return 1

        print(f"\n✅ Download completato! {total_saved:,} immagini in {args.output}")
        return 0

    except Exception as e:
        import traceback
        print(f"\n❌ Errore durante il download: {e}")
        traceback.print_exc()
        print(f"\n💡 Suggerimenti:")
        print(f"   1. Verifica la connessione internet")
        print(f"   2. Prova con meno immagini: --limit 100")
        print(f"   3. Prova la TUI: --interactive")
        return 1


if __name__ == "__main__":
    try:
        code = main()
        # Aggirare il bug di pyarrow/datasets core dump allo shutdown in WSL
        os._exit(code)
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
