#!/usr/bin/env python3
"""
Script per scaricare il dataset Tiny GenImage da Kaggle.

Tiny GenImage è una versione ridotta del dataset GenImage, ideale per 
sperimentazioni su macchine con risorse limitate.

Caratteristiche:
- 5000 immagini per generatore (train + val)
- Dimensione totale: ~8 GB (vs ~60-100 GB del dataset completo)
- Escluso: Stable Diffusion V1.4
- Generatori: Midjourney, SD V1.5, ADM, GLIDE, Wukong, VQDM, BigGAN

Link Kaggle: https://www.kaggle.com/datasets/yangsangtai/tiny-genimage
Licenza: CC BY-NC-SA 4.0
"""

import os
import sys
import argparse
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

KAGGLE_DATASET = "yangsangtai/tiny-genimage"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
DATASET_SIZE = "~8 GB"

# Generatori disponibili in Tiny GenImage (escluso SD V1.4)
GENERATORS = [
    "Midjourney",
    "Stable Diffusion V1.5",
    "ADM",
    "GLIDE",
    "Wukong",
    "VQDM",
    "BigGAN",
]


def check_kaggle_cli() -> bool:
    """Verifica se kaggle CLI è installato e configurato."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_kaggle_credentials() -> bool:
    """Verifica se le credenziali Kaggle sono configurate."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    
    # Controlla variabili d'ambiente
    return bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))


def install_kaggle_cli() -> None:
    """Installa kaggle CLI."""
    print("📦 Installazione kaggle CLI...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "kaggle", "--quiet"
    ])
    print("✅ kaggle CLI installato!")


def setup_kaggle_credentials() -> None:
    """Guida l'utente nella configurazione delle credenziali Kaggle."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CONFIGURAZIONE CREDENZIALI KAGGLE                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Per scaricare dataset da Kaggle, devi configurare le credenziali API.      ║
║                                                                              ║
║  OPZIONE 1 - File kaggle.json (consigliato):                                ║
║  1. Vai su https://www.kaggle.com/settings                                  ║
║  2. Scorri fino a "API" e clicca "Create New Token"                         ║
║  3. Salva il file kaggle.json scaricato in: ~/.kaggle/kaggle.json           ║
║  4. Imposta i permessi: chmod 600 ~/.kaggle/kaggle.json                     ║
║                                                                              ║
║  OPZIONE 2 - Variabili d'ambiente:                                          ║
║  export KAGGLE_USERNAME="tuo_username"                                       ║
║  export KAGGLE_KEY="tua_api_key"                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def download_dataset(output_dir: Path, unzip: bool = True) -> bool:
    """
    Scarica il dataset Tiny GenImage da Kaggle.
    
    Args:
        output_dir: Directory di destinazione
        unzip: Se True, estrae automaticamente il dataset
        
    Returns:
        True se il download è riuscito
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Download Tiny GenImage da Kaggle...")
    print(f"📁 Destinazione: {output_dir.absolute()}")
    print(f"📦 Dataset: {KAGGLE_DATASET}")
    print(f"💾 Dimensione: {DATASET_SIZE}\n")
    
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(output_dir),
    ]
    
    if unzip:
        cmd.append("--unzip")
    
    try:
        print(f"⏳ Esecuzione: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Download completato!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Errore durante il download: {e}")
        return False


def extract_dataset(zip_path: Path, output_dir: Path) -> None:
    """Estrae manualmente il dataset se non è stato estratto automaticamente."""
    if not zip_path.exists():
        print(f"❌ File non trovato: {zip_path}")
        return
    
    print(f"📦 Estrazione: {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"✅ Estratto in: {output_dir}")


def verify_dataset(data_dir: Path) -> dict:
    """
    Verifica la struttura del dataset scaricato.
    
    Args:
        data_dir: Directory del dataset
        
    Returns:
        dict: Statistiche del dataset
    """
    stats = {
        "generators": {},
        "total_train_ai": 0,
        "total_train_nature": 0,
        "total_val_ai": 0,
        "total_val_nature": 0,
    }
    
    print("\n🔍 Verifica dataset Tiny GenImage...")
    
    # Cerca la directory principale del dataset
    possible_roots = [
        data_dir,
        data_dir / "tiny-genimage",
        data_dir / "tiny_genimage",
    ]
    
    root = None
    for p in possible_roots:
        if p.exists() and any(p.iterdir()):
            root = p
            break
    
    if not root:
        print(f"⚠️  Nessuna directory valida trovata in {data_dir}")
        return stats
    
    print(f"📁 Directory root: {root}")
    
    for generator in GENERATORS:
        # Prova diverse convenzioni di naming
        gen_paths = [
            root / generator,
            root / generator.replace(" ", "_"),
            root / generator.replace(" ", "-"),
            root / generator.lower().replace(" ", "_"),
        ]
        
        gen_path = None
        for gp in gen_paths:
            if gp.exists():
                gen_path = gp
                break
        
        if gen_path and gen_path.exists():
            gen_stats = {}
            for split in ["train", "val"]:
                for category in ["ai", "nature"]:
                    path = gen_path / split / category
                    if path.exists():
                        count = sum(1 for f in path.iterdir() if f.is_file())
                        gen_stats[f"{split}_{category}"] = count
                        stats[f"total_{split}_{category}"] += count
            
            stats["generators"][generator] = gen_stats
            total = sum(gen_stats.values())
            print(f"   ✅ {generator}: {total:,} immagini")
        else:
            print(f"   ⚠️  {generator}: non trovato")
    
    print(f"\n📊 Totale immagini:")
    print(f"   Train - AI:     {stats['total_train_ai']:,}")
    print(f"   Train - Nature: {stats['total_train_nature']:,}")
    print(f"   Val - AI:       {stats['total_val_ai']:,}")
    print(f"   Val - Nature:   {stats['total_val_nature']:,}")
    total = sum([
        stats['total_train_ai'], stats['total_train_nature'],
        stats['total_val_ai'], stats['total_val_nature']
    ])
    print(f"   TOTALE:         {total:,}")
    
    return stats


def print_info() -> None:
    """Stampa informazioni sul dataset."""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          TINY GENIMAGE DATASET                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Versione ridotta del dataset GenImage per sperimentazioni veloci.          ║
║                                                                              ║
║  📊 STATISTICHE:                                                             ║
║  - Immagini per generatore: ~5000 (train + val)                             ║
║  - Dimensione totale: ~8 GB                                                 ║
║  - Licenza: CC BY-NC-SA 4.0                                                 ║
║                                                                              ║
║  🔗 LINK:                                                                    ║
║  {KAGGLE_URL:<68} ║
║                                                                              ║
║  🤖 GENERATORI INCLUSI:                                                      ║""")
    
    for gen in GENERATORS:
        print(f"║  - {gen:<71} ║")
    
    print("""║                                                                              ║
║  ⚠️  ESCLUSO: Stable Diffusion V1.4                                         ║
║                                                                              ║
║  📁 STRUTTURA:                                                               ║
║  tiny-genimage/                                                              ║
║  ├── <generator>/                                                            ║
║  │   ├── train/                                                              ║
║  │   │   ├── ai/                                                             ║
║  │   │   └── nature/                                                         ║
║  │   └── val/                                                                ║
║  │       ├── ai/                                                             ║
║  │       └── nature/                                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Scarica il dataset Tiny GenImage da Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Mostra informazioni sul dataset
  python download_tiny_genimage.py --info

  # Scarica il dataset
  python download_tiny_genimage.py --output ./data/tiny-genimage

  # Scarica senza estrarre
  python download_tiny_genimage.py --output ./data/tiny-genimage --no-unzip

  # Verifica un dataset esistente
  python download_tiny_genimage.py --verify ./data/tiny-genimage

  # Estrai un archivio esistente
  python download_tiny_genimage.py --extract ./data/tiny-genimage.zip --output ./data/
        """,
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/tiny-genimage",
        help="Directory di destinazione (default: ./data/tiny-genimage)",
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Mostra informazioni sul dataset",
    )
    parser.add_argument(
        "--verify", "-v",
        type=str,
        metavar="PATH",
        help="Verifica un dataset esistente",
    )
    parser.add_argument(
        "--extract", "-e",
        type=str,
        metavar="ZIP_PATH",
        help="Estrai un archivio zip esistente",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Non estrarre automaticamente il dataset dopo il download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forza il download anche se i file esistono già",
    )
    
    args = parser.parse_args()
    
    # Info
    if args.info:
        print_info()
        return 0
    
    # Verifica
    if args.verify:
        verify_dataset(Path(args.verify))
        return 0
    
    # Estrazione manuale
    if args.extract:
        extract_dataset(
            Path(args.extract),
            Path(args.output)
        )
        return 0
    
    # Download
    print_info()
    
    # Verifica kaggle CLI
    if not check_kaggle_cli():
        print("⚠️  kaggle CLI non trovato.")
        install_kaggle_cli()
    
    # Verifica credenziali
    if not check_kaggle_credentials():
        setup_kaggle_credentials()
        print("\n❌ Credenziali Kaggle non configurate. Segui le istruzioni sopra.")
        return 1
    
    output_dir = Path(args.output).expanduser().resolve()
    
    # Verifica se esiste già
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"\n⚠️  La directory {output_dir} non è vuota.")
        response = input("Continuare comunque? [y/N]: ")
        if response.lower() not in ['y', 'yes', 's', 'si']:
            print("❌ Download annullato.")
            return 0
    
    # Download
    success = download_dataset(output_dir, unzip=not args.no_unzip)
    
    if success:
        print("\n" + "=" * 60)
        verify_dataset(output_dir)
        print(f"""
✅ DOWNLOAD COMPLETATO!

Il dataset Tiny GenImage è stato scaricato in:
{output_dir}

Prossimi passi:
1. Verifica il dataset: python {Path(__file__).name} --verify {output_dir}
2. Usa il dataset per il training o l'estrazione di features
        """)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
