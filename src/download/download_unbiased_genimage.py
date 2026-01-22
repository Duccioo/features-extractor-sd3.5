#!/usr/bin/env python3
"""
Script per scaricare il dataset Unbiased GenImage da Harvard Dataverse.

Unbiased GenImage è un'estensione del dataset GenImage che include:
- Il dataset GenImage originale (~500GB)
- Un file metadata.csv con informazioni aggiuntive su:
  - JPEG Quality Factor (QF)
  - Dimensioni dell'immagine (width/height)
  - Contenuto/classe dell'immagine
  - Compression rate

Questo metadata è necessario per creare subset non biased del dataset,
rimuovendo i bias di compressione e dimensione.

Link ufficiali:
- GitHub: https://github.com/gendetection/UnbiasedGenImage
- Homepage: https://www.unbiased-genimage.org/
- Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AKDIHF
- Paper: https://arxiv.org/abs/2403.17608

Struttura del dataset dopo l'estrazione:
├── GenImage/
│   ├── <generator_name>/
│   │   ├── train/
│   │   │   ├── ai/
│   │   │   ├── nature/
│   │   ├── val/
│   │   │   ├── ai/
│   │   │   ├── nature/
├── metadata.csv

Generatori disponibili (8 in totale):
- Midjourney
- Stable Diffusion V1.4
- Stable Diffusion V1.5
- ADM (Ablated Diffusion Model)
- GLIDE
- Wukong
- VQDM (VQ-Diffusion)
- BigGAN

"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib
from datetime import datetime

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "requests", "tqdm", "--quiet"]
    )
    import requests
    from tqdm import tqdm

# ============================================================================
# CONFIGURAZIONE DEL DATASET
# ============================================================================

# Harvard Dataverse API
DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
DATASET_DOI = "doi:10.7910/DVN/AKDIHF"

# Dimensioni stimate del dataset
DATASET_SIZE = "~500 GB"

# Generatori disponibili nel dataset
GENERATORS = [
    "Midjourney",
    "Stable Diffusion V1.4",
    "Stable Diffusion V1.5",
    "ADM",
    "GLIDE",
    "Wukong",
    "VQDM",
    "BigGAN",
]

# Link alternativi
GOOGLE_DRIVE_URL = (
    "https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
)
PAPER_URL = "https://arxiv.org/abs/2403.17608"
GITHUB_URL = "https://github.com/gendetection/UnbiasedGenImage"


def check_dependencies() -> dict:
    """
    Verifica che le dipendenze necessarie siano installate.

    Returns:
        dict: Stato delle dipendenze (True se installato)
    """
    deps = {}

    # Verifica requests
    try:
        import requests

        deps["requests"] = True
    except ImportError:
        deps["requests"] = False

    # Verifica tqdm per le progress bar
    try:
        import tqdm

        deps["tqdm"] = True
    except ImportError:
        deps["tqdm"] = False

    # Verifica pandas (opzionale, per manipolare il metadata.csv)
    try:
        import pandas

        deps["pandas"] = True
    except ImportError:
        deps["pandas"] = False

    return deps


def install_dependencies(deps: dict) -> None:
    """
    Installa le dipendenze mancanti.

    Args:
        deps: Dizionario con lo stato delle dipendenze
    """
    required = ["requests", "tqdm"]
    missing = [name for name in required if not deps.get(name, False)]

    if missing:
        print(f"📦 Installazione dipendenze mancanti: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "--quiet"]
        )
        print("✅ Dipendenze installate con successo!")


def get_dataset_files() -> List[Dict[str, Any]]:
    """
    Ottiene la lista dei file disponibili nel dataset Dataverse.

    Returns:
        Lista di dizionari con informazioni sui file
    """
    url = f"{DATAVERSE_BASE_URL}/api/datasets/:persistentId/?persistentId={DATASET_DOI}"

    print(f"🔍 Recupero lista file dal Dataverse...")
    print(f"   DOI: {DATASET_DOI}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        data = response.json()
        files = data["data"]["latestVersion"]["files"]

        print(f"✅ Trovati {len(files)} file nel dataset")
        return files

    except requests.exceptions.RequestException as e:
        print(f"❌ Errore nella connessione al Dataverse: {e}")
        raise
    except KeyError as e:
        print(f"❌ Errore nel parsing della risposta: {e}")
        raise


def download_file_from_dataverse(
    file_id: int,
    output_path: Path,
    file_size: Optional[int] = None,
    show_progress: bool = True,
) -> bool:
    """
    Scarica un singolo file dal Dataverse.

    Args:
        file_id: ID del file nel Dataverse
        output_path: Percorso di destinazione
        file_size: Dimensione del file in bytes (per la progress bar)
        show_progress: Se mostrare la progress bar

    Returns:
        True se il download è riuscito
    """
    url = f"{DATAVERSE_BASE_URL}/api/access/datafile/{file_id}"

    try:
        # Usa streaming per file grandi
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        # Crea le directory necessarie
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Scarica con progress bar
        total_size = file_size or int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            if show_progress and total_size > 0:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=output_path.name[:40],
                    leave=False,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Errore download {output_path.name}: {e}")
        # Rimuovi file parziale
        if output_path.exists():
            output_path.unlink()
        return False


def download_unbiased_genimage(
    output_dir: Path,
    continue_download: bool = True,
    metadata_only: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Scarica il dataset Unbiased GenImage dal Dataverse.

    Args:
        output_dir: Directory di destinazione
        continue_download: Se True, salta i file già scaricati
        metadata_only: Se True, scarica solo il file metadata.csv
        show_progress: Se mostrare le progress bar

    Returns:
        Statistiche del download
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Download Unbiased GenImage Dataset")
    print(f"   DOI: {DATASET_DOI}")
    print(f"   Destinazione: {output_dir.absolute()}")
    print(f"   Continua download: {continue_download}")
    print(f"   Solo metadata: {metadata_only}\n")

    # Ottieni la lista dei file
    files = get_dataset_files()

    # Statistiche
    stats = {
        "total_files": len(files),
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "total_bytes": 0,
        "downloaded_bytes": 0,
    }

    # Filtra se richiesto solo metadata
    if metadata_only:
        files = [f for f in files if "metadata" in f["dataFile"]["filename"].lower()]
        print(f"📋 Filtrato a {len(files)} file (solo metadata)")

    # Calcola dimensione totale
    for file_entry in files:
        file_info = file_entry["dataFile"]
        stats["total_bytes"] += file_info.get("filesize", 0)

    total_size_gb = stats["total_bytes"] / (1024**3)
    print(f"📊 Dimensione totale: {total_size_gb:.2f} GB ({len(files)} file)\n")

    # Download dei file
    for i, file_entry in enumerate(files, 1):
        file_info = file_entry["dataFile"]
        file_id = file_info["id"]
        filename = file_info["filename"]
        filesize = file_info.get("filesize", 0)

        # Costruisci il percorso
        directory_label = file_entry.get("directoryLabel", "")
        if directory_label:
            file_path = output_dir / directory_label / filename
        else:
            file_path = output_dir / filename

        # Verifica se già esiste
        if continue_download and file_path.exists():
            existing_size = file_path.stat().st_size
            if existing_size >= filesize:  # File completo
                print(f"⏭️  [{i}/{len(files)}] Skipped: {filename}")
                stats["skipped"] += 1
                continue

        # Download
        print(f"📥 [{i}/{len(files)}] Downloading: {filename}")
        success = download_file_from_dataverse(
            file_id=file_id,
            output_path=file_path,
            file_size=filesize,
            show_progress=show_progress,
        )

        if success:
            print(f"   ✅ Completato: {filename}")
            stats["downloaded"] += 1
            stats["downloaded_bytes"] += filesize
        else:
            stats["failed"] += 1

    # Stampa riepilogo
    print(f"\n{'='*60}")
    print("📊 RIEPILOGO DOWNLOAD")
    print(f"{'='*60}")
    print(f"   File totali:    {stats['total_files']}")
    print(f"   Scaricati:      {stats['downloaded']}")
    print(f"   Saltati:        {stats['skipped']}")
    print(f"   Falliti:        {stats['failed']}")
    print(f"   Bytes scaricati: {stats['downloaded_bytes'] / (1024**3):.2f} GB")

    return stats


def list_dataset_files() -> None:
    """
    Elenca tutti i file disponibili nel dataset.
    """
    files = get_dataset_files()

    print(f"\n📋 FILE NEL DATASET ({len(files)} totale)")
    print("=" * 80)

    total_size = 0
    for i, file_entry in enumerate(files, 1):
        file_info = file_entry["dataFile"]
        filename = file_info["filename"]
        filesize = file_info.get("filesize", 0)
        directory = file_entry.get("directoryLabel", "/")

        total_size += filesize
        size_mb = filesize / (1024**2)

        print(f"{i:3d}. {directory}/{filename}")
        print(f"     Size: {size_mb:.1f} MB | ID: {file_info['id']}")

    print("=" * 80)
    print(f"Dimensione totale: {total_size / (1024**3):.2f} GB")


def extract_archives(data_dir: Path) -> None:
    """
    Estrae tutti gli archivi zip/tar trovati nella directory.

    Args:
        data_dir: Directory contenente gli archivi
    """
    data_dir = Path(data_dir)

    # Cerca file split (GenImage.z01, GenImage.z02, etc.)
    split_files = sorted(data_dir.glob("GenImage.z*"))

    if split_files:
        print(f"\n📦 Trovati {len(split_files)} file split (GenImage.z*)")
        print("   Per estrarre, esegui:")
        print(f"   cat {data_dir}/GenImage.z* > {data_dir}/GenImage_restored.zip")
        print(f"   unzip {data_dir}/GenImage_restored.zip -d {data_dir}")
        return

    # Estrai archivi normali
    archives = list(data_dir.glob("*.zip")) + list(data_dir.glob("*.tar*"))

    if not archives:
        print("⚠️  Nessun archivio trovato.")
        return

    print(f"\n📦 Estrazione di {len(archives)} archivi...")

    for archive in archives:
        print(f"   📂 {archive.name}...")
        try:
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
            elif archive.suffix in [".tar", ".gz", ".tgz"]:
                with tarfile.open(archive, "r:*") as tar_ref:
                    tar_ref.extractall(data_dir)
            print(f"   ✅ Estratto: {archive.name}")
        except Exception as e:
            print(f"   ❌ Errore: {e}")


def verify_dataset(data_dir: Path) -> dict:
    """
    Verifica l'integrità del dataset scaricato.

    Args:
        data_dir: Directory del dataset

    Returns:
        dict: Statistiche del dataset
    """
    data_dir = Path(data_dir)

    stats = {
        "files_found": 0,
        "total_size_bytes": 0,
        "generators": {},
        "metadata_found": False,
    }

    print("\n🔍 Verifica del dataset...")

    # Verifica metadata.csv
    metadata_paths = list(data_dir.rglob("metadata.csv"))
    if metadata_paths:
        stats["metadata_found"] = True
        print(f"   ✅ metadata.csv trovato: {metadata_paths[0]}")
    else:
        print("   ⚠️  metadata.csv non trovato")

    # Conta file e dimensioni
    for file in data_dir.rglob("*"):
        if file.is_file():
            stats["files_found"] += 1
            stats["total_size_bytes"] += file.stat().st_size

    # Cerca i generatori
    for generator in GENERATORS:
        gen_variants = [
            generator,
            generator.replace(" ", "_"),
            generator.replace(" ", "-"),
            generator.lower(),
            generator.lower().replace(" ", "_"),
        ]

        for variant in gen_variants:
            gen_path = data_dir / variant
            if gen_path.exists():
                # Conta immagini
                img_count = sum(1 for _ in gen_path.rglob("*.png"))
                img_count += sum(1 for _ in gen_path.rglob("*.jpg"))
                img_count += sum(1 for _ in gen_path.rglob("*.jpeg"))

                stats["generators"][generator] = {
                    "path": str(gen_path),
                    "images": img_count,
                }
                print(f"   ✅ {generator}: {img_count:,} immagini")
                break

    print(f"\n📊 Riepilogo:")
    print(f"   File totali: {stats['files_found']:,}")
    print(f"   Dimensione: {stats['total_size_bytes'] / (1024**3):.2f} GB")
    print(f"   Generatori trovati: {len(stats['generators'])}/{len(GENERATORS)}")

    return stats


def create_unbiased_subset_example(metadata_path: Path, output_dir: Path) -> None:
    """
    Mostra un esempio di come creare un subset unbiased.

    Questo è solo un esempio - l'utente può personalizzare i filtri.
    """
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  pandas non installato. Installa con: pip install pandas")
        return

    print(f"\n📊 Esempio di creazione subset unbiased")
    print(f"   Metadata: {metadata_path}")

    df = pd.read_csv(metadata_path)

    print(f"\n   Colonne disponibili: {list(df.columns)}")
    print(f"   Righe totali: {len(df):,}")

    if "generator" in df.columns:
        print(f"\n   Generatori:")
        for gen, count in df["generator"].value_counts().items():
            print(f"      - {gen}: {count:,}")

    print(
        """
    
📝 Per creare un subset unbiased (esempio Wukong 512x512):

    import pandas as pd
    
    df = pd.read_csv("metadata.csv")
    
    # Seleziona immagini naturali con dimensioni specifiche
    df_unbiased_natural = df[
        (df["generator"] == "nature") &
        (df["width"] >= 450) &
        (df["height"] >= 450) &
        (df["width"] <= 550) &
        (df["height"] <= 550) &
        (df["compression_rate"] == 96)
    ]
    
    # Seleziona immagini generate
    df_unbiased_ai = df[df["generator"] == "wukong"]
    
    # Combina
    df_unbiased = pd.concat([df_unbiased_natural, df_unbiased_ai])
    
    print(f"Subset unbiased: {len(df_unbiased)} immagini")
    """
    )


def print_download_instructions() -> None:
    """Stampa le istruzioni per il download."""
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║               ISTRUZIONI DOWNLOAD UNBIASED GENIMAGE DATASET                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Unbiased GenImage è un'estensione del dataset GenImage che include:        ║
║  - Il dataset GenImage originale (~500GB)                                   ║
║  - Un file metadata.csv con informazioni sui bias                           ║
║                                                                              ║
║  FONTI DI DOWNLOAD:                                                          ║
║                                                                              ║
║  1. HARVARD DATAVERSE (dataset completo + metadata):                        ║
║     https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AKDIHF
║                                                                              ║
║  2. GOOGLE DRIVE (solo GenImage, più veloce):                               ║
║     https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
║     ℹ️ Poi scarica solo metadata.csv dal Dataverse                          ║
║                                                                              ║
║  OPZIONI DI DOWNLOAD:                                                        ║
║                                                                              ║
║  A. Solo metadata.csv (~100MB):                                             ║
║     python download_unbiased_genimage.py --output ./data --metadata-only    ║
║                                                                              ║
║  B. Dataset completo (~500GB):                                              ║
║     python download_unbiased_genimage.py --output ./data                    ║
║                                                                              ║
║  C. Continua un download interrotto:                                        ║
║     python download_unbiased_genimage.py --output ./data --continue         ║
║                                                                              ║
║  GENERATORI DISPONIBILI (8 totale):                                         ║
║  - Midjourney          - GLIDE                                              ║
║  - Stable Diffusion V1.4  - Wukong                                          ║
║  - Stable Diffusion V1.5  - VQDM                                            ║
║  - ADM                    - BigGAN                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Script per scaricare il dataset Unbiased GenImage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

  # Mostra informazioni sul dataset
  python download_unbiased_genimage.py --info

  # Elenca tutti i file disponibili
  python download_unbiased_genimage.py --list

  # Scarica solo il metadata.csv
  python download_unbiased_genimage.py --output ./data/unbiased_genimage --metadata-only

  # Scarica tutto il dataset
  python download_unbiased_genimage.py --output ./data/unbiased_genimage

  # Continua un download interrotto
  python download_unbiased_genimage.py --output ./data/unbiased_genimage --continue

  # Verifica un dataset esistente
  python download_unbiased_genimage.py --verify ./data/unbiased_genimage

  # Estrazione archivi (dopo il download)
  python download_unbiased_genimage.py --extract ./data/unbiased_genimage
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./data/unbiased_genimage",
        help="Directory di destinazione per il download (default: ./data/unbiased_genimage)",
    )

    parser.add_argument(
        "--continue",
        dest="continue_download",
        action="store_true",
        default=True,
        help="Continua il download, saltando file già scaricati (default: True)",
    )

    parser.add_argument(
        "--no-continue",
        dest="continue_download",
        action="store_false",
        help="Ri-scarica tutti i file, anche se esistono",
    )

    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Scarica solo il file metadata.csv (consigliato se GenImage già scaricato)",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Mostra informazioni sul dataset e le istruzioni di download",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Elenca tutti i file disponibili nel dataset",
    )

    parser.add_argument(
        "--verify",
        type=str,
        metavar="PATH",
        help="Verifica l'integrità di un dataset esistente",
    )

    parser.add_argument(
        "--extract",
        type=str,
        metavar="PATH",
        help="Estrai tutti gli archivi nella directory specificata",
    )

    parser.add_argument(
        "--no-check-deps",
        action="store_true",
        help="Non verificare/installare le dipendenze",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Modalità silenziosa, riduce l'output",
    )

    args = parser.parse_args()

    # Mostra info
    if args.info:
        print_download_instructions()
        print("\n📚 Paper: https://arxiv.org/abs/2403.17608")
        print(f"🔗 GitHub: {GITHUB_URL}")
        print(f"💾 Dimensione stimata: {DATASET_SIZE}")
        print("\nGeneratori disponibili:")
        for i, gen in enumerate(GENERATORS, 1):
            print(f"  {i}. {gen}")
        return 0

    # Lista file
    if args.list:
        list_dataset_files()
        return 0

    # Verifica dataset esistente
    if args.verify:
        verify_dataset(Path(args.verify))
        return 0

    # Estrai archivi
    if args.extract:
        extract_archives(Path(args.extract))
        return 0

    # Verifica dipendenze
    if not args.no_check_deps:
        deps = check_dependencies()
        if not all(deps.get(d, False) for d in ["requests", "tqdm"]):
            install_dependencies(deps)

    # Download del dataset
    print_download_instructions()

    print(f"\n{'='*60}")
    print("AVVIO DOWNLOAD")
    print(f"{'='*60}")

    output_dir = Path(args.output)

    print(f"\n📁 Directory di output: {output_dir.absolute()}")

    if args.metadata_only:
        print("📋 Modalità: Solo metadata.csv")
    else:
        print(f"📋 Modalità: Dataset completo ({DATASET_SIZE})")

    # Conferma utente
    if not args.quiet:
        if args.metadata_only:
            response = input("\n⚠️  Procedere con il download del metadata? [Y/n]: ")
        else:
            response = input("\n⚠️  Il download completo è ~500GB. Continuare? [y/N]: ")
            if response.lower() not in ["y", "yes", "si", "s"]:
                print("❌ Download annullato dall'utente.")
                return 1

    try:
        stats = download_unbiased_genimage(
            output_dir=output_dir,
            continue_download=args.continue_download,
            metadata_only=args.metadata_only,
            show_progress=not args.quiet,
        )

        if stats["failed"] > 0:
            print(f"\n⚠️  {stats['failed']} file non scaricati. Riprova con --continue")
            return 1

        print(
            f"""
✅ DOWNLOAD COMPLETATO!

Prossimi passi:
1. Se hai file split (GenImage.z*), combina e estrai:
   cat {output_dir}/GenImage.z* > {output_dir}/GenImage_restored.zip
   unzip {output_dir}/GenImage_restored.zip -d {output_dir}

2. Verifica il dataset:
   python {sys.argv[0]} --verify {output_dir}

3. Consulta il paper per creare subset unbiased:
   {PAPER_URL}
"""
        )
        return 0

    except Exception as e:
        print(f"\n❌ Errore durante il download: {e}")
        print("\n💡 Suggerimenti:")
        print("   1. Verifica la connessione internet")
        print("   2. Prova a scaricare manualmente da Google Drive:")
        print(f"      {GOOGLE_DRIVE_URL}")
        print("   3. Poi scarica solo metadata.csv con --metadata-only")
        return 1


if __name__ == "__main__":
    sys.exit(main())
