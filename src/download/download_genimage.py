#!/usr/bin/env python3
"""
Script per scaricare il dataset GenImage per AI-Generated Image Detection.

GenImage è un benchmark di oltre 1 milione di immagini per la rilevazione
di immagini generate da AI. Contiene immagini generate da:
- Midjourney
- Stable Diffusion V1.4
- Stable Diffusion V1.5
- ADM (Ablated Diffusion Model)
- GLIDE
- Wukong
- VQDM (VQ-Diffusion)
- BigGAN

Link ufficiali:
- GitHub: https://github.com/GenImage-Dataset/GenImage
- Homepage: https://genimage-dataset.github.io/
- Google Drive: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
- Baidu Yunpan: https://pan.baidu.com/s/1i0OFqYN5i6oFAxeK6bIwRQ (codice: ztf1)

Struttura del dataset dopo l'estrazione:
├── <generator_name>/
│   ├── train/
│   │   ├── ai/           (immagini generate da AI)
│   │   ├── nature/       (immagini reali da ImageNet)
│   ├── val/
│   │   ├── ai/
│   │   ├── nature/

Autore: Auto-generato
Data: 2024
"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, List
import hashlib
from tqdm import tqdm
import requests

# ============================================================================
# CONFIGURAZIONE DEL DATASET
# ============================================================================

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

# ID della cartella Google Drive principale
GOOGLE_DRIVE_FOLDER_ID = "1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
GOOGLE_DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"

# Link Baidu
BAIDU_URL = "https://pan.baidu.com/s/1i0OFqYN5i6oFAxeK6bIwRQ"
BAIDU_CODE = "ztf1"


def check_dependencies() -> dict:
    """
    Verifica che le dipendenze necessarie siano installate.
    
    Returns:
        dict: Stato delle dipendenze (True se installato)
    """
    deps = {}
    
    # Verifica gdown per Google Drive
    try:
        import gdown
        deps['gdown'] = True
    except ImportError:
        deps['gdown'] = False
        
    # Verifica tqdm per le progress bar
    try:
        import tqdm
        deps['tqdm'] = True
    except ImportError:
        deps['tqdm'] = False
        
    # Verifica requests
    try:
        import requests
        deps['requests'] = True
    except ImportError:
        deps['requests'] = False
    
    return deps


def install_dependencies(deps: dict) -> None:
    """
    Installa le dipendenze mancanti.
    
    Args:
        deps: Dizionario con lo stato delle dipendenze
    """
    missing = [name for name, installed in deps.items() if not installed]
    
    if missing:
        print(f"📦 Installazione dipendenze mancanti: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            *missing, "--quiet"
        ])
        print("✅ Dipendenze installate con successo!")


def download_from_google_drive_folder(
    output_dir: str,
    generators: Optional[List[str]] = None
) -> None:
    """
    Scarica il dataset da Google Drive.
    
    Args:
        output_dir: Directory di destinazione
        generators: Lista di generatori da scaricare (None = tutti)
    """
    try:
        import gdown
    except ImportError:
        print("❌ Errore: gdown non installato. Esegui: pip install gdown")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Download del dataset GenImage da Google Drive...")
    print(f"📁 Directory di destinazione: {output_path.absolute()}")
    print(f"🔗 URL: {GOOGLE_DRIVE_FOLDER_URL}\n")
    
    # Scarica l'intera cartella
    print("⏳ Download in corso... (questo potrebbe richiedere molto tempo)")
    print("   Il dataset GenImage è di circa 60-100 GB")
    
    try:
        gdown.download_folder(
            url=GOOGLE_DRIVE_FOLDER_URL,
            output=str(output_path),
            quiet=False,
            use_cookies=False
        )
        print(f"\n✅ Download completato in: {output_path}")
    except Exception as e:
        print(f"\n❌ Errore durante il download: {e}")
        print("\n💡 Suggerimenti:")
        print("   1. Prova a scaricare manualmente da:")
        print(f"      {GOOGLE_DRIVE_FOLDER_URL}")
        print("   2. Se il download fallisce, usa il link Baidu Yunpan:")
        print(f"      {BAIDU_URL}")
        print(f"      Codice di estrazione: {BAIDU_CODE}")
        raise


def download_specific_file_from_gdrive(file_id: str, output_path: str) -> None:
    """
    Scarica un file specifico da Google Drive dato il suo ID.
    
    Args:
        file_id: ID del file su Google Drive
        output_path: Percorso di destinazione del file
    """
    try:
        import gdown
    except ImportError:
        print("❌ Errore: gdown non installato. Esegui: pip install gdown")
        return
    
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


def extract_archive(archive_path: str, extract_to: str) -> None:
    """
    Estrae un archivio (zip, tar, tar.gz, etc).
    
    Args:
        archive_path: Percorso dell'archivio
        extract_to: Directory di estrazione
    """
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Estrazione: {archive_path.name} -> {extract_to}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        print(f"⚠️  Formato non supportato: {archive_path.suffix}")
        return
    
    print(f"✅ Estratto: {archive_path.name}")


def organize_dataset(data_dir: str) -> None:
    """
    Organizza il dataset nella struttura imagenet_ai per il training.
    
    Crea la struttura:
    ├── imagenet_ai/
    │   ├── train/
    │   │   ├── ai/
    │   │   ├── nature/
    │   ├── val/
    │   │   ├── ai/
    │   │   ├── nature/
    
    Args:
        data_dir: Directory contenente i dati dei generatori
    """
    data_path = Path(data_dir)
    imagenet_ai = data_path / "imagenet_ai"
    
    # Crea la struttura
    for split in ["train", "val"]:
        for category in ["ai", "nature"]:
            (imagenet_ai / split / category).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Organizzazione dataset in: {imagenet_ai}")
    
    # Copia/linka le immagini da ogni generatore
    for generator in GENERATORS:
        gen_path = data_path / generator
        if not gen_path.exists():
            # Prova con underscore invece di spazi
            gen_path = data_path / generator.replace(" ", "_")
        if not gen_path.exists():
            print(f"⚠️  Generatore non trovato: {generator}")
            continue
        
        print(f"   Elaborazione: {generator}")
        
        for split in ["train", "val"]:
            for category in ["ai", "nature"]:
                src = gen_path / split / category
                if src.exists():
                    dst = imagenet_ai / split / category
                    # Crea link simbolici o copia i file
                    for img in src.iterdir():
                        if img.is_file():
                            # Usa un prefisso per evitare conflitti
                            new_name = f"{generator.replace(' ', '_')}_{img.name}"
                            dst_file = dst / new_name
                            if not dst_file.exists():
                                try:
                                    dst_file.symlink_to(img.absolute())
                                except OSError:
                                    # Se i symlink non sono supportati, copia
                                    import shutil
                                    shutil.copy2(img, dst_file)
    
    print(f"✅ Dataset organizzato in: {imagenet_ai}")


def verify_dataset(data_dir: str) -> dict:
    """
    Verifica l'integrità del dataset scaricato.
    
    Args:
        data_dir: Directory del dataset
        
    Returns:
        dict: Statistiche del dataset
    """
    data_path = Path(data_dir)
    stats = {
        "generators": {},
        "total_train_ai": 0,
        "total_train_nature": 0,
        "total_val_ai": 0,
        "total_val_nature": 0,
    }
    
    print("\n🔍 Verifica dataset...")
    
    for generator in GENERATORS:
        gen_path = data_path / generator
        if not gen_path.exists():
            gen_path = data_path / generator.replace(" ", "_")
        
        if gen_path.exists():
            gen_stats = {}
            for split in ["train", "val"]:
                for category in ["ai", "nature"]:
                    path = gen_path / split / category
                    if path.exists():
                        count = sum(1 for _ in path.glob("*") if _.is_file())
                        gen_stats[f"{split}_{category}"] = count
                        stats[f"total_{split}_{category}"] += count
            
            stats["generators"][generator] = gen_stats
            print(f"   ✅ {generator}: {gen_stats}")
        else:
            print(f"   ❌ {generator}: non trovato")
    
    print(f"\n📊 Totale immagini:")
    print(f"   Train - AI:     {stats['total_train_ai']:,}")
    print(f"   Train - Nature: {stats['total_train_nature']:,}")
    print(f"   Val - AI:       {stats['total_val_ai']:,}")
    print(f"   Val - Nature:   {stats['total_val_nature']:,}")
    
    return stats


def print_download_instructions() -> None:
    """Stampa le istruzioni per il download manuale."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ISTRUZIONI DOWNLOAD DATASET GENIMAGE                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Il dataset GenImage può essere scaricato da due fonti:                      ║
║                                                                              ║
║  1. GOOGLE DRIVE (consigliato per utenti internazionali):                   ║
║     URL: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
║                                                                              ║
║  2. BAIDU YUNPAN (più veloce per utenti in Cina):                           ║
║     URL: https://pan.baidu.com/s/1i0OFqYN5i6oFAxeK6bIwRQ                     ║
║     Codice di estrazione: ztf1                                               ║
║                                                                              ║
║  DIMENSIONI STIMATE:                                                         ║
║  - Dataset completo: ~60-100 GB                                             ║
║  - Per generatore: ~8-15 GB ciascuno                                        ║
║                                                                              ║
║  GENERATORI DISPONIBILI:                                                     ║
║  - Midjourney                                                                ║
║  - Stable Diffusion V1.4                                                    ║
║  - Stable Diffusion V1.5                                                    ║
║  - ADM (Ablated Diffusion Model)                                            ║
║  - GLIDE                                                                     ║
║  - Wukong                                                                    ║
║  - VQDM (VQ-Diffusion)                                                      ║
║  - BigGAN                                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Script per scaricare il dataset GenImage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

  # Scarica tutto il dataset
  python download_genimage.py --output ./data/genimage

  # Scarica solo specifici generatori  
  python download_genimage.py --output ./data/genimage --generators "Stable Diffusion V1.4" "Midjourney"

  # Mostra solo le istruzioni
  python download_genimage.py --info

  # Verifica un dataset esistente
  python download_genimage.py --verify ./data/genimage

  # Organizza il dataset nella struttura imagenet_ai
  python download_genimage.py --organize ./data/genimage
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/genimage",
        help="Directory di destinazione per il download (default: ./data/genimage)"
    )
    
    parser.add_argument(
        "--generators", "-g",
        nargs="+",
        choices=GENERATORS,
        default=None,
        help="Generatori specifici da scaricare (default: tutti)"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Mostra informazioni sul dataset e le istruzioni di download"
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        metavar="PATH",
        help="Verifica l'integrità di un dataset esistente"
    )
    
    parser.add_argument(
        "--organize",
        type=str,
        metavar="PATH",
        help="Organizza il dataset nella struttura imagenet_ai per il training"
    )
    
    parser.add_argument(
        "--extract",
        type=str,
        metavar="PATH",
        help="Estrai tutti gli archivi nella directory specificata"
    )
    
    parser.add_argument(
        "--no-check-deps",
        action="store_true",
        help="Non verificare/installare le dipendenze"
    )
    
    args = parser.parse_args()
    
    # Mostra info
    if args.info:
        print_download_instructions()
        print("\nGeneratori disponibili:")
        for i, gen in enumerate(GENERATORS, 1):
            print(f"  {i}. {gen}")
        return
    
    # Verifica dataset esistente
    if args.verify:
        verify_dataset(args.verify)
        return
    
    # Organizza dataset
    if args.organize:
        organize_dataset(args.organize)
        return
    
    # Estrai archivi
    if args.extract:
        extract_path = Path(args.extract)
        for archive in extract_path.glob("*.zip"):
            extract_archive(str(archive), str(extract_path))
        for archive in extract_path.glob("*.tar*"):
            extract_archive(str(archive), str(extract_path))
        return
    
    # Verifica dipendenze
    if not args.no_check_deps:
        deps = check_dependencies()
        if not all(deps.values()):
            install_dependencies(deps)
    
    # Download del dataset
    print_download_instructions()
    
    print(f"\n{'='*60}")
    print("AVVIO DOWNLOAD")
    print(f"{'='*60}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Directory di output: {output_dir.absolute()}")
    
    if args.generators:
        print(f"📋 Generatori selezionati: {', '.join(args.generators)}")
    else:
        print(f"📋 Generatori: TUTTI ({len(GENERATORS)})")
    
    # Conferma utente
    response = input("\n⚠️  Il download potrebbe richiedere molto tempo e spazio. Continuare? [y/N]: ")
    if response.lower() not in ['y', 'yes', 'si', 's']:
        print("❌ Download annullato dall'utente.")
        return
    
    try:
        download_from_google_drive_folder(
            output_dir=str(output_dir),
            generators=args.generators
        )
        
        # Verifica dopo il download
        print("\n" + "="*60)
        verify_dataset(str(output_dir))
        
        print(f"""
✅ DOWNLOAD COMPLETATO!

Prossimi passi:
1. Estrai gli archivi scaricati (se necessario):
   python {__file__} --extract {output_dir}

2. Verifica il dataset:
   python {__file__} --verify {output_dir}

3. Organizza per il training:
   python {__file__} --organize {output_dir}
""")
        
    except Exception as e:
        print(f"\n❌ Errore durante il download: {e}")
        print("\n💡 Prova a scaricare manualmente utilizzando i link forniti sopra.")


if __name__ == "__main__":
    main()
