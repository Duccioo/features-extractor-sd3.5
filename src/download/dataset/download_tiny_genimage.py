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
import shutil

# Per download streaming a basso consumo di memoria
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Carica .env se disponibile
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Prova anche dalla directory dello script
    script_dir_env = Path(__file__).resolve().parent.parent.parent / ".env"
    if script_dir_env.exists():
        load_dotenv(script_dir_env)
except ImportError:
    pass

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

# Mapping tra nomi display e pattern delle cartelle effettive nel dataset
# Il dataset Tiny GenImage usa convenzioni di naming diverse
GENERATOR_FOLDER_PATTERNS = {
    "Midjourney": ["imagenet_midjourney", "midjourney"],
    "Stable Diffusion V1.5": ["imagenet_ai_0424_sdv5", "sdv5", "sd_v1.5", "stable_diffusion_v1.5"],
    "ADM": ["imagenet_ai_0508_adm", "adm"],
    "GLIDE": ["imagenet_glide", "glide"],
    "Wukong": ["imagenet_ai_0424_wukong", "wukong"],
    "VQDM": ["imagenet_ai_0419_vqdm", "vqdm"],
    "BigGAN": ["imagenet_ai_0419_biggan", "biggan"],
}


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


# ============================================================================
# FUNZIONI PER DOWNLOAD A BASSO CONSUMO DI MEMORIA
# ============================================================================

def get_kaggle_api():
    """Ottiene un'istanza dell'API Kaggle configurata."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"⚠️  Impossibile inizializzare API Kaggle: {e}")
        return None


def download_with_streaming(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Scarica un file usando streaming per ridurre l'uso di memoria.
    
    Args:
        url: URL del file da scaricare
        output_path: Path di destinazione
        chunk_size: Dimensione dei chunk (default 8KB)
        
    Returns:
        True se il download è riuscito
    """
    if not HAS_REQUESTS:
        print("❌ Modulo 'requests' non disponibile. Installa con: pip install requests")
        return False
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if HAS_TQDM and total_size > 0:
            progress = tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name)
        else:
            progress = None
            
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if progress:
                        progress.update(len(chunk))
        
        if progress:
            progress.close()
            
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il download: {e}")
        return False


def download_dataset_low_memory(output_dir: Path, chunk_size: int = 8192) -> bool:
    """
    Scarica il dataset usando l'API Kaggle Python con streaming a basso consumo di memoria.
    
    Questo metodo evita il problema di OOM che si verifica con il CLI standard.
    
    Args:
        output_dir: Directory di destinazione
        chunk_size: Dimensione dei chunk per lo streaming (default 8KB)
        
    Returns:
        True se il download è riuscito
    """
    api = get_kaggle_api()
    if not api:
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "tiny-genimage.zip"
    
    print(f"\n📥 Download Tiny GenImage (modalità basso consumo memoria)...")
    print(f"📁 Destinazione: {output_dir.absolute()}")
    print(f"📦 Dataset: {KAGGLE_DATASET}")
    print(f"💾 Dimensione: {DATASET_SIZE}")
    print(f"🔧 Chunk size: {chunk_size / 1024:.1f} KB\n")
    
    try:
        # Usa download senza unzip per controllare meglio la memoria
        print("⏳ Scaricamento in corso (questo potrebbe richiedere tempo)...")
        
        # L'API Kaggle ha un metodo che scarica direttamente su disco
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(output_dir),
            unzip=False,  # NON estrarre durante il download
            quiet=False,
            force=True
        )
        
        print(f"\n✅ Download completato: {zip_path}")
        return True
        
    except MemoryError:
        print("\n❌ MemoryError durante il download!")
        print("💡 Prova con --download-parts per scaricare i file singolarmente.")
        return False
        
    except Exception as e:
        print(f"\n❌ Errore durante il download: {e}")
        
        # Se fallisce, prova il metodo alternativo con file singoli
        if "memory" in str(e).lower():
            print("💡 Prova con --download-parts per scaricare i file singolarmente.")
        return False


def extract_zip_streaming(zip_path: Path, output_dir: Path, chunk_size: int = 8192) -> bool:
    """
    Estrae un file ZIP in streaming per ridurre l'uso di memoria.
    
    Args:
        zip_path: Path del file ZIP
        output_dir: Directory di destinazione
        chunk_size: Dimensione dei chunk (default 8KB)
        
    Returns:
        True se l'estrazione è riuscita
    """
    if not zip_path.exists():
        print(f"❌ File non trovato: {zip_path}")
        return False
    
    print(f"\n📦 Estrazione streaming: {zip_path.name}")
    print(f"📁 Destinazione: {output_dir}")
    print(f"🔧 Chunk size: {chunk_size / 1024:.1f} KB\n")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            total = len(members)
            
            if HAS_TQDM:
                iterator = tqdm(members, desc="Estrazione", unit="file")
            else:
                iterator = members
                print(f"⏳ Estrazione di {total} file...")
            
            for i, member in enumerate(iterator):
                # Estrae un file alla volta
                target_path = output_dir / member
                
                if member.endswith('/'):
                    # È una directory
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    # È un file - estrai in streaming
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with zf.open(member) as source:
                        with open(target_path, 'wb') as target:
                            while True:
                                chunk = source.read(chunk_size)
                                if not chunk:
                                    break
                                target.write(chunk)
                
                if not HAS_TQDM and (i + 1) % 1000 == 0:
                    print(f"   Estratti {i + 1}/{total} file...")
        
        print(f"\n✅ Estrazione completata: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ Errore durante l'estrazione: {e}")
        return False


def download_dataset_by_parts(output_dir: Path) -> bool:
    """
    Scarica il dataset file per file invece che tutto insieme.
    Utile quando la memoria è molto limitata.
    
    Args:
        output_dir: Directory di destinazione
        
    Returns:
        True se almeno alcuni file sono stati scaricati
    """
    api = get_kaggle_api()
    if not api:
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Download Tiny GenImage per parti...")
    print(f"📁 Destinazione: {output_dir.absolute()}")
    
    try:
        # Ottieni la lista dei file nel dataset
        print("📋 Recupero lista file...")
        files = api.dataset_list_files(KAGGLE_DATASET).files
        
        print(f"📦 Trovati {len(files)} file nel dataset\n")
        
        downloaded = 0
        failed = 0
        
        for i, file_info in enumerate(files, 1):
            filename = file_info.name
            print(f"[{i}/{len(files)}] {filename}...")
            
            try:
                api.dataset_download_file(
                    KAGGLE_DATASET,
                    filename,
                    path=str(output_dir),
                    force=True,
                    quiet=True
                )
                downloaded += 1
                print(f"   ✅ Scaricato")
                
            except Exception as e:
                failed += 1
                print(f"   ❌ Errore: {e}")
        
        print(f"\n📊 Risultato: {downloaded} scaricati, {failed} falliti")
        return downloaded > 0
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        return False


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
        # Usa il mapping delle cartelle per trovare il generatore
        folder_patterns = GENERATOR_FOLDER_PATTERNS.get(generator, [])
        
        # Costruisci la lista di possibili path
        gen_paths = []
        
        # Prima prova i pattern noti dal mapping
        for pattern in folder_patterns:
            gen_paths.append(root / pattern)
            gen_paths.append(root / pattern.lower())
            gen_paths.append(root / pattern.upper())
        
        # Poi prova anche le variazioni del nome display (per compatibilità)
        gen_paths.extend([
            root / generator,
            root / generator.replace(" ", "_"),
            root / generator.replace(" ", "-"),
            root / generator.lower().replace(" ", "_"),
        ])
        
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

  # Scarica il dataset (modalità standard)
  python download_tiny_genimage.py --output ./data/tiny-genimage

  # ⭐ Scarica con BASSO CONSUMO DI MEMORIA (consigliato se hai poca RAM)
  python download_tiny_genimage.py --low-memory --output ./data/tiny-genimage

  # Scarica senza estrarre (poi estrai con --extract-streaming)
  python download_tiny_genimage.py --low-memory --no-unzip --output ./data/tiny-genimage

  # Estrai un archivio esistente in streaming (basso consumo memoria)
  python download_tiny_genimage.py --extract-streaming ./data/tiny-genimage.zip --output ./data/

  # Verifica un dataset esistente
  python download_tiny_genimage.py --verify ./data/tiny-genimage
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
        help="Estrai un archivio zip esistente (metodo standard)",
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
    
    # Nuove opzioni per basso consumo di memoria
    parser.add_argument(
        "--low-memory", "-l",
        action="store_true",
        default=True,   
        help="⭐ Modalità basso consumo memoria: scarica senza unzip, poi estrai in streaming",
    )
    parser.add_argument(
        "--extract-streaming",
        type=str,
        metavar="ZIP_PATH",
        help="Estrai un archivio zip in streaming (basso consumo memoria)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Dimensione chunk per streaming in bytes (default: 8192 = 8KB)",
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
    
    # Estrazione manuale (standard)
    if args.extract:
        extract_dataset(
            Path(args.extract),
            Path(args.output)
        )
        return 0
    
    # Estrazione streaming (basso consumo memoria)
    if args.extract_streaming:
        success = extract_zip_streaming(
            Path(args.extract_streaming),
            Path(args.output),
            chunk_size=args.chunk_size
        )
        if success:
            verify_dataset(Path(args.output))
        return 0 if success else 1
    
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
    
    # Scegli il metodo di download
    if args.low_memory:
        # Modalità basso consumo di memoria
        print("\n" + "=" * 60)
        print("🔧 MODALITÀ BASSO CONSUMO MEMORIA ATTIVA")
        print("=" * 60)
        print("Questa modalità scarica senza estrarre, poi estrae in streaming.")
        print("Ideale per sistemi con poca RAM.\n")
        
        # Step 1: Scarica senza estrarre
        success = download_dataset_low_memory(output_dir, chunk_size=args.chunk_size)
        
        if success and not args.no_unzip:
            # Step 2: Estrai in streaming
            zip_path = output_dir / "tiny-genimage.zip"
            if zip_path.exists():
                print("\n" + "=" * 60)
                print("📦 ESTRAZIONE STREAMING")
                print("=" * 60)
                success = extract_zip_streaming(zip_path, output_dir, chunk_size=args.chunk_size)
                
                if success:
                    # Opzionale: rimuovi il file zip dopo l'estrazione
                    print(f"\n💡 Puoi rimuovere il file zip per liberare spazio:")
                    print(f"   del \"{zip_path}\"")
            else:
                print(f"\n⚠️  File ZIP non trovato: {zip_path}")
                print("   Il download potrebbe aver salvato con un nome diverso.")
                # Cerca file zip nella directory
                zips = list(output_dir.glob("*.zip"))
                if zips:
                    print(f"   File ZIP trovati: {[z.name for z in zips]}")
                    print(f"   Estrai manualmente con: python {Path(__file__).name} --extract-streaming <file.zip> -o {output_dir}")
    else:
        # Modalità standard (può causare OOM)
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
