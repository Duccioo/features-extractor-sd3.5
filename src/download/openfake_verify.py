"""
Funzioni di verifica e scoperta per il dataset OpenFake.

- list_available_models(): scansiona i modelli disponibili via PyArrow
- verify_dataset(): verifica un dataset già scaricato
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Set

try:
    from huggingface_hub import list_repo_files, hf_hub_url, get_token
    import pyarrow.parquet as pq
    import fsspec
except ImportError:
    print(
        "Errore: pacchetti necessari mancanti. Installa con:\n"
        "  pip install pyarrow fsspec huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from .openfake_config import HF_DATASET_ID
except ImportError:
    from openfake_config import HF_DATASET_ID


def list_available_models(split: str = "train", max_scan: int = 50_000) -> List[str]:
    """
    Elenca i modelli disponibili nel dataset usando scansione low-level pyarrow.
    Legge solo le colonne metadata (model, label, type) senza toccare le immagini.
    """
    print(f"🔍 Scansione modelli disponibili nello split '{split}'...")
    print(f"   (scansione metadata fino a {max_scan:,} esempi con pyarrow)\n")

    try:
        all_files = list_repo_files(HF_DATASET_ID, repo_type="dataset")
        shards = sorted([
            f for f in all_files
            if f.startswith(f"data/{split}-") and f.endswith(".parquet")
        ])
        if not shards:
            return []

        token = get_token()
        storage_options = {"headers": {"Authorization": f"Bearer {token}"}} if token else {}
        fs = fsspec.filesystem("https", **storage_options)

        models: Set[str] = set()
        labels: Set[str] = set()
        types: Set[str] = set()
        scanned = 0

        for shard_file in shards:
            url = hf_hub_url(HF_DATASET_ID, filename=shard_file, repo_type="dataset")
            with fs.open(url) as f:
                pf = pq.ParquetFile(f)
                table = pf.read(columns=["model", "label", "type"])
                n = table.num_rows
                col_model = table.column("model")
                col_label = table.column("label")
                col_type = table.column("type")
                for i in range(n):
                    scanned += 1
                    m = col_model[i].as_py()
                    l = col_label[i].as_py()
                    t = col_type[i].as_py()
                    if m:
                        models.add(m)
                    if l:
                        labels.add(l)
                    if t:
                        types.add(t)
                    if scanned >= max_scan:
                        break
                del table, col_model, col_label, col_type
            if scanned >= max_scan:
                break

        print(f"\n✅ Trovati {len(models)} modelli unici in {scanned:,} esempi")
        return sorted(list(models))
    except Exception as e:
        print(f"⚠️ Errore durante la scansione modelli: {e}")
        return []


def verify_dataset(data_dir: Path) -> dict:
    """
    Verifica il dataset OpenFake scaricato.
    Conta immagini per split/label/model e stampa un riepilogo.

    Args:
        data_dir: Directory del dataset

    Returns:
        dict con statistiche
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"❌ Directory non trovata: {data_dir}")
        return {}

    stats = {
        "splits": {},
        "total_images": 0,
        "total_size_bytes": 0,
        "metadata_files": [],
    }

    print(f"\n🔍 Verifica dataset: {data_dir.absolute()}")
    print(f"{'='*60}")

    # Cerca metadata CSV
    for csv_file in data_dir.glob("metadata_*.csv"):
        stats["metadata_files"].append(str(csv_file))
        with open(csv_file, "r") as f:
            n_rows = sum(1 for _ in f) - 1
        print(f"   📋 {csv_file.name}: {n_rows:,} righe")

    # Scansiona struttura directory
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        split_stats = {"labels": {}, "total": 0}

        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            label_name = label_dir.name

            if label_name == "fake":
                model_counts = {}
                for model_dir in sorted(label_dir.iterdir()):
                    if model_dir.is_dir():
                        img_count = sum(
                            1 for f in model_dir.iterdir()
                            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
                        )
                        model_counts[model_dir.name] = img_count
                        split_stats["total"] += img_count
                        stats["total_images"] += img_count

                split_stats["labels"]["fake"] = model_counts
                total_fake = sum(model_counts.values())
                print(f"\n   📂 {split_name}/fake/ ({total_fake:,} immagini)")
                for model, count in sorted(model_counts.items()):
                    print(f"      └── {model}: {count:,}")

            else:
                img_count = sum(
                    1 for f in label_dir.iterdir()
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg")
                )
                split_stats["labels"][label_name] = img_count
                split_stats["total"] += img_count
                stats["total_images"] += img_count
                print(f"\n   📂 {split_name}/{label_name}/ ({img_count:,} immagini)")

        stats["splits"][split_name] = split_stats

    # Dimensione totale
    total_bytes = sum(
        f.stat().st_size for f in data_dir.rglob("*") if f.is_file()
    )
    stats["total_size_bytes"] = total_bytes

    print(f"\n{'='*60}")
    print(f"   📊 Totale immagini: {stats['total_images']:,}")
    print(f"   💾 Dimensione:      {total_bytes / (1024**2):.1f} MB")
    print(f"{'='*60}\n")

    return stats
