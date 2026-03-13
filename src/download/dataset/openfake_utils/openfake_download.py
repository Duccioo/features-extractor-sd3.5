"""
Core download logic per il dataset OpenFake.

Approccio ottimizzato (two-phase):
  Phase 1 – Scansione PARALLELA dei soli metadata (model + label).
             Usa ThreadPoolExecutor per scansionare N shard in contemporanea.
             Legge tutti i metadata di una shard in un colpo solo (non row-group per row-group).
             Costruisce un piano: {shard_idx → {rg_idx → [row_indices]}}.

  Phase 2 – Download mirato: apre solo le shard/row-group che servono.
             Nessun to_pylist(), accesso diretto PyArrow, pulizia immediata.
"""

from __future__ import annotations

import csv
import gc
import random
import time
import io
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Dict, Tuple

try:
    import PIL.Image
    import pyarrow as pa
    import pyarrow.parquet as pq
    import fsspec
    from huggingface_hub import list_repo_files, hf_hub_url, get_token, HfFileSystem
except ImportError:
    print(
        "Errore: pacchetti necessari mancanti. Installa con:\n"
        "  pip install pyarrow fsspec huggingface_hub Pillow",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from .openfake_config import HF_DATASET_ID, sanitize_dirname
except ImportError:
    from openfake_config import HF_DATASET_ID, sanitize_dirname


def _make_progress():
    """Crea il progress bar Rich se disponibile, altrimenti None."""
    try:
        from rich.progress import (
            Progress, BarColumn, TextColumn,
            MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn,
        )
        from rich.console import Console
        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        return progress
    except ImportError:
        return None


# ────────────────────────────────────────────────────────────────
#  Phase 1: Parallel metadata scan
# ────────────────────────────────────────────────────────────────

def _scan_one_shard(
    shard_idx: int,
    shard_file: str,
    storage_options: dict,
    models_filter: Optional[Set[str]],
    labels_filter: Optional[Set[str]],
) -> List[Tuple[int, int, str, str]]:
    """
    Scansiona UNA shard: legge solo model+label in un colpo solo.
    Pre-filtra nel thread per ridurre i dati trasferiti al main thread.
    Ritorna lista di (rg_idx, row_within_rg, model, label) per le righe rilevanti.
    """
    results = []
    try:
        # Usiamo HfFileSystem (più robusto per HF)
        token = storage_options.get("token")
        fs = HfFileSystem(token=token)
        # Il percorso per HfFileSystem è datasets/REPO_ID/path
        hf_path = f"datasets/{HF_DATASET_ID}/{shard_file}"

        # Retry rudimentale per l'apertura
        for attempt in range(2):
            try:
                with fs.open(hf_path) as f:
                    pf = pq.ParquetFile(f)
                    
                    # Leggi TUTTI i metadata in un colpo solo
                    table = pf.read(columns=["model", "label"])
                    col_model = table.column("model")
                    col_label = table.column("label")
                    total_rows = table.num_rows

                    # Calcola gli offset dei row-group per mappare row → rg_idx
                    rg_boundaries = []
                    offset = 0
                    for rg_idx in range(pf.metadata.num_row_groups):
                        n = pf.metadata.row_group(rg_idx).num_rows
                        rg_boundaries.append((offset, offset + n, rg_idx))
                        offset += n

                    # Mappa ogni riga al suo row-group, con pre-filtro
                    rg_iter = iter(rg_boundaries)
                    rg_start, rg_end, rg_idx = next(rg_iter)

                    for i in range(total_rows):
                        while i >= rg_end:
                            rg_start, rg_end, rg_idx = next(rg_iter)

                        model = col_model[i].as_py() or "unknown"
                        label = col_label[i].as_py() or "unknown"

                        # Pre-filtro: scarta subito label e modelli non richiesti
                        if labels_filter and label not in labels_filter:
                            continue
                        if models_filter and label == "fake" and model not in models_filter:
                            continue

                        row_within_rg = i - rg_start
                        results.append((rg_idx, row_within_rg, model, label))

                    del table, col_model, col_label
                    return results # Success

            except Exception:
                if attempt == 1: # Last attempt
                    raise
                time.sleep(1)

    except Exception:
        pass  # Shard fallita, verrà saltata

    return results


def _check_all_filled(
    counts: Dict[str, int],
    completed: set,
    models_filter: Optional[Set[str]],
    labels_filter: Optional[Set[str]],
    limit_per_model: Optional[int],
    eff_limit_real: Optional[int],
    limit: Optional[int],
    total_planned: int,
) -> bool:
    """Controlla se tutti i limiti sono stati raggiunti."""
    if limit and total_planned >= limit:
        return True
    if not limit_per_model and not eff_limit_real:
        return False  # Nessun limite, non possiamo fermarci

    if models_filter and labels_filter:
        needed = set()
        if "fake" in labels_filter:
            needed.update(models_filter)
        if "real" in labels_filter:
            needed.add("__real__")
        return bool(needed) and needed.issubset(completed)
    elif models_filter:
        needed = set(models_filter) | {"__real__"}
        return needed.issubset(completed)

    return False


def _scan_metadata_parallel(
    shards: List[str],
    models_filter: Optional[Set[str]],
    labels_filter: Optional[Set[str]],
    limit: Optional[int],
    limit_per_model: Optional[int],
    limit_real: Optional[int],
    num_workers: int = 16,
    storage_options: Optional[dict] = None,
    progress=None,
    main_task=None,
    quiet: bool = False,
) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[str, int], dict]:
    """
    Scansiona metadata in parallelo con cancellazione immediata.
    
    Invia TUTTI i job in una volta, processa i risultati man mano,
    e CANCELLA i rimanenti appena tutti i limiti sono raggiunti.
    Non resta bloccato da shard lente.

    Returns:
        plan: {shard_idx: {rg_idx: [row_indices_within_rg]}}
        counts: {count_key: num_planned}
        stats_filter: {"skipped_filter": N, "skipped_limit": N}
    """
    eff_limit_real = limit_real if limit_real is not None else limit_per_model

    token = get_token()
    # HfFileSystem usa direttamente il token
    storage_options = {"token": token} if token else {}

    num_shards = len(shards)
    shards_done = 0

    if not quiet:
        print(f"   🚀 Scanning parallelo con {num_workers} thread...")

    # Stato incrementale (aggiornato man mano che i risultati arrivano)
    all_shard_data: Dict[int, List[Tuple[int, int, str, str]]] = {}
    counts: Dict[str, int] = defaultdict(int)
    completed: set = set()
    total_planned = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Invia TUTTI i job in una volta
        future_to_idx = {
            executor.submit(
                _scan_one_shard, idx, shards[idx], storage_options,
                models_filter, labels_filter,
            ): idx
            for idx in range(num_shards)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            shards_done += 1

            if progress and main_task is not None:
                progress.update(
                    main_task,
                    description=f"[yellow]⚡ Scan metadata: {shards_done}/{num_shards} shard"
                )

            try:
                result = future.result()
                if result:
                    all_shard_data[idx] = result

                    # ── Conteggio incrementale ──
                    for rg_idx, row_within_rg, model, label in result:
                        count_key = model if label == "fake" else "__real__"

                        if count_key in completed:
                            continue

                        cur_limit = limit_per_model if label == "fake" else eff_limit_real
                        if cur_limit and cur_limit > 0 and counts[count_key] >= cur_limit:
                            completed.add(count_key)
                            continue

                        counts[count_key] += 1
                        total_planned += 1

            except Exception:
                pass

            # ── Check early stop ──
            all_filled = _check_all_filled(
                counts, completed, models_filter, labels_filter,
                limit_per_model, eff_limit_real, limit, total_planned,
            )
            if all_filled:
                if not quiet:
                    print(f"   ⏩ Limiti raggiunti dopo {shards_done}/{num_shards} shard — cancello il resto!")
                # Cancella tutti i futures non ancora completati
                for f in future_to_idx:
                    f.cancel()
                break

    # ── Costruisci il piano finale ──
    plan, final_counts, _, _, skipped_filter, skipped_limit = _build_plan(
        all_shard_data, models_filter, labels_filter,
        limit, limit_per_model, eff_limit_real,
    )

    del all_shard_data
    gc.collect()

    return plan, final_counts, {"skipped_filter": skipped_filter, "skipped_limit": skipped_limit}


def _build_plan(
    all_shard_data: Dict[int, List[Tuple[int, int, str, str]]],
    models_filter: Optional[Set[str]],
    labels_filter: Optional[Set[str]],
    limit: Optional[int],
    limit_per_model: Optional[int],
    eff_limit_real: Optional[int],
) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[str, int], set, int, int, int]:
    """
    Costruisce il piano di download dai dati scansionati.
    Ritorna (plan, counts, completed, total_planned, skipped_filter, skipped_limit).
    """
    plan: Dict[int, Dict[int, List[int]]] = {}
    counts: Dict[str, int] = defaultdict(int)
    completed: set = set()
    total_planned = 0
    skipped_filter = 0
    skipped_limit = 0

    for shard_idx in sorted(all_shard_data.keys()):
        rows = all_shard_data[shard_idx]

        for rg_idx, row_within_rg, model, label in rows:
            count_key = model if label == "fake" else "__real__"

            if count_key in completed:
                skipped_limit += 1
                continue

            # Labels/models filtering already done in the thread, but
            # we still need to enforce limits
            current_limit = limit_per_model if label == "fake" else eff_limit_real
            if current_limit and current_limit > 0 and counts[count_key] >= current_limit:
                completed.add(count_key)
                skipped_limit += 1
                continue

            if limit and total_planned >= limit:
                break

            if shard_idx not in plan:
                plan[shard_idx] = {}
            if rg_idx not in plan[shard_idx]:
                plan[shard_idx][rg_idx] = []
            plan[shard_idx][rg_idx].append(row_within_rg)

            counts[count_key] += 1
            total_planned += 1

        if limit and total_planned >= limit:
            break

    return plan, dict(counts), completed, total_planned, skipped_filter, skipped_limit


# ────────────────────────────────────────────────────────────────
#  Phase 2: Targeted image download
# ────────────────────────────────────────────────────────────────

def _download_planned(
    shards: List[str],
    plan: Dict[int, Dict[int, List[int]]],
    base_path: Path,
    split_name: str,
    img_format: str,
    save_metadata: bool,
    quiet: bool,
    progress=None,
    main_task=None,
) -> dict:
    """
    Scarica solo le immagini indicate nel piano.
    Apre solo le shard e i row-group necessari.
    """
    stats = {
        "saved": 0,
        "skipped_error": 0,
        "per_model": defaultdict(int),
        "per_label": defaultdict(int),
    }

    ext = "jpg" if img_format.upper() == "JPEG" else "png"

    token = get_token()
    fs = HfFileSystem(token=token)
    # HfFileSystem ha i suoi timeout interni ottimizzati per HF

    # Metadata CSV
    metadata_path = base_path / f"metadata_{split_name}.csv" if save_metadata else None
    csv_file = None
    csv_writer = None
    fieldnames = ["filename", "label", "model", "prompt", "type", "release_date"]
    if metadata_path:
        csv_file = open(metadata_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    file_counters: dict[str, int] = defaultdict(int)

    total_shards_needed = len(plan)
    total_images_planned = sum(
        len(rows) for rg_map in plan.values() for rows in rg_map.values()
    )
    shards_done = 0

    for shard_idx in sorted(plan.keys()):
        shard_file = shards[shard_idx]
        hf_path = f"datasets/{HF_DATASET_ID}/{shard_file}"
        rg_map = plan[shard_idx]
        shards_done += 1

        if progress and main_task is not None:
            progress.update(
                main_task,
                description=(
                    f"[cyan]📥 Download: shard {shards_done}/{total_shards_needed} "
                    f"({stats['saved']}/{total_images_planned} salvati)"
                ),
            )

        max_retries = 3
        retry_delay = 2
        shard_success = False

        for attempt in range(max_retries):
            try:
                with fs.open(hf_path) as f:
                    pf = pq.ParquetFile(f)

                    for rg_idx in sorted(rg_map.keys()):
                        row_indices = rg_map[rg_idx]

                        # Read full row-group (with images) — solo per i RG necessari
                        table = pf.read_row_group(rg_idx)

                        col_model = table.column("model")
                        col_label = table.column("label")
                        col_image = table.column("image")
                        col_prompt = table.column("prompt")
                        col_type = table.column("type")
                        col_date = table.column("release_date")

                        for i in row_indices:
                            model = col_model[i].as_py() or "unknown"
                            label = col_label[i].as_py() or "unknown"
                            count_key = model if label == "fake" else "__real__"

                            try:
                                img_struct = col_image[i].as_py()
                                if not img_struct or not img_struct.get("bytes"):
                                    stats["skipped_error"] += 1
                                    continue

                                raw_bytes = img_struct["bytes"]

                                if label == "fake":
                                    out_dir = base_path / split_name / "fake" / sanitize_dirname(model)
                                else:
                                    out_dir = base_path / split_name / "real"
                                out_dir.mkdir(parents=True, exist_ok=True)

                                idx = file_counters[count_key]
                                file_counters[count_key] += 1
                                img_path = out_dir / f"img_{idx:06d}.{ext}"

                                if img_format.upper() == "JPEG":
                                    with PIL.Image.open(io.BytesIO(raw_bytes)) as pil_img:
                                        if pil_img.mode in ("RGBA", "P"):
                                            pil_img = pil_img.convert("RGB")
                                        pil_img.save(img_path, "JPEG")
                                else:
                                    with open(img_path, "wb") as f_out:
                                        f_out.write(raw_bytes)

                                del raw_bytes, img_struct

                                stats["saved"] += 1
                                stats["per_model"][model] += 1
                                stats["per_label"][label] += 1

                                if csv_writer:
                                    csv_writer.writerow({
                                        "filename": str(img_path.relative_to(base_path)),
                                        "label": label,
                                        "model": model,
                                        "prompt": col_prompt[i].as_py() or "",
                                        "type": col_type[i].as_py() or "",
                                        "release_date": col_date[i].as_py() or "",
                                    })

                            except Exception:
                                stats["skipped_error"] += 1

                        # Cleanup after each row-group
                        del table, col_model, col_label, col_image, col_prompt, col_type, col_date
                        gc.collect()
                        pa.default_memory_pool().release_unused()
                
                shard_success = True
                break  # Success, exit retry loop

            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait = retry_delay * (2 ** attempt)
                    if not quiet:
                        print(f"⚠️ Rate limited su {shard_file}. Retry in {wait}s...")
                    time.sleep(wait)
                else:
                    if not quiet:
                        print(f"⚠️ Errore su shard {shard_file} (tentativo {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                    time.sleep(1)
        
        if not shard_success and not quiet:
            print(f"❌ Impossibile scaricare shard {shard_file} dopo {max_retries} tentativi.")

        # Cleanup after each shard
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

    if progress and main_task is not None:
        progress.update(
            main_task,
            description=f"[green]✅ Completato: {stats['saved']}/{total_images_planned} salvati"
        )

    if csv_file:
        csv_file.close()
        if stats["saved"] > 0:
            print(f"   📋 Metadata salvato: {metadata_path}")
        else:
            metadata_path.unlink(missing_ok=True)

    return dict(stats)


# ────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────

def save_streaming_images(
    shards: List[str],
    base_path: Path,
    split_name: str,
    models_filter: Optional[Set[str]] = None,
    labels_filter: Optional[Set[str]] = None,
    limit: Optional[int] = None,
    limit_per_model: Optional[int] = None,
    limit_real: Optional[int] = None,
    save_metadata: bool = True,
    img_format: str = "PNG",
    quiet: bool = False,
    scan_workers: int = 16,
    seed: int = 42,
    no_shuffle: bool = False,
) -> dict:
    """
    Scarica immagini dal dataset OpenFake con approccio two-phase.

    Phase 1: Scansiona metadata in parallelo (16 thread) — velocissimo.
    Phase 2: Scarica solo i row-group che contengono righe necessarie.

    Lo shuffle dell'ordine delle shard è attivo di default quando ci sono limiti,
    per permettere all'early termination di scattare prima (campionando da tutto il dataset).
    """
    # Shuffle delle shard per trovare tutti i modelli prima
    has_limits = limit is not None or limit_per_model is not None
    if has_limits and not no_shuffle:
        shards = list(shards)  # copia per non modificare l'originale
        random.Random(seed).shuffle(shards)
        if not quiet:
            print(f"   🔀 Shard shuffled (seed={seed}) per early termination più veloce")
    
    token = get_token()
    storage_options = {"token": token} if token else {}
    
    # Usiamo un wrapper per iniettare le opzioni nella Phase 1
    scan_storage_options = storage_options.copy()
    
    progress = _make_progress() if not quiet else None
    main_task = None
    if progress:
        main_task = progress.add_task(f"[cyan]Scansione {split_name}", total=None)
        progress.start()

    # ── Phase 1: Parallel metadata scan ──
    if not quiet:
        print(f"\n   ⚡ Fase 1: Scansione metadata ({len(shards)} shard, {scan_workers} thread paralleli)...")

    plan, counts, filter_stats = _scan_metadata_parallel(
        shards=shards,
        models_filter=models_filter,
        labels_filter=labels_filter,
        limit=limit,
        limit_per_model=limit_per_model,
        limit_real=limit_real,
        num_workers=scan_workers,
        storage_options=scan_storage_options, # Passiamo le opzioni
        progress=progress,
        main_task=main_task,
        quiet=quiet,
    )

    total_planned = sum(len(rows) for rg_map in plan.values() for rows in rg_map.values())
    shards_needed = len(plan)

    if not quiet:
        print(f"   ✅ Piano: {total_planned:,} immagini in {shards_needed} shard "
              f"(su {len(shards)} totali)")
        if counts:
            for key, n in sorted(counts.items()):
                label = key if key != "__real__" else "real"
                print(f"      - {label}: {n}")

    if total_planned == 0:
        if progress:
            progress.stop()
        return {
            "saved": 0,
            "skipped_filter": filter_stats["skipped_filter"],
            "skipped_limit": filter_stats["skipped_limit"],
            "skipped_error": 0,
            "per_model": {},
            "per_label": {},
        }

    # ── Phase 2: Targeted download ──
    if not quiet:
        print(f"\n   📥 Fase 2: Download mirato da {shards_needed} shard...")

    dl_stats = _download_planned(
        shards=shards,
        plan=plan,
        base_path=base_path,
        split_name=split_name,
        img_format=img_format,
        save_metadata=save_metadata,
        quiet=quiet,
        progress=progress,
        main_task=main_task,
    )

    if progress:
        progress.stop()

    return {
        "saved": dl_stats["saved"],
        "skipped_filter": filter_stats["skipped_filter"],
        "skipped_limit": filter_stats["skipped_limit"],
        "skipped_error": dl_stats["skipped_error"],
        "per_model": dl_stats["per_model"],
        "per_label": dl_stats["per_label"],
    }


def download_openfake(
    output_dir: Path,
    splits: List[str] = None,
    models_filter=None,
    labels_filter=None,
    limit=None,
    limit_per_model=None,
    limit_real=None,
    seed: int = 42,
    img_format: str = "PNG",
    quiet: bool = False,
    no_shuffle: bool = True,
) -> dict:
    """
    Scarica il dataset OpenFake con streaming e filtraggio.
    """
    if splits is None:
        splits = ["train", "test"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"\n{'='*60}")
        print("📥 DOWNLOAD OPENFAKE DATASET")
        print(f"{'='*60}")
        print(f"   📁 Destinazione:     {output_dir.absolute()}")
        print(f"   📂 Splits:           {', '.join(splits)}")
        print(f"   🏷️  Label filter:     {labels_filter or 'tutti'}")
        print(f"   🤖 Model filter:     {models_filter or 'tutti'}")
        print(f"   📊 Limite globale:   {limit or 'nessuno'}")
        print(f"   📊 Limite/modello:   {limit_per_model or 'nessuno'}")
        
        real_limit_str = 'Tutte (massimo)' if limit_real == -1 else (limit_real or 'come limite/modello')
        print(f"   📊 Limite real:      {real_limit_str}")
        print(f"   🖼️  Formato:          {img_format}")
        print()

    all_stats = {}

    for split in splits:
        if not quiet:
            print(f"\n--- Split: {split} ---")

        raw_models = models_filter[split] if isinstance(models_filter, dict) else models_filter
        raw_labels = labels_filter[split] if isinstance(labels_filter, dict) else labels_filter

        s_models = set(raw_models) if raw_models else None
        s_labels = set(raw_labels) if raw_labels else None
        s_limit = limit[split] if isinstance(limit, dict) else limit
        s_limit_per_model = limit_per_model[split] if isinstance(limit_per_model, dict) else limit_per_model
        s_limit_real = limit_real[split] if isinstance(limit_real, dict) else limit_real

        if not quiet:
            print(f"🔍 Ricerca file Parquet per lo split '{split}'...")

        try:
            all_files = list_repo_files(HF_DATASET_ID, repo_type="dataset")
            split_shards = sorted([
                f for f in all_files
                if f.startswith(f"data/{split}-") and f.endswith(".parquet")
            ])

            if not split_shards:
                print(f"⚠️  Nessuna shard trovata per '{split}'.")
                continue

            if not quiet:
                print(f"   Trovate {len(split_shards)} shards.")
        except Exception as e:
            print(f"❌ Errore nel recupero lista shard: {e}")
            continue

        stats = save_streaming_images(
            shards=split_shards,
            base_path=output_dir,
            split_name=split,
            models_filter=s_models,
            labels_filter=s_labels,
            limit=s_limit,
            limit_per_model=s_limit_per_model,
            limit_real=s_limit_real,
            img_format=img_format,
            quiet=quiet,
            seed=seed,
            no_shuffle=no_shuffle,
        )

        all_stats[split] = stats

        if not quiet:
            print(f"\n   ✅ Split '{split}' completato:")
            print(f"      Salvate:          {stats['saved']:,}")
            print(f"      Filtrate:         {stats['skipped_filter']:,}")
            print(f"      Limite raggiunto: {stats['skipped_limit']:,}")
            print(f"      Errori:           {stats['skipped_error']:,}")
            if stats["per_model"]:
                print(f"      Modelli:")
                for model, count in sorted(stats["per_model"].items()):
                    print(f"         - {model}: {count:,}")

    if not quiet:
        total_saved = sum(s["saved"] for s in all_stats.values())
        print(f"\n{'='*60}")
        print("📊 RIEPILOGO DOWNLOAD")
        print(f"{'='*60}")
        print(f"   Immagini totali salvate: {total_saved:,}")
        print(f"   Directory: {output_dir.absolute()}")
        print(f"{'='*60}\n")

    return all_stats
