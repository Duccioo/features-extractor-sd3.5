"""
TUI interattiva per il download del dataset OpenFake.

Guida l'utente passo passo nella configurazione del download.
Richiede: pip install rich questionary
"""

from __future__ import annotations

from pathlib import Path

try:
    from .openfake_config import (
        HF_DATASET_ID, DATASET_SIZE, PAPER_URL,
        TRAIN_EXAMPLES, TEST_EXAMPLES, KNOWN_MODELS,
    )
    from .openfake_download import download_openfake
    from .openfake_verify import verify_dataset
except ImportError:
    from openfake_config import (
        HF_DATASET_ID, DATASET_SIZE, PAPER_URL,
        TRAIN_EXAMPLES, TEST_EXAMPLES, KNOWN_MODELS,
    )
    from openfake_download import download_openfake
    from openfake_verify import verify_dataset


def check_tui_deps() -> bool:
    """Verifica che le dipendenze TUI siano disponibili."""
    try:
        import rich  # noqa: F401
        import questionary  # noqa: F401
        return True
    except ImportError:
        return False


def interactive_mode():
    """
    TUI interattiva per configurare e lanciare il download di OpenFake.

    Guida l'utente passo passo nella scelta di:
    - Split da scaricare
    - Modelli da includere
    - Label da filtrare
    - Limiti di download
    - Directory di output
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    import questionary
    from questionary import Style as QStyle

    console = Console()

    # Stile custom per questionary
    custom_style = QStyle([
        ("qmark", "fg:cyan bold"),
        ("question", "fg:white bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:gray"),
        ("instruction", "fg:gray italic"),
    ])

    # ── Header ─────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔍 OpenFake Dataset Downloader[/bold cyan]\n"
        "[dim]Deepfake Detection & Attribution Dataset[/dim]\n"
        f"[dim]{HF_DATASET_ID}[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # ── Info Table ─────────────────────────────────────────────
    info_table = Table(
        show_header=False, box=box.SIMPLE,
        padding=(0, 2), show_edge=False,
    )
    info_table.add_column("Key", style="bold yellow")
    info_table.add_column("Value", style="white")
    info_table.add_row("📦 Dimensione totale", DATASET_SIZE)
    info_table.add_row("🏋️  Train", f"{TRAIN_EXAMPLES:,} immagini")
    info_table.add_row("🧪 Test", f"{TEST_EXAMPLES:,} immagini")
    info_table.add_row("🤖 Modelli", f"{len(KNOWN_MODELS)}+ generatori")
    info_table.add_row("📄 Paper", PAPER_URL)
    console.print(info_table)
    console.print()

    # ── Step 1: Seleziona split ────────────────────────────────
    console.print("[bold cyan]━━━ Step 1/6: Split ━━━[/bold cyan]")
    split_choice = questionary.select(
        "Quali split vuoi scaricare?",
        choices=[
            questionary.Choice("🏋️  Solo Train (1.87M immagini)", value="train"),
            questionary.Choice("🧪 Solo Test (59.6K immagini)", value="test"),
            questionary.Choice("📦 Entrambi (Train + Test)", value="both"),
        ],
        style=custom_style,
    ).ask()

    if split_choice is None:
        console.print("[red]❌ Operazione annullata.[/red]")
        return None

    splits = ["train", "test"] if split_choice == "both" else [split_choice]
    console.print()

    # ── Step 2-4: Configurazione ───────────────────────────────
    split_configs = {}

    separate_config = False
    if len(splits) > 1:
        separate_config = questionary.confirm(
            "Vuoi configurare ogni split separatamente?",
            default=False,
            style=custom_style,
        ).ask()
        if separate_config is None:
            console.print("[red]❌ Operazione annullata.[/red]")
            return None

    if not separate_config:
        config = _configure_split(console, custom_style, questionary, "Globale")
        if config is None:
            return None
        for s in splits:
            split_configs[s] = config
    else:
        for s in splits:
            config = _configure_split(console, custom_style, questionary, s.upper())
            if config is None:
                return None
            split_configs[s] = config

    # ── Step 5: Opzioni e Output ────────────────────────────
    console.print("\n[bold cyan]━━━ Step 5/6: Opzioni e Output ━━━[/bold cyan]")

    use_shuffle = questionary.confirm(
        "Vuoi attivare lo shuffle?",
        default=False,
        style=custom_style,
    ).ask()
    if use_shuffle is None:
        console.print("[red]❌ Operazione annullata.[/red]")
        return None

    if not use_shuffle:
        console.print("   [yellow]⚠️  Shuffle disattivato. Le immagini verranno scaricate nell'ordine originale (più veloce).[/yellow]")

    img_format = questionary.select(
        "Formato immagini:",
        choices=[
            questionary.Choice("🖼️  PNG (lossless, file più grandi)", value="PNG"),
            questionary.Choice("📷 JPEG (lossy, file più piccoli)", value="JPEG"),
        ],
        style=custom_style,
    ).ask()
    if img_format is None:
        console.print("[red]❌ Operazione annullata.[/red]")
        return None

    dest_folder = questionary.text(
        "Directory di destinazione:",
        default="data/OpenFake",
        style=custom_style,
    ).ask()
    if dest_folder is None:
        console.print("[red]❌ Operazione annullata.[/red]")
        return None

    output_dir = Path(dest_folder).expanduser().resolve()
    console.print()

    # ── Step 6: Riepilogo e conferma ───────────────────────────
    console.print("[bold cyan]━━━ Step 6/6: Conferma ━━━[/bold cyan]")

    summary_table = Table(
        title="📋 Riepilogo Configurazione",
        box=box.ROUNDED, title_style="bold green",
        border_style="green", padding=(0, 1),
    )
    summary_table.add_column("Parametro", style="bold yellow")
    summary_table.add_column("Valore", style="white")
    summary_table.add_row("Splits", ", ".join(splits))

    if not separate_config:
        conf = split_configs[splits[0]]
        summary_table.add_row("Labels", ", ".join(conf["labels"]) if conf["labels"] else "tutti")
        summary_table.add_row("Modelli", ", ".join(conf["models"]) if conf["models"] else "tutti (20+)")
        summary_table.add_row("Limite globale", str(conf["limit"]) if conf["limit"] else "illimitato")
        summary_table.add_row("Limite/modello", str(conf["limit_per_model"]) if conf["limit_per_model"] else "illimitato")
        summary_table.add_row("Limite REAL", str(conf["limit_real"]) if conf["limit_real"] is not None else "come limite/modello")
    else:
        for s in splits:
            conf = split_configs[s]
            summary_table.add_row(f"[bold]{s.upper()}[/bold]", "")
            summary_table.add_row("  Labels", ", ".join(conf["labels"]) if conf["labels"] else "tutti")
            summary_table.add_row("  Modelli", f"{len(conf['models'])} selezionati" if conf["models"] else "tutti")
            summary_table.add_row("  Limite glob.", str(conf["limit"]) if conf["limit"] else "illimitato")
            summary_table.add_row("  Limite/mod.", str(conf["limit_per_model"]) if conf["limit_per_model"] else "illimitato")
            summary_table.add_row("  Limite REAL", str(conf["limit_real"]) if conf["limit_real"] is not None else "come limite/modello")

    summary_table.add_row("Shuffle", "Sì" if use_shuffle else "No")
    summary_table.add_row("Formato", img_format)
    summary_table.add_row("Destinazione", str(output_dir))

    # Stima immagini
    est_images = 0
    for s in splits:
        conf = split_configs[s]
        if conf["limit"]:
            est_images += conf["limit"]
        else:
            n_models = len(conf["models"]) if conf["models"] else len(KNOWN_MODELS)
            est_fake = (conf["limit_per_model"] or (TRAIN_EXAMPLES if s == "train" else TEST_EXAMPLES)) * n_models
            if conf["limit_real"] is not None:
                est_real = conf["limit_real"]
            else:
                est_real = conf["limit_per_model"] or (TRAIN_EXAMPLES if s == "train" else TEST_EXAMPLES)
            est_images += (est_fake + est_real)

    if est_images < 100_000:
        est_size = f"~{est_images * 0.3:.0f} MB"
    else:
        est_size = f"~{est_images * 0.3 / 1024:.1f} GB"

    summary_table.add_row("Stima immagini", f"~{est_images:,}")
    summary_table.add_row("Stima dimensione", est_size)

    console.print(summary_table)
    console.print()

    if est_images > 100_000:
        console.print(
            "[bold yellow]⚠️  Attenzione: stai per scaricare molte immagini. "
            "Considera di usare limiti più restrittivi.[/bold yellow]\n"
        )

    confirm = questionary.confirm(
        "Procedere con il download?",
        default=True,
        style=custom_style,
    ).ask()

    if not confirm:
        console.print("[yellow]❌ Download annullato.[/yellow]")
        return None

    # ── Lancio download ────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold green]🚀 Avvio download...[/bold green]",
        border_style="green",
    ))
    console.print()

    stats = download_openfake(
        output_dir=output_dir,
        splits=splits,
        models_filter={s: split_configs[s]["models"] for s in splits} if separate_config else split_configs[splits[0]]["models"],
        labels_filter={s: split_configs[s]["labels"] for s in splits} if separate_config else split_configs[splits[0]]["labels"],
        limit={s: split_configs[s]["limit"] for s in splits} if separate_config else split_configs[splits[0]]["limit"],
        limit_per_model={s: split_configs[s]["limit_per_model"] for s in splits} if separate_config else split_configs[splits[0]]["limit_per_model"],
        limit_real={s: split_configs[s]["limit_real"] for s in splits} if separate_config else split_configs[splits[0]]["limit_real"],
        img_format=img_format,
        no_shuffle=not use_shuffle,
    )

    # ── Risultato ──────────────────────────────────────────────
    total_saved = sum(s["saved"] for s in stats.values())
    console.print(Panel.fit(
        f"[bold green]✅ Download completato![/bold green]\n"
        f"[white]Immagini salvate: {total_saved:,}[/white]\n"
        f"[dim]{output_dir}[/dim]",
        border_style="green",
        padding=(1, 3),
    ))

    # Chiedi se verificare
    if questionary.confirm(
        "Vuoi verificare il dataset scaricato?",
        default=True,
        style=custom_style,
    ).ask():
        verify_dataset(output_dir)

    return stats


def _configure_split(console, custom_style, questionary, label: str) -> dict | None:
    """Configura un singolo split (o globale). Ritorna None se annullato."""
    console.print(f"\n[bold cyan]━━━ Configurazione {label} ━━━[/bold cyan]")

    # Label Filter
    label_choice = questionary.select(
        f"[{label}] Quali label vuoi includere?",
        choices=[
            questionary.Choice("📸 Solo Real", value="real"),
            questionary.Choice("🤖 Solo Fake", value="fake"),
            questionary.Choice("🔄 Entrambi (Real + Fake)", value="both"),
        ],
        style=custom_style,
    ).ask()
    if label_choice is None:
        return None
    l_filter = None if label_choice == "both" else [label_choice]

    # Model Filter
    m_filter = None
    if label_choice != "real":
        model_mode = questionary.select(
            f"[{label}] Vuoi filtrare per modello generativo?",
            choices=[
                questionary.Choice("🌐 Tutti i modelli (20+ generatori)", value="all"),
                questionary.Choice("🎯 Seleziona modelli specifici", value="select"),
            ],
            style=custom_style,
        ).ask()
        if model_mode == "select":
            m_filter = questionary.checkbox(
                f"[{label}] Seleziona i modelli da includere:",
                choices=[questionary.Choice(m, checked=False) for m in KNOWN_MODELS],
                style=custom_style,
            ).ask()
            if m_filter is None:
                return None

    # Limits
    console.print("   [dim]Imposta limiti (lascia vuoto per illimitato):[/dim]")
    lim_str = questionary.text(f"[{label}] Limite globale per split:", style=custom_style).ask()
    if lim_str is None:
        return None
    lim = int(lim_str) if lim_str.strip() else None

    lim_mod_str = questionary.text(f"[{label}] Limite per singolo modello:", style=custom_style).ask()
    if lim_mod_str is None:
        return None
    lim_mod = int(lim_mod_str) if lim_mod_str.strip() else None

    # Limite REAL specifico
    lim_real = None
    if l_filter is None or "real" in l_filter:
        real_mode = questionary.select(
            f"[{label}] Quante immagini REAL vuoi scaricare?",
            choices=[
                questionary.Choice("Uguale al limite per modello", value="same"),
                questionary.Choice("Bilanciato (somma di tutti i modelli fake)", value="balanced"),
                questionary.Choice("Numero personalizzato", value="custom"),
            ],
            style=custom_style,
        ).ask()
        if real_mode is None:
            return None

        if real_mode == "same":
            lim_real = lim_mod
        elif real_mode == "balanced":
            n_models = len(m_filter) if m_filter else len(KNOWN_MODELS)
            lim_real = (lim_mod or 1) * n_models
        elif real_mode == "custom":
            custom_real_str = questionary.text(f"[{label}] Numero di immagini REAL:", style=custom_style).ask()
            if custom_real_str is None:
                return None
            lim_real = int(custom_real_str) if custom_real_str.strip() else None

    return {
        "labels": l_filter,
        "models": m_filter,
        "limit": lim,
        "limit_per_model": lim_mod,
        "limit_real": lim_real,
    }
