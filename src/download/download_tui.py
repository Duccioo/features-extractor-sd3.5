#!/usr/bin/env python3
"""
Unified TUI for downloading SD3.5 models and datasets.

This script provides an interactive Text User Interface (TUI) to:
- Download SD3.5 Models from HuggingFace
- Download Tiny GenImage dataset from Kaggle
- Configure credentials for both platforms
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    import questionary
except ImportError:
    print(
        "Errore: mancano pacchetti necessari. Installa con: pip install rich questionary",
        file=sys.stderr,
    )
    sys.exit(1)

console = Console()

# Import download functions from existing scripts
# These are imported lazily to avoid loading unnecessary dependencies
SCRIPT_DIR = Path(__file__).resolve().parent


def download_models_interactive():
    """Launch model download TUI."""
    # Import and run the model downloader
    sys.path.insert(0, str(SCRIPT_DIR))
    from download_models import interactive_mode as model_tui, download_model, download_controlnets, postprocess, ensure_token, MODELS
    
    args = model_tui()
    ensure_token(args.token)
    dest = Path(args.dest).expanduser().resolve()
    
    console.print(f"\n[bold blue]📁 Destinazione:[/bold blue] {dest}")
    console.print(f"[bold blue]📦 Modelli:[/bold blue] {', '.join(args.model)}")
    
    for model_key in args.model:
        download_model(
            model_key=model_key,
            dest=dest,
            overwrite=args.overwrite,
            skip_encoders=args.skip_encoders,
            only_encoders=args.only_encoders,
        )
    
    if args.controlnets:
        download_controlnets(args.controlnets, dest, args.overwrite)
    
    if not args.no_postprocess:
        console.print("\n[bold]📋 Postprocessing...[/bold]")
        postprocess(dest)
    
    console.print(f"\n[green]✅ Download modelli completato![/green] File in: {dest}")


def download_tiny_genimage_interactive():
    """Launch Tiny GenImage dataset download TUI."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from download_tiny_genimage import (
        check_kaggle_cli, check_kaggle_credentials, install_kaggle_cli,
        setup_kaggle_credentials, download_dataset, verify_dataset, print_info,
        KAGGLE_DATASET, DATASET_SIZE
    )
    
    console.print(Panel.fit("📊 [bold green]Tiny GenImage Dataset Downloader[/bold green]", border_style="green"))
    console.print(f"[dim]Dataset: {KAGGLE_DATASET}[/dim]")
    console.print(f"[dim]Dimensione: {DATASET_SIZE}[/dim]\n")
    
    # Check Kaggle CLI
    if not check_kaggle_cli():
        console.print("[yellow]⚠️ Kaggle CLI non trovato.[/yellow]")
        if questionary.confirm("Vuoi installare kaggle CLI?").ask():
            install_kaggle_cli()
        else:
            console.print("[red]❌ Kaggle CLI richiesto per il download.[/red]")
            return
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        console.print("[yellow]⚠️ Credenziali Kaggle non configurate.[/yellow]")
        setup_kaggle_credentials()
        
        # Ask user to configure and retry
        if not questionary.confirm("Hai configurato le credenziali? Vuoi riprovare?").ask():
            return
        
        if not check_kaggle_credentials():
            console.print("[red]❌ Credenziali ancora mancanti.[/red]")
            return
    
    console.print("[green]✅ Credenziali Kaggle OK[/green]\n")
    
    # Destination folder
    dest_folder = questionary.path(
        "Cartella di destinazione:",
        default="data/tiny-genimage",
        only_directories=True
    ).ask()
    
    if not dest_folder:
        console.print("[red]❌ Nessuna cartella selezionata.[/red]")
        return
    
    output_dir = Path(dest_folder).expanduser().resolve()
    
    # Options
    unzip = questionary.confirm("Estrarre automaticamente il dataset dopo il download?", default=True).ask()
    
    # Check if exists
    if output_dir.exists() and any(output_dir.iterdir()):
        if not questionary.confirm(f"La directory {output_dir} non è vuota. Continuare?").ask():
            console.print("[yellow]Download annullato.[/yellow]")
            return
    
    # Download
    success = download_dataset(output_dir, unzip=unzip)
    
    if success:
        console.print("\n[bold]🔍 Verifica dataset...[/bold]")
        verify_dataset(output_dir)
        console.print(f"\n[green]✅ Download Tiny GenImage completato![/green] File in: {output_dir}")


def main_menu():
    """Main TUI menu."""
    console.print(Panel.fit(
        "🚀 [bold cyan]Download Manager[/bold cyan]\n"
        "[dim]Stable Diffusion 3.5 & Datasets[/dim]",
        border_style="cyan"
    ))
    
    choices = [
        questionary.Choice("📦 Scarica Modelli SD3.5 (HuggingFace)", value="models"),
        questionary.Choice("📊 Scarica Tiny GenImage Dataset (Kaggle)", value="tiny_genimage"),
        questionary.Choice("❌ Esci", value="exit"),
    ]
    
    action = questionary.select(
        "Cosa vuoi scaricare?",
        choices=choices
    ).ask()
    
    if action == "models":
        download_models_interactive()
    elif action == "tiny_genimage":
        download_tiny_genimage_interactive()
    elif action == "exit" or action is None:
        console.print("[dim]Arrivederci![/dim]")
        return 0
    
    # Ask if user wants to do more
    if questionary.confirm("\nVuoi scaricare altro?").ask():
        return main_menu()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified TUI for downloading SD3.5 models and datasets",
    )
    parser.add_argument(
        "--models", "-m",
        action="store_true",
        help="Avvia direttamente il download modelli",
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["tiny_genimage"],
        help="Avvia direttamente il download di un dataset",
    )
    
    args = parser.parse_args()
    
    if args.models:
        download_models_interactive()
        return 0
    elif args.dataset == "tiny_genimage":
        download_tiny_genimage_interactive()
        return 0
    else:
        return main_menu()


if __name__ == "__main__":
    sys.exit(main())
