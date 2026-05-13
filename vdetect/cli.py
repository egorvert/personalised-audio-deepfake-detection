import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from vdetect.service import DetectionService, ModelType


app = typer.Typer(name="vdetect", help="Audio deepfake detection CLI", add_completion=False)
console = Console()


@app.command()
def detect(
    audio: Path = typer.Argument(..., help="Audio file to analyse"),
    weights: Path = typer.Option("assets/checkpoints/wavlm_baseline.pt", "--weights", "-w"),
    model_type: ModelType = typer.Option(ModelType.wavlm, "--model", "-m"),
    threshold: float = typer.Option(0.5, "--threshold", "-t"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    speaker_id: Optional[str] = typer.Option(
        None, "--speaker-id", help="Speaker ID for personalised detection (fusion only)",
    ),
    db: Path = typer.Option("assets/enrollments/prototypes.json", "--db"),
):
    """Score a single audio file as bonafide or spoof."""
    if not audio.exists():
        console.print(f"[red]Error: Audio file not found: {audio}[/red]")
        raise typer.Exit(1)
    if not weights.exists():
        console.print(f"[red]Error: Checkpoint not found: {weights}[/red]")
        raise typer.Exit(1)
    if speaker_id and model_type != ModelType.fusion:
        console.print("[red]Error: Speaker personalisation requires the fusion model.[/red]")
        raise typer.Exit(1)

    if not json_output:
        if model_type == ModelType.aasist:
            console.print("Loading AASIST model (pretrained)...")
        else:
            console.print(f"Loading {model_type.value} model from {weights}...")

    svc = DetectionService()
    svc.load_model(model_type, weights)

    if not json_output:
        console.print(f"Processing {audio.name}...")

    try:
        result = svc.detect_file(audio, threshold, speaker_id, db)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(asdict(result), indent=2))
        return

    table = Table(title="Detection results", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("File", audio.name)
    table.add_row("Model", model_type.value.upper())
    table.add_row("Score", f"{result.score:.4f}")
    table.add_row("Threshold", f"{threshold:.4f}")
    if speaker_id:
        table.add_row("Speaker ID", speaker_id)
    label_display = (
        f"[green]{result.label}[/green]" if result.label == "bonafide"
        else f"[red]{result.label}[/red]"
    )
    table.add_row("Prediction", label_display)
    table.add_row("Confidence", f"{result.confidence:.2%}")
    console.print(table)

    if result.label == "spoof":
        console.print("\n[red]This audio appears to be spoofed/fake[/red]")
    else:
        console.print("\n[green]This audio appears to be bonafide/genuine[/green]")


@app.command()
def info(weights: Path = typer.Argument(..., help="Checkpoint path")):
    """Show metadata for a trained checkpoint."""
    if not weights.exists():
        console.print(f"[red]Error: Checkpoint not found: {weights}[/red]")
        raise typer.Exit(1)

    result = DetectionService.get_checkpoint_info(weights)

    table = Table(title=f"Checkpoint info: {weights.name}", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    if result.eer is not None:
        table.add_row("Dev EER", f"{result.eer:.4f}")
    if result.threshold is not None:
        table.add_row("Optimal threshold", f"{result.threshold:.4f}")
    if result.epoch is not None:
        table.add_row("Training epoch", str(result.epoch))
    if result.model_name is not None:
        table.add_row("Model", result.model_name)
    if result.lr is not None:
        table.add_row("Learning rate", str(result.lr))
    if result.batch_size is not None:
        table.add_row("Batch size", str(result.batch_size))

    console.print(table)


@app.command()
def batch_detect(
    input_dir: Path = typer.Argument(..., help="Directory of audio files"),
    weights: Path = typer.Option("assets/checkpoints/wavlm_baseline.pt", "--weights", "-w"),
    model_type: ModelType = typer.Option(ModelType.wavlm, "--model", "-m"),
    threshold: float = typer.Option(0.5, "--threshold", "-t"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write results to JSON"),
    extensions: str = typer.Option(".wav,.flac,.mp3", "--extensions", "-e"),
    speaker_id: Optional[str] = typer.Option(None, "--speaker-id"),
    db: Path = typer.Option("assets/enrollments/prototypes.json", "--db"),
):
    """Score every audio file in a directory."""
    if not input_dir.exists():
        console.print(f"[red]Error: Directory not found: {input_dir}[/red]")
        raise typer.Exit(1)
    if not weights.exists():
        console.print(f"[red]Error: Checkpoint not found: {weights}[/red]")
        raise typer.Exit(1)
    if speaker_id and model_type != ModelType.fusion:
        console.print("[red]Error: Speaker personalisation requires the fusion model.[/red]")
        raise typer.Exit(1)

    ext_list = [e.strip() for e in extensions.split(",")]

    svc = DetectionService()
    svc.load_model(model_type, weights)
    console.print(f"Using device: {svc.device}")
    console.print(f"Model: {model_type.value.upper()}")

    with console.status("[bold green]Processing files...") as status:
        def on_progress(i: int, total: int, name: str) -> None:
            status.update(f"[bold green]Processing {i}/{total}: {name}")

        result = svc.batch_detect(
            input_dir,
            threshold=threshold,
            extensions=ext_list,
            speaker_id=speaker_id,
            db_path=db,
            on_progress=on_progress,
        )

    for err in result.errors:
        console.print(f"[yellow]Warning: Failed to process {err}[/yellow]")

    summary = Table(title="Batch detection summary", show_header=True, header_style="bold magenta")
    summary.add_column("Category", style="cyan")
    summary.add_column("Count", style="white", justify="right")
    summary.add_row("Model", model_type.value.upper())
    summary.add_row("Total files", str(result.total))
    summary.add_row("Bonafide", f"[green]{result.bonafide_count}[/green]")
    summary.add_row("Spoof", f"[red]{result.spoof_count}[/red]")
    console.print(summary)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump([asdict(r) for r in result.results], f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def enroll(
    speaker_id: str = typer.Argument(..., help="Speaker ID"),
    audios: List[Path] = typer.Option(..., "--audios", "-a", help="3-5 bonafide samples"),
    weights: Path = typer.Option(
        "assets/checkpoints/two_stream.pt", "--weights", "-w",
        help="Fusion checkpoint used for embedding extraction",
    ),
    db: Path = typer.Option("assets/enrollments/prototypes.json", "--db", "-d"),
    normalize: bool = typer.Option(
        True, "--normalize/--no-normalize", help="L2-normalise the prototype vector",
    ),
):
    """Enrol a speaker by building a prototype embedding from a few samples."""
    if not 3 <= len(audios) <= 5:
        console.print("[red]Error: Provide 3-5 enrolment samples.[/red]")
        raise typer.Exit(1)
    for a in audios:
        if not a.exists():
            console.print(f"[red]Error: Audio file not found: {a}[/red]")
            raise typer.Exit(1)
    if not weights.exists():
        console.print(f"[red]Error: Checkpoint not found: {weights}[/red]")
        raise typer.Exit(1)

    svc = DetectionService()
    console.print(f"Using device: {svc.device or 'auto'}")
    console.print(f"Loading fusion model from {weights}...")
    console.print(f"Extracting embeddings for {len(audios)} samples...")

    result = svc.enroll_speaker(
        speaker_id=speaker_id,
        audio_paths=audios,
        weights=weights,
        db_path=db,
        normalize=normalize,
    )

    console.print(f"[green]{result.action} speaker '{result.speaker_id}' in {result.db_path}[/green]")


if __name__ == "__main__":
    app()
