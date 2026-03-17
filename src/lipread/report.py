"""Rich console report rendering for lip reading results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lipread.models import TranscriptionResult


def render_result(result: TranscriptionResult, console: Console | None = None) -> None:
    """Render a transcription result to the console."""
    console = console or Console()

    console.print(Panel(
        f"[bold]Lip Reading Transcription[/bold]\n"
        f"Frames: {result.frame_count} | Duration: {result.duration_ms:.0f}ms\n"
        f"Confidence: {result.confidence:.1%}",
        title="LipRead-AI",
    ))

    console.print(f"\n[bold]Transcription:[/bold] {result.text}")

    if result.viseme_sequence:
        console.print(f"[dim]Visemes: {' -> '.join(result.viseme_sequence[:20])}{'...' if len(result.viseme_sequence) > 20 else ''}[/dim]")

    if result.words:
        table = Table(title="Word Breakdown")
        table.add_column("Word", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Visemes", style="dim")
        for w in result.words:
            table.add_row(
                w.word,
                f"{w.confidence:.1%}",
                " ".join(w.visemes[:5]),
            )
        console.print(table)
