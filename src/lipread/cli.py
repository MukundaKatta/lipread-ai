"""CLI entry point for lipread-ai."""

from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def cli():
    """LipRead-AI - Video lip reading and visual speech recognition."""
    pass


@cli.command()
@click.argument("text")
@click.option("--duration", "-d", default=2000.0, help="Simulation duration in ms")
def simulate(text: str, duration: float):
    """Simulate lip reading pipeline on synthetic data."""
    from lipread.simulator import LipReadingSimulator
    from lipread.report import render_result

    sim = LipReadingSimulator()
    result = sim.simulate_pipeline(text, duration)
    render_result(result, console)


@cli.command()
@click.argument("text")
def visemes(text: str):
    """Show viseme sequence for input text."""
    from lipread.recognizer.vocabulary import VisemeVocabulary
    from lipread.simulator import LipReadingSimulator

    sim = LipReadingSimulator()
    vocab = VisemeVocabulary()
    sequence = sim.generate_viseme_sequence(text)

    console.print(f"[bold]Text:[/bold] {text}")
    for vid in sequence:
        v = vocab.get_viseme(vid)
        if v:
            console.print(f"  {vid}: {v.label} - {v.description} ({v.mouth_shape})")


@cli.command()
def vocab():
    """Display the full viseme vocabulary."""
    from lipread.recognizer.vocabulary import VisemeVocabulary
    from rich.table import Table

    vocabulary = VisemeVocabulary()
    table = Table(title=f"Viseme Vocabulary ({vocabulary.size()} visemes)")
    table.add_column("ID", style="cyan")
    table.add_column("Label", style="bold")
    table.add_column("Shape")
    table.add_column("Phonemes", style="dim")
    table.add_column("Description")

    for v in vocabulary.visemes:
        table.add_row(v.id, v.label, v.mouth_shape, ", ".join(v.phonemes), v.description)
    console.print(table)


if __name__ == "__main__":
    cli()
