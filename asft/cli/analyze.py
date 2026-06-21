"""CLI command: asft analyze <config_file>"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from asft.analysis import Analyzer

console = Console()


def analyze_cmd(
    config_file: str = typer.Argument(..., help="Path to the task config JSON file"),
) -> None:
    """Analyze an LLM task and receive a Prompt / RAG / Fine-tuning recommendation."""
    path = Path(config_file)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {config_file}")
        raise typer.Exit(code=1)

    try:
        with open(path, encoding="utf-8") as f:
            task_data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Error parsing JSON:[/bold red] {exc}")
        raise typer.Exit(code=1)  # noqa: B904

    Analyzer.from_config(task_data).recommend()
