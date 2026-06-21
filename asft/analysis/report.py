"""Report formatting for ASFT Decision Reports."""
from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from asft.analysis.evaluator import PromptEvaluationResult
from asft.analysis.finetune_estimator import FinetuneEstimateResult
from asft.analysis.rag_analyzer import RAGEvaluationResult
from asft.analysis.recommender import RecommendationResult

console = Console()

_METHOD_COLOR = {
    "RAG": "green",
    "PROMPTING": "cyan",
    "FINE-TUNING": "magenta",
}

_DIVIDER = "================================"
_THIN_DIV = "--------------------------------"


@dataclass
class DecisionReportData:
    """All data needed to render an ASFT Decision Report."""

    task_name: str
    prompt: PromptEvaluationResult
    rag: RAGEvaluationResult
    ft: FinetuneEstimateResult
    recommendation: RecommendationResult


def print_report(data: DecisionReportData) -> None:
    """Render a formatted ASFT Decision Report to the terminal."""
    color = _METHOD_COLOR.get(data.recommendation.method, "white")

    console.print()
    console.print(_DIVIDER)
    console.print("[bold cyan]ASFT Decision Report[/bold cyan]")
    console.print(_DIVIDER)
    console.print()

    console.print(f"[bold]Task:[/bold]  {data.task_name}")
    console.print()

    # ── Results ──────────────────────────────────────────────────────────────
    console.print("[bold]Prompt:[/bold]")
    console.print(f"  {data.prompt.score:.0f}%")
    console.print()

    console.print("[bold]RAG:[/bold]")
    if data.rag.retrieval_available:
        console.print(f"  {data.rag.score:.0f}%")
    else:
        console.print(f"  {data.rag.score:.0f}%  [dim](no documents provided)[/dim]")
    console.print()

    console.print("[bold]Fine-tuning:[/bold]")
    console.print(f"  {data.ft.expected_score:.0f}% estimated")
    console.print()

    console.print(_THIN_DIV)

    # ── Recommendation ───────────────────────────────────────────────────────
    console.print()
    console.print("[bold]Recommendation:[/bold]")
    console.print(f"  [bold {color}]{data.recommendation.method}[/bold {color}]")
    console.print()

    console.print(f"[bold]Confidence:[/bold]  {data.recommendation.confidence:.0f}%")
    console.print()

    console.print("[bold]Reason:[/bold]")
    console.print(f"  {data.recommendation.reason}")
    console.print()

    # ── Savings ──────────────────────────────────────────────────────────────
    if data.recommendation.savings_usd > 0:
        console.print("[bold green]Estimated savings:[/bold green]")
        console.print(f"  ${data.recommendation.savings_usd:.0f} GPU cost avoided")
        console.print(f"  {data.recommendation.savings_hours:.0f} GPU hours avoided")
        console.print()

    console.print(_DIVIDER)
    console.print()
