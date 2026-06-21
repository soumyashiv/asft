"""Public `Analyzer` API — the main entry point for the ASFT decision pipeline.

Usage (Python API)::

    from asft import Analyzer

    # From a task config dict
    result = Analyzer.from_config({
        "task_name": "customer support chatbot",
        "model": "meta-llama/Llama-3",
        "documents": "./knowledge_base/",
    })
    result.recommend()

    # From HuggingFace identifiers
    result = Analyzer.from_huggingface(
        model="meta-llama/Llama-3",
        dataset="my_dataset",
    )
    result.recommend()
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from asft.analysis.evaluator import MockPromptEvaluator, PromptEvaluator
from asft.analysis.finetune_estimator import FinetuneEstimator
from asft.analysis.rag_analyzer import MockRAGAnalyzer, RAGAnalyzer
from asft.analysis.recommender import DecisionRecommender
from asft.analysis.report import DecisionReportData, print_report

console = Console()


class Analyzer:
    """Orchestrates the ASFT 4-stage LLM decision pipeline.

    Stages
    ------
    1. Prompt baseline evaluation
    2. RAG evaluation (if documents / dataset provided)
    3. Fine-tuning cost & accuracy estimation
    4. Recommendation engine

    The class is intentionally thin — each stage is delegated to a dedicated,
    independently testable component.  Swap in real evaluators by subclassing
    :class:`~asft.analysis.evaluator.PromptEvaluator` or
    :class:`~asft.analysis.rag_analyzer.RAGAnalyzer`.
    """

    def __init__(
        self,
        task_config: dict[str, Any],
        prompt_evaluator: PromptEvaluator | None = None,
        rag_analyzer: RAGAnalyzer | None = None,
        ft_estimator: FinetuneEstimator | None = None,
        recommender: DecisionRecommender | None = None,
    ) -> None:
        self._config = task_config
        self._prompt_evaluator = prompt_evaluator or MockPromptEvaluator()
        self._rag_analyzer = rag_analyzer or MockRAGAnalyzer()
        self._ft_estimator = ft_estimator or FinetuneEstimator()
        self._recommender = recommender or DecisionRecommender()

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, task_config: dict[str, Any], **kwargs: Any) -> Analyzer:
        """Build an :class:`Analyzer` from a task config dictionary.

        Parameters
        ----------
        task_config:
            Dictionary with keys: ``task_name``, ``model``, ``dataset``,
            ``documents``, ``evaluation_metric``.
        **kwargs:
            Forwarded to :class:`Analyzer.__init__` — use to inject real
            evaluators or a custom recommender.
        """
        return cls(task_config=task_config, **kwargs)

    @classmethod
    def from_huggingface(
        cls,
        model: str,
        dataset: str | None = None,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> Analyzer:
        """Build an :class:`Analyzer` from HuggingFace model/dataset identifiers.

        Parameters
        ----------
        model:
            HuggingFace model ID, e.g. ``"meta-llama/Llama-3"``.
        dataset:
            HuggingFace dataset name or local path (optional).
        task_name:
            Human-readable task description.  Defaults to the model ID.
        **kwargs:
            Forwarded to :class:`Analyzer.__init__` — use to inject real
            evaluators once you have GPU access.

        Notes
        -----
        The model and dataset IDs are validated against the HuggingFace Hub
        so typos surface immediately.  Actual inference is delegated to the
        evaluator; by default :class:`MockPromptEvaluator` is used (a
        ``[mock]`` notice is printed) so the command works without a GPU.
        """
        task_config: dict[str, Any] = {
            "task_name": task_name or model,
            "model": model,
        }
        if dataset:
            task_config["dataset"] = dataset

        # Soft-validate the model ID so users catch typos early
        try:
            from huggingface_hub import model_info  # type: ignore[import]

            model_info(model)
            console.print(f"[dim]✓ HuggingFace model resolved: [cyan]{model}[/cyan][/dim]")
        except Exception:  # noqa: BLE001
            console.print(
                f"[yellow]⚠  Could not resolve '{model}' on HuggingFace Hub "
                f"(offline or private repo) — continuing anyway.[/yellow]"
            )

        # Notify that mock evaluators are active
        if "prompt_evaluator" not in kwargs:
            console.print(
                "[dim]ℹ  Using [bold]MockPromptEvaluator[/bold] "
                "(no GPU required). Subclass PromptEvaluator to add real scoring.)[/dim]"
            )

        return cls(task_config=task_config, **kwargs)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run(self) -> DecisionReportData:
        """Execute all four pipeline stages and return a :class:`DecisionReportData`."""
        # Stage 1 — Prompt baseline
        prompt_result = self._prompt_evaluator.evaluate_baseline(self._config)

        # Stage 2 — RAG
        rag_result = self._rag_analyzer.evaluate_rag(self._config, prompt_result)

        # Stage 3 — Fine-tuning estimate
        ft_result = self._ft_estimator.estimate(self._config, rag_result)

        # Stage 4 — Recommendation
        rec_result = self._recommender.recommend(prompt_result, rag_result, ft_result)

        task_name = self._config.get("task_name", "Unnamed Task").title()
        return DecisionReportData(
            task_name=task_name,
            prompt=prompt_result,
            rag=rag_result,
            ft=ft_result,
            recommendation=rec_result,
        )

    def recommend(self) -> DecisionReportData:
        """Run the pipeline and print the formatted report.

        Returns the :class:`DecisionReportData` so callers can inspect results
        programmatically after printing.
        """
        report = self.run()
        print_report(report)
        return report
