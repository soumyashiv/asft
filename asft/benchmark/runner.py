"""
Benchmarking Suite — Measures training time, memory, accuracy, inference speed,
skill effectiveness, and compares: Full FT vs LoRA vs QLoRA vs Sparse vs ASFT.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRecord:
    method: str
    task_type: str
    # Performance
    accuracy: float = 0.0
    task_success_rate: float = 0.0
    # Resources
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    # Efficiency
    trainable_params: int = 0
    total_params: int = 0
    sparsity: float = 0.0
    # Quality
    hallucination_rate: float = 0.0
    reliability_score: float = 0.0
    # Meta
    timestamp: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def param_efficiency(self) -> float:
        if self.total_params == 0:
            return 0.0
        return 1.0 - self.trainable_params / self.total_params

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["param_efficiency"] = self.param_efficiency
        return d


class BenchmarkRunner:
    """Runs benchmarks against all configured methods."""

    def __init__(self, output_dir: str = "./asft_data/benchmarks"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[BenchmarkRecord] = []

    def time_inference(self, model, tokenizer, prompt: str, device: str = "cpu",
                       n_runs: int = 10) -> float:
        """Returns average inference time in milliseconds."""
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model.generate(**inputs, max_new_tokens=50, do_sample=False)
                times.append((time.perf_counter() - t0) * 1000)
        return sum(times) / len(times)

    def measure_peak_memory_mb(self) -> float:
        """Measure current peak GPU memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def evaluate_accuracy(self, model, tokenizer, eval_samples: List[Dict],
                           device: str = "cpu") -> tuple:
        """
        Evaluate model on a set of (prompt, expected) pairs.
        Returns (accuracy, hallucination_rate, reliability).
        """
        from asft.accuracy.confidence_scorer import ConfidenceScorer
        scorer = ConfidenceScorer()

        correct = 0
        hallucinations = 0
        total = len(eval_samples)

        import torch
        for sample in eval_samples:
            prompt = sample["prompt"]
            expected = sample.get("expected", "")

            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=512).to(device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Simple accuracy: check if expected key terms appear in output
                if expected and all(kw.lower() in output_text.lower()
                                    for kw in expected.split()[:3]):
                    correct += 1

                # Confidence-based hallucination estimate
                score = scorer.score(output_text)
                if score.reliability < 0.5:
                    hallucinations += 1
            except Exception:
                pass

        accuracy = correct / max(1, total)
        hallucination_rate = hallucinations / max(1, total)
        reliability = 1.0 - hallucination_rate
        return accuracy, hallucination_rate, reliability

    def record(self, rec: BenchmarkRecord) -> None:
        self._records.append(rec)

    def save_results(self, name: str = "benchmark") -> str:
        output_path = self._output_dir / f"{name}_{int(time.time())}.json"
        data = {
            "records": [r.to_dict() for r in self._records],
            "summary": self._summarize(),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Benchmark results saved: %s", output_path)
        return str(output_path)

    def _summarize(self) -> Dict[str, Any]:
        if not self._records:
            return {}
        by_method: Dict[str, List[BenchmarkRecord]] = {}
        for r in self._records:
            by_method.setdefault(r.method, []).append(r)

        summary = {}
        for method, records in by_method.items():
            summary[method] = {
                "avg_accuracy": sum(r.accuracy for r in records) / len(records),
                "avg_training_time_s": sum(r.training_time_seconds for r in records) / len(records),
                "avg_inference_ms": sum(r.inference_time_ms for r in records) / len(records),
                "avg_peak_memory_mb": sum(r.peak_memory_mb for r in records) / len(records),
                "avg_param_efficiency": sum(r.param_efficiency for r in records) / len(records),
                "avg_hallucination_rate": sum(r.hallucination_rate for r in records) / len(records),
                "n_runs": len(records),
            }
        return summary

    def print_comparison_table(self) -> None:
        """Print a formatted comparison table to stdout."""
        summary = self._summarize()
        if not summary:
            print("No benchmark data available")
            return

        header = f"{'Method':<15} {'Accuracy':>10} {'Train(s)':>10} {'Infer(ms)':>10} {'Mem(MB)':>10} {'Eff%':>8} {'Halluc':>8}"
        print("\n" + "=" * len(header))
        print("ASFT Benchmark Comparison")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for method, stats in sorted(summary.items()):
            print(
                f"{method:<15} "
                f"{stats['avg_accuracy']:>10.3f} "
                f"{stats['avg_training_time_s']:>10.1f} "
                f"{stats['avg_inference_ms']:>10.1f} "
                f"{stats['avg_peak_memory_mb']:>10.1f} "
                f"{stats['avg_param_efficiency'] * 100:>8.1f} "
                f"{stats['avg_hallucination_rate']:>8.3f}"
            )
        print("=" * len(header) + "\n")
