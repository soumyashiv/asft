"""
ASFT Training Cost Estimator — Evidence-based cost and time projection.

PURPOSE:
    Estimate the cost, time, and expected accuracy gain of a training job
    BEFORE committing any GPU compute. Low-ROI jobs are rejected early.

METHODOLOGY:
    Uses two complementary scaling law frameworks:

    1. Kaplan et al. (2020) — "Scaling Laws for Neural Language Models"
       - Loss ≈ (N_c / N)^α_N + (D_c / D)^α_D
       - α_N ≈ 0.076, α_D ≈ 0.095 (empirical constants from the paper)
       - N = model parameters, D = training tokens
       - N_c, D_c = irreducible compute thresholds

    2. Hoffmann et al. (2022) — "Chinchilla Scaling Laws"
       - Compute-optimal: D_optimal ≈ 20 × N (train 20 tokens per parameter)
       - Used to estimate how much data is actually needed

    3. GPU TFLOPS estimates (NVIDIA H100: 312 TFLOPS bf16)
       - FLOPs per step ≈ 6 × N × batch_tokens (forward + backward)
       - GPU hours = total_FLOPs / (GPU_TFLOPS × 3600 × 1e12)

    4. Cloud pricing: ~$2–5/GPU-hour (on-demand A100/H100, mid-2024)

HONEST LIMITATIONS:
    - These are ORDER-OF-MAGNITUDE estimates, not guarantees.
    - Actual training loss depends heavily on data quality, learning rate,
      architecture choices, and hardware efficiency utilization.
    - The accuracy gain estimate is a weak heuristic: there is no universal
      formula relating training steps to downstream task accuracy.
    - Use this to decide "is this worth investigating?" not for billing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known model parameter counts (billions)
# Used to look up N when only a name is provided.
# ---------------------------------------------------------------------------
_MODEL_PARAM_BILLIONS: dict[str, float] = {
    # Qwen2 family
    "qwen/qwen2-0.5b": 0.5,
    "qwen/qwen2-1.5b": 1.5,
    "qwen/qwen2-7b": 7.0,
    "qwen/qwen2-72b": 72.0,
    # Qwen2.5 family
    "qwen/qwen2.5-0.5b": 0.5,
    "qwen/qwen2.5-1.5b": 1.5,
    "qwen/qwen2.5-3b": 3.0,
    "qwen/qwen2.5-7b": 7.0,
    "qwen/qwen2.5-14b": 14.0,
    "qwen/qwen2.5-32b": 32.0,
    "qwen/qwen2.5-72b": 72.0,
    # Llama 3 family
    "meta-llama/meta-llama-3-8b": 8.0,
    "meta-llama/meta-llama-3-8b-instruct": 8.0,
    "meta-llama/meta-llama-3-70b": 70.0,
    "meta-llama/llama-3.1-8b": 8.0,
    "meta-llama/llama-3.1-70b": 70.0,
    "meta-llama/llama-3.2-1b": 1.0,
    "meta-llama/llama-3.2-3b": 3.0,
    # Mistral family
    "mistralai/mistral-7b-v0.1": 7.0,
    "mistralai/mistral-7b-instruct-v0.2": 7.0,
    "mistralai/mistral-nemo-12b": 12.0,
    # Phi family
    "microsoft/phi-2": 2.7,
    "microsoft/phi-3-mini-4k-instruct": 3.8,
    "microsoft/phi-3.5-mini-instruct": 3.8,
    # Gemma family
    "google/gemma-2b": 2.0,
    "google/gemma-7b": 7.0,
    "google/gemma-2-2b": 2.0,
    "google/gemma-2-9b": 9.0,
}

# GPU TFLOPS for bf16 (approximate peak; real utilization is ~30-50% of peak)
_GPU_TFLOPS: dict[str, float] = {
    "h100": 312.0,
    "a100_80gb": 312.0,
    "a100_40gb": 312.0,
    "a6000": 154.0,
    "rtx4090": 82.6,
    "rtx3090": 35.6,
    "v100": 112.0,
    "t4": 65.0,
    "cpu": 0.05,
    "unknown": 25.0,  # conservative default for unidentified GPU
}

# Cost per GPU-hour (USD, on-demand, mid-2024 estimates)
_GPU_COST_PER_HOUR: dict[str, float] = {
    "h100": 4.50,
    "a100_80gb": 3.20,
    "a100_40gb": 2.80,
    "a6000": 1.80,
    "rtx4090": 0.80,
    "rtx3090": 0.50,
    "v100": 2.00,
    "t4": 0.60,
    "cpu": 0.08,
    "unknown": 2.00,
}

# Trainable parameter fraction for each method (realistic, from literature)
_TRAINABLE_FRACTION: dict[str, float] = {
    "peft_lora": 0.005,  # LoRA: ~0.1–1% of params; 0.5% is typical
    "qlora": 0.005,  # QLoRA: same LoRA fraction, 4-bit base
    "lora": 0.005,
    "full": 1.0,
    "sparse": 0.05,  # Sparse: selective layer training, ~5%
}


@dataclass
class TrainingEstimate:
    """Results of a cost/time estimation."""

    model_name: str
    n_params_billions: float
    method: str
    dataset_size: int

    # Compute estimates
    total_flops: float = 0.0  # raw FLOPs
    gpu_hours: float = 0.0
    cost_usd: float = 0.0
    wall_time_minutes: float = 0.0

    # Parameter efficiency
    trainable_params_billions: float = 0.0
    trainable_fraction: float = 0.0

    # Accuracy projection
    accuracy_gain_estimate: float = 0.0  # very rough heuristic

    # Decision
    recommendation: str = "proceed"
    reasoning: str = ""
    roi_score: float = 0.0  # accuracy_gain / cost_usd

    # Warnings
    warnings: list = field(default_factory=list)


class CostEstimator:
    """
    Estimates training cost and time before any GPU is allocated.

    Usage:
        estimator = CostEstimator()
        est = estimator.estimate("Qwen/Qwen2-7B", dataset_size=10_000, method="qlora")
        print(est.recommendation, est.cost_usd)
    """

    def __init__(
        self,
        gpu_utilization_factor: float = 0.40,  # 40% of peak TFLOPS (realistic)
        gpu_cost_per_hour: float | None = None,
    ):
        self._utilization = gpu_utilization_factor
        self._gpu_cost_override = gpu_cost_per_hour

    def estimate(
        self,
        model_name: str,
        dataset_size: int,
        method: str = "qlora",
        hardware_profile: Any | None = None,
        max_steps: int = 500,
        batch_size: int = 1,
        seq_len: int = 512,
    ) -> TrainingEstimate:
        """
        Produce a training cost estimate.

        Args:
            model_name:     HuggingFace model name or local path.
            dataset_size:   Number of training samples.
            method:         Training method (qlora | peft_lora | full).
            hardware_profile: HardwareProfiler output. If None, uses 'unknown' GPU.
            max_steps:      Maximum training steps.
            batch_size:     Per-device batch size.
            seq_len:        Average sequence length in tokens.

        Returns:
            TrainingEstimate with cost, time, and recommendation.
        """
        n_params = self._lookup_params(model_name)
        gpu_name = self._get_gpu_name(hardware_profile)
        tflops = _GPU_TFLOPS.get(gpu_name, _GPU_TFLOPS["unknown"])
        cost_per_hour = self._gpu_cost_override or _GPU_COST_PER_HOUR.get(
            gpu_name, _GPU_COST_PER_HOUR["unknown"]
        )
        trainable_frac = _TRAINABLE_FRACTION.get(method, 0.005)

        # ---------------------------------------------------------------
        # FLOPs calculation
        # Reference: Kaplan et al. 2020, Section A.1
        # FLOPs_per_step ≈ 6 × N × batch_tokens
        # (factor 6: forward=2N, backward=4N)
        # ---------------------------------------------------------------
        actual_steps = min(max_steps, dataset_size)
        batch_tokens = batch_size * seq_len
        n_params_abs = n_params * 1e9

        # For PEFT methods, only trainable params require backward compute
        # Base model forward: 2N FLOPs; trainable backward: 4 × N_trainable
        n_trainable = n_params_abs * trainable_frac
        flops_per_step = (2 * n_params_abs + 4 * n_trainable) * batch_tokens
        total_flops = flops_per_step * actual_steps

        # ---------------------------------------------------------------
        # GPU hours = FLOPs / (effective TFLOPS × 3600 × 1e12)
        # ---------------------------------------------------------------
        effective_tflops = tflops * self._utilization
        gpu_hours = total_flops / (effective_tflops * 3.6e15)  # 3.6e15 = 1e12 * 3600
        cost_usd = gpu_hours * cost_per_hour
        wall_time_minutes = gpu_hours * 60

        # ---------------------------------------------------------------
        # Accuracy gain heuristic
        # This is extremely rough. Based on: a 7B model fine-tuned with
        # LoRA on 1k high-quality domain samples typically gains 5–15%
        # on domain-specific benchmarks (empirical observation, not a law).
        # Scale logarithmically with dataset size, diminishing returns.
        # ---------------------------------------------------------------
        base_gain = {
            "qlora": 0.08,
            "peft_lora": 0.09,
            "lora": 0.09,
            "full": 0.12,
            "sparse": 0.05,
        }.get(method, 0.07)
        dataset_scale = math.log10(max(10, dataset_size)) / 4.0  # 10→0.25, 10k→1.0, 1M→1.5
        size_penalty = 1.0 / math.log10(max(10, n_params * 1e3))  # larger model → smaller gain
        accuracy_gain = min(0.25, base_gain * dataset_scale * size_penalty)

        # ---------------------------------------------------------------
        # ROI and recommendation
        # ---------------------------------------------------------------
        roi_score = accuracy_gain / max(cost_usd, 0.001)
        recommendation, reasoning, warnings = self._recommend(
            method,
            cost_usd,
            gpu_hours,
            accuracy_gain,
            dataset_size,
            n_params,
            trainable_frac,
            hardware_profile,
        )

        logger.info(
            "CostEstimate | model=%s method=%s params=%.1fB dataset=%d "
            "steps=%d gpu_hours=%.2f cost=$%.2f accuracy_gain≈%.1f%%",
            model_name,
            method,
            n_params,
            dataset_size,
            actual_steps,
            gpu_hours,
            cost_usd,
            accuracy_gain * 100,
        )

        return TrainingEstimate(
            model_name=model_name,
            n_params_billions=n_params,
            method=method,
            dataset_size=dataset_size,
            total_flops=total_flops,
            gpu_hours=round(gpu_hours, 3),
            cost_usd=round(cost_usd, 4),
            wall_time_minutes=round(wall_time_minutes, 1),
            trainable_params_billions=round(n_trainable / 1e9, 4),
            trainable_fraction=trainable_frac,
            accuracy_gain_estimate=round(accuracy_gain, 4),
            recommendation=recommendation,
            reasoning=reasoning,
            roi_score=round(roi_score, 6),
            warnings=warnings,
        )

    def compare_methods(
        self, model_name: str, dataset_size: int, **kwargs
    ) -> dict[str, TrainingEstimate]:
        """Compare all training methods side-by-side."""
        return {
            method: self.estimate(model_name, dataset_size, method=method, **kwargs)
            for method in ["qlora", "peft_lora", "sparse", "full"]
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lookup_params(self, model_name: str) -> float:
        """Return parameter count in billions. Falls back to 7B if unknown."""
        key = model_name.lower()
        for known_name, params in _MODEL_PARAM_BILLIONS.items():
            if known_name in key:
                return params
        # Heuristic from name: look for patterns like "7b", "13b", "70b"
        import re

        m = re.search(r"(\d+\.?\d*)b", key)
        if m:
            return float(m.group(1))
        logger.warning("Unknown model size for '%s' — assuming 7B", model_name)
        return 7.0

    def _get_gpu_name(self, hardware_profile: Any | None) -> str:
        """Extract GPU name from hardware profile."""
        if hardware_profile is None:
            return "unknown"
        name = getattr(hardware_profile, "gpu_name", "").lower()
        for known in _GPU_TFLOPS:
            if known in name:
                return known
        return "unknown"

    def _recommend(
        self,
        method: str,
        cost_usd: float,
        gpu_hours: float,
        accuracy_gain: float,
        dataset_size: int,
        n_params: float,
        trainable_frac: float,
        hardware_profile: Any | None,
    ) -> tuple[str, str, list]:
        """Produce a human-readable recommendation."""
        warnings = []

        # Check for obvious mismatches
        if dataset_size < 50:
            warnings.append(
                f"Dataset has only {dataset_size} samples. "
                "Fine-tuning on fewer than 50 samples typically causes overfitting. "
                "Consider using retrieval or a skill pack instead."
            )

        if (
            n_params > 30
            and method == "peft_lora"
            and getattr(hardware_profile, "vram_gb", 999) < 24
        ):
            warnings.append(
                f"Model ({n_params}B params) may not fit in available VRAM with {method}. "
                "Consider QLoRA (4-bit) instead."
            )

        # ROI gate
        if accuracy_gain < 0.02:
            return (
                "retrieve",
                f"Estimated accuracy gain ({accuracy_gain*100:.1f}%) is very low. "
                "Try retrieval-augmented generation or a domain skill pack first. "
                "Only train if those alternatives are insufficient.",
                warnings,
            )

        if cost_usd > 100 and method != "qlora":
            return (
                "use_qlora",
                f"Estimated cost (${cost_usd:.2f}) is high for {method}. "
                "Switch to QLoRA (4-bit quantization) to reduce cost by ~60–70% "
                "with <2% accuracy degradation (Dettmers et al. 2023).",
                warnings,
            )

        if gpu_hours < 0.1 and dataset_size < 500:
            return (
                "proceed_cheap",
                f"Low cost (${cost_usd:.4f}, {gpu_hours*60:.1f} min). "
                "This is a cheap experiment — proceed.",
                warnings,
            )

        return (
            "proceed",
            f"Estimated {accuracy_gain*100:.1f}% accuracy gain at ${cost_usd:.2f} "
            f"({gpu_hours:.2f} GPU-hours). ROI: {accuracy_gain/max(cost_usd, 0.001):.4f}. "
            "Proceed with training.",
            warnings,
        )
