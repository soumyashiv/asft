"""
Dynamic Layer Selection — Benchmarks layer importance, selects training targets,
and generates explainability reports and visualizations.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ===========================================================================
# Layer Benchmarker
# ===========================================================================

@dataclass
class LayerProfile:
    name: str
    param_count: int
    importance_score: float = 0.0
    gradient_norm: float = 0.0
    activation_norm: float = 0.0
    weight_magnitude: float = 0.0
    selected: bool = False
    layer_type: str = "unknown"


class LayerBenchmarker:
    """
    Probes each layer's gradient magnitude and activation norm
    to produce an importance map used by the layer selector.
    """

    def __init__(self, model: nn.Module):
        self._model = model

    def profile_all_layers(self, dataloader=None, n_batches: int = 3,
                            device: str = "cpu") -> Dict[str, LayerProfile]:
        """Collect importance metrics for all layers."""
        profiles: Dict[str, LayerProfile] = {}

        # 1. Weight magnitudes (always available)
        for name, module in self._model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data.abs()
                profiles[name] = LayerProfile(
                    name=name,
                    param_count=module.weight.numel(),
                    weight_magnitude=float(w.mean()),
                    layer_type=type(module).__name__,
                )

        # 2. Gradient norms (if dataloader provided)
        if dataloader is not None:
            self._collect_gradient_norms(profiles, dataloader, n_batches, device)

        # 3. Compute composite importance
        for profile in profiles.values():
            profile.importance_score = self._compute_importance(profile)

        return profiles

    def _collect_gradient_norms(self, profiles, dataloader, n_batches, device):
        self._model.train()
        grad_sums: Dict[str, float] = {n: 0.0 for n in profiles}
        n_processed = 0

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            try:
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items()
                              if isinstance(v, torch.Tensor) and k in ("input_ids", "attention_mask", "labels")}
                    outputs = self._model(**inputs)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                else:
                    loss = self._model(batch[0].to(device))[0]

                loss.backward()

                for name, module in self._model.named_modules():
                    if name in profiles and hasattr(module, "weight") and module.weight.grad is not None:
                        grad_sums[name] += float(module.weight.grad.norm(2))

                self._model.zero_grad()
                n_processed += 1
            except Exception as e:
                logger.debug("Gradient collection failed for batch %d: %s", i, e)
                self._model.zero_grad()

        if n_processed > 0:
            for name in profiles:
                profiles[name].gradient_norm = grad_sums[name] / n_processed

        self._model.eval()

    def _compute_importance(self, p: LayerProfile) -> float:
        # Weighted combination of metrics
        score = (
            0.4 * min(1.0, p.gradient_norm / 0.1)       # gradient contribution
            + 0.3 * min(1.0, p.weight_magnitude / 0.5)  # weight magnitude
            + 0.3 * min(1.0, p.activation_norm / 0.5)   # activation norm
        )
        return round(score, 4)


# ===========================================================================
# Layer Selector
# ===========================================================================

class LayerSelector:
    """
    Selects the top-K layers for training based on importance scores.
    Only relevant layers are trained; all others are frozen.
    """

    def __init__(self, sparsity_ratio: float = 0.95,
                 always_train: Optional[List[str]] = None):
        self._sparsity_ratio = sparsity_ratio
        # Layer name fragments to always include (e.g., "lm_head")
        self._always_train = always_train or ["lm_head", "embed_tokens"]

    def select(
        self,
        layer_profiles: Dict[str, LayerProfile],
        task_complexity: float = 0.5,
    ) -> List[str]:
        """
        Return list of layer names to train.
        Adjusts selection based on task complexity.
        """
        # Adjust ratio: complex tasks → train more layers
        ratio = max(0.5, self._sparsity_ratio - task_complexity * 0.2)
        n_to_train = max(1, int(len(layer_profiles) * (1.0 - ratio)))

        # Sort by importance
        ranked = sorted(
            layer_profiles.values(),
            key=lambda p: p.importance_score,
            reverse=True,
        )

        selected = []

        # Always include forced layers
        for p in ranked:
            if any(forced in p.name for forced in self._always_train):
                p.selected = True
                selected.append(p.name)

        # Add top-N by importance (up to budget)
        for p in ranked:
            if len(selected) >= n_to_train:
                break
            if p.name not in selected:
                p.selected = True
                selected.append(p.name)

        logger.info(
            "LayerSelector: selected %d/%d layers (sparsity=%.2f)",
            len(selected), len(layer_profiles), ratio
        )
        return selected

    def apply_to_model(self, model: nn.Module, selected_layers: List[str]) -> None:
        """Freeze non-selected layers, unfreeze selected."""
        for name, module in model.named_modules():
            for param in module.parameters(recurse=False):
                should_train = any(sel in name or name in sel for sel in selected_layers)
                param.requires_grad_(should_train)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Trainable: %d / %d params (%.2f%%)",
            trainable, total, 100 * trainable / max(1, total)
        )


# ===========================================================================
# Explainability
# ===========================================================================

class LayerExplainer:
    """
    Generates human-readable reports and visualizations
    explaining which layers were selected and why.
    """

    def generate_report(
        self,
        layer_profiles: Dict[str, LayerProfile],
        selected_layers: List[str],
        output_dir: str = "./asft_data/reports",
    ) -> str:
        """Generate a JSON + text report of layer selection."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        report_data = {
            "total_layers": len(layer_profiles),
            "selected_count": len(selected_layers),
            "sparsity": 1.0 - len(selected_layers) / max(1, len(layer_profiles)),
            "layers": [],
        }

        ranked = sorted(layer_profiles.values(), key=lambda p: p.importance_score, reverse=True)
        for p in ranked:
            report_data["layers"].append({
                "name": p.name,
                "type": p.layer_type,
                "params": p.param_count,
                "importance": p.importance_score,
                "gradient_norm": p.gradient_norm,
                "weight_magnitude": p.weight_magnitude,
                "selected": p.name in selected_layers,
            })

        json_path = Path(output_dir) / "layer_selection_report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)

        # Text summary
        lines = [
            "=== ASFT Layer Selection Report ===",
            f"Total layers    : {report_data['total_layers']}",
            f"Selected        : {report_data['selected_count']}",
            f"Sparsity        : {report_data['sparsity']:.2%}",
            "",
            f"{'Layer':<50} {'Type':<15} {'Importance':>10} {'Grad':>8} {'Sel':>5}",
            "-" * 95,
        ]
        for layer in report_data["layers"][:30]:
            lines.append(
                f"{layer['name']:<50} {layer['type']:<15} "
                f"{layer['importance']:>10.4f} {layer['gradient_norm']:>8.4f} "
                f"{'✓' if layer['selected'] else '':>5}"
            )

        text_path = Path(output_dir) / "layer_selection_report.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Layer report saved: %s", json_path)
        return str(json_path)

    def plot_importance(self, layer_profiles: Dict[str, LayerProfile],
                         output_dir: str = "./asft_data/reports") -> Optional[str]:
        """Generate a bar chart of layer importances."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            names = [p.name.split(".")[-1] for p in list(layer_profiles.values())[:30]]
            scores = [p.importance_score for p in list(layer_profiles.values())[:30]]
            colors = ["#2ecc71" if p.selected else "#e74c3c"
                      for p in list(layer_profiles.values())[:30]]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(range(len(names)), scores, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Importance Score")
            ax.set_title("ASFT Layer Importance (green=selected, red=frozen)")
            plt.tight_layout()

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plot_path = str(Path(output_dir) / "layer_importance.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Layer importance plot saved: %s", plot_path)
            return plot_path
        except Exception as e:
            logger.warning("Could not generate plot: %s", e)
            return None
