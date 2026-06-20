"""
Representative Selector — Selects the most informative samples from each cluster.
Maximizes coverage while minimizing dataset size.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RepresentativeSelector:
    """
    Given cluster labels and embeddings, selects K representative samples
    from each cluster — minimizing redundancy while maximizing coverage.

    Strategies:
      - centroid:    Sample closest to cluster centroid
      - diversity:   Maximize pairwise diversity within selected set
      - hybrid:      Centroid + diversity combined
    """

    def __init__(self, strategy: str = "centroid", samples_per_cluster: int = 1):
        self._strategy = strategy
        self._samples_per_cluster = samples_per_cluster

    def select(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        texts: list[str] | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """
        Select representative indices from each cluster.

        Returns:
            selected_indices: list of selected sample indices
            stats: selection statistics
        """
        unique_clusters = sorted(set(cluster_labels))
        selected: list[int] = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # DBSCAN noise — keep all noise points
                noise_idx = np.where(cluster_labels == -1)[0].tolist()
                selected.extend(noise_idx)
                continue

            cluster_idx = np.where(cluster_labels == cluster_id)[0]
            cluster_emb = embeddings[cluster_idx]

            if len(cluster_idx) <= self._samples_per_cluster:
                selected.extend(cluster_idx.tolist())
                continue

            if self._strategy == "centroid":
                chosen = self._centroid_select(cluster_emb, cluster_idx)
            elif self._strategy == "diversity":
                chosen = self._diversity_select(cluster_emb, cluster_idx)
            else:  # hybrid
                chosen = self._hybrid_select(cluster_emb, cluster_idx)

            selected.extend(chosen)

        stats = {
            "original_count": len(embeddings),
            "selected_count": len(selected),
            "n_clusters": len([c for c in unique_clusters if c != -1]),
            "reduction_ratio": 1.0 - len(selected) / max(1, len(embeddings)),
            "strategy": self._strategy,
        }
        logger.info(
            "Representative selection: %d → %d (%.1f%% reduction)",
            stats["original_count"],
            stats["selected_count"],
            stats["reduction_ratio"] * 100,  # type: ignore
        )
        return selected, stats

    def _centroid_select(self, cluster_emb: np.ndarray, cluster_idx: np.ndarray) -> list[int]:
        centroid = cluster_emb.mean(axis=0)
        distances = np.linalg.norm(cluster_emb - centroid, axis=1)
        nearest = np.argsort(distances)[: self._samples_per_cluster]
        return cluster_idx[nearest].tolist()

    def _diversity_select(self, cluster_emb: np.ndarray, cluster_idx: np.ndarray) -> list[int]:
        """Greedy maximum-coverage selection."""
        k = min(self._samples_per_cluster, len(cluster_idx))
        selected_local = [0]  # start with first
        for _ in range(k - 1):
            # Select point maximally distant from already-selected
            min_dists = np.full(len(cluster_emb), np.inf)
            for sel in selected_local:
                dists = np.linalg.norm(cluster_emb - cluster_emb[sel], axis=1)
                min_dists = np.minimum(min_dists, dists)
            next_point = int(np.argmax(min_dists))
            selected_local.append(next_point)
        return cluster_idx[selected_local].tolist()

    def _hybrid_select(self, cluster_emb: np.ndarray, cluster_idx: np.ndarray) -> list[int]:
        """Centroid + diversity: half from each strategy."""
        k = self._samples_per_cluster
        half = max(1, k // 2)
        centroid_sel = self._centroid_select(cluster_emb, cluster_idx)[:half]
        diversity_sel = self._diversity_select(cluster_emb, cluster_idx)[: k - half]
        return list(set(centroid_sel + diversity_sel))
