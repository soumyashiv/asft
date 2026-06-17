"""
Dataset Clusterer — Groups similar samples to identify redundancy.
Uses KMeans or DBSCAN on sentence embeddings.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DatasetClusterer:
    """
    Clusters dataset samples by semantic similarity.
    Enables representative sample selection within each cluster.
    """

    def __init__(self, method: str = "kmeans", n_clusters: Optional[int] = None,
                 min_cluster_size: int = 3, reduction_ratio: float = 0.3):
        self._method = method
        self._n_clusters = n_clusters
        self._min_cluster_size = min_cluster_size
        self._reduction_ratio = reduction_ratio

    def embed_texts(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """Embed texts using sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings

    def cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster embeddings.
        Returns (cluster_labels, stats).
        """
        n = len(embeddings)
        if self._method == "kmeans":
            return self._kmeans(embeddings, n)
        elif self._method == "dbscan":
            return self._dbscan(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self._method}")

    def _kmeans(self, embeddings: np.ndarray, n: int) -> Tuple[np.ndarray, Dict]:
        from sklearn.cluster import KMeans
        k = self._n_clusters or max(2, int(n * self._reduction_ratio))
        k = min(k, n)
        logger.info("KMeans clustering: n=%d k=%d", n, k)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        stats = {
            "method": "kmeans",
            "n_clusters": k,
            "inertia": float(km.inertia_),
            "n_samples": n,
        }
        return labels, stats

    def _dbscan(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=0.3, min_samples=self._min_cluster_size, metric="cosine")
        labels = db.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.info("DBSCAN: n_clusters=%d noise=%d", n_clusters, n_noise)
        stats = {
            "method": "dbscan",
            "n_clusters": n_clusters,
            "noise_points": n_noise,
            "n_samples": len(embeddings),
        }
        return labels, stats

    def cluster_texts(self, texts: List[str],
                      embedding_model: str = "all-MiniLM-L6-v2") -> Tuple[np.ndarray, Dict]:
        """Convenience: embed + cluster in one call."""
        embeddings = self.embed_texts(texts, model_name=embedding_model)
        return self.cluster(embeddings)
