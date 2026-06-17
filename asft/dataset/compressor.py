"""
Dataset Compressor — Full end-to-end compression pipeline.
Deduplicate → Cluster → Select → Export compressed dataset.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DatasetCompressor:
    """
    Orchestrates the full dataset compression pipeline:
      1. Deduplication (MinHash LSH)
      2. Clustering (KMeans/DBSCAN)
      3. Representative selection (centroid/diversity/hybrid)
      4. Export compressed dataset

    Targets: 30–70% size reduction with <1% accuracy impact.
    """

    def __init__(self, config=None):
        self._config = config
        threshold = getattr(config, "dedup_threshold", 0.85) if config else 0.85
        num_perm = getattr(config, "dedup_num_perm", 128) if config else 128
        cluster_method = getattr(config, "cluster_method", "kmeans") if config else "kmeans"
        reduction_ratio = getattr(config, "cluster_reduction_ratio", 0.3) if config else 0.3
        output_dir = getattr(config, "compressed_output_dir", "./asft_data/datasets") if config else "./asft_data/datasets"

        from asft.dataset.deduplicator import DatasetDeduplicator
        from asft.dataset.clusterer import DatasetClusterer
        from asft.dataset.representative_selector import RepresentativeSelector

        self._deduplicator = DatasetDeduplicator(threshold=threshold, num_perm=num_perm)
        self._clusterer = DatasetClusterer(method=cluster_method, reduction_ratio=reduction_ratio)
        self._selector = RepresentativeSelector(strategy="hybrid", samples_per_cluster=2)
        self._output_dir = Path(output_dir)

    def compress(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        output_name: str = "compressed_dataset",
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Full compression pipeline.

        Args:
            texts: list of text samples
            metadata: optional per-sample metadata dicts
            embedding_model: sentence-transformer model name
            output_name: output file prefix

        Returns:
            compressed_texts: the compressed dataset
            report: full pipeline stats
        """
        report: Dict[str, Any] = {"pipeline": []}
        original_count = len(texts)
        ids = [str(i) for i in range(len(texts))]

        # Stage 1: Deduplication
        logger.info("Stage 1: Deduplication (%d samples)", len(texts))
        texts, ids, dedup_stats = self._deduplicator.deduplicate(texts, ids)
        report["deduplication"] = dedup_stats
        report["pipeline"].append(f"dedup: {original_count} → {len(texts)}")

        # Stage 2: Clustering
        logger.info("Stage 2: Clustering (%d samples)", len(texts))
        if len(texts) < 10:
            cluster_labels = list(range(len(texts)))
            cluster_stats = {"skipped": True, "n_samples": len(texts)}
        else:
            import numpy as np
            embeddings = self._clusterer.embed_texts(texts, model_name=embedding_model)
            cluster_labels, cluster_stats = self._clusterer.cluster(embeddings)
        report["clustering"] = cluster_stats
        report["pipeline"].append(f"cluster: {len(texts)} → {len(set(cluster_labels))} clusters")

        # Stage 3: Representative selection
        logger.info("Stage 3: Representative selection")
        if not cluster_stats.get("skipped"):
            import numpy as np
            selected_indices, sel_stats = self._selector.select(
                embeddings=embeddings,
                cluster_labels=np.array(cluster_labels),
                texts=texts,
            )
            compressed_texts = [texts[i] for i in selected_indices]
            compressed_meta = [metadata[int(ids[i])] for i in selected_indices] if metadata else None
        else:
            compressed_texts = texts
            compressed_meta = metadata
            sel_stats = {"skipped": True}
        report["selection"] = sel_stats
        report["pipeline"].append(f"select: → {len(compressed_texts)} samples")

        # Stage 4: Export
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / f"{output_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(compressed_texts):
                entry = {"text": text}
                if compressed_meta and i < len(compressed_meta):
                    entry["metadata"] = compressed_meta[i]
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        report["output_path"] = str(output_path)
        report["original_count"] = original_count
        report["final_count"] = len(compressed_texts)
        report["total_reduction"] = 1.0 - len(compressed_texts) / max(1, original_count)

        logger.info(
            "Compression complete: %d → %d (%.1f%% reduction) → %s",
            original_count, len(compressed_texts), report["total_reduction"] * 100, output_path,
        )
        return compressed_texts, report

    def compress_jsonl(self, input_path: str, text_field: str = "text", **kwargs) -> Tuple[List[str], Dict]:
        """Load a JSONL dataset, compress it, and save."""
        texts = []
        meta = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                texts.append(obj.get(text_field, ""))
                meta.append({k: v for k, v in obj.items() if k != text_field})
        return self.compress(texts, metadata=meta, **kwargs)
