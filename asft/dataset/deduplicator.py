"""
Dataset Deduplicator — MinHash LSH for near-duplicate detection and removal.
Dramatically reduces dataset size while preserving diversity.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class DatasetDeduplicator:
    """
    Uses MinHash + LSH to detect and remove near-duplicate samples.

    Target: reduce dataset size by 30–70% with <1% accuracy impact.
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self._threshold = threshold
        self._num_perm = num_perm

    def _shingle(self, text: str, k: int = 5) -> Set[str]:
        """Character k-gram shingling."""
        text = text.lower().strip()
        if len(text) <= k:
            return {text}
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def _minhash(self, shingles: Set[str]):
        try:
            from datasketch import MinHash
            m = MinHash(num_perm=self._num_perm)
            for s in shingles:
                m.update(s.encode("utf8"))
            return m
        except ImportError:
            raise ImportError("Install datasketch: pip install datasketch")

    def deduplicate(
        self, texts: List[str], ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Remove near-duplicates from a list of texts.

        Returns:
            kept_texts: deduplicated text list
            kept_ids: corresponding IDs
            stats: deduplication statistics
        """
        from datasketch import MinHashLSH

        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        lsh = MinHashLSH(threshold=self._threshold, num_perm=self._num_perm)
        minhashes = []

        for text in texts:
            shingles = self._shingle(text)
            mh = self._minhash(shingles)
            minhashes.append(mh)

        kept_indices: List[int] = []
        duplicate_count = 0

        for i, (mh, doc_id) in enumerate(zip(minhashes, ids)):
            candidates = lsh.query(mh)
            if not candidates:
                lsh.insert(doc_id, mh)
                kept_indices.append(i)
            else:
                duplicate_count += 1

        kept_texts = [texts[i] for i in kept_indices]
        kept_ids = [ids[i] for i in kept_indices]

        stats = {
            "original_count": len(texts),
            "kept_count": len(kept_texts),
            "removed_count": duplicate_count,
            "reduction_ratio": duplicate_count / max(1, len(texts)),
        }
        logger.info(
            "Deduplication: %d → %d (%.1f%% removed)",
            stats["original_count"], stats["kept_count"], stats["reduction_ratio"] * 100,
        )
        return kept_texts, kept_ids, stats


# Fix missing Optional import
from typing import Optional
