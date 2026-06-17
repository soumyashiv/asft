"""
Memory Manager — Unified orchestrator for all 5 memory systems.

Priority resolution order (ASFT specification):
  1. Working Memory  (session-local, instant)
  2. Episodic Memory (recent events)
  3. Semantic Memory (structured facts)
  4. Long-Term Memory (consolidated knowledge)
  5. Vector Memory   (semantic nearest-neighbor)

Always try higher-priority memory first. Only escalate when needed.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from asft.memory.working_memory import WorkingMemory
from asft.memory.episodic_memory import EpisodicMemory, EventRecord
from asft.memory.semantic_memory import SemanticMemory, FactRecord
from asft.memory.long_term_memory import LongTermMemory
from asft.memory.vector_memory import VectorMemory
from asft.memory.consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)


@dataclass
class MemoryQueryResult:
    source: str  # "working" | "episodic" | "semantic" | "long_term" | "vector"
    content: Any
    confidence: float = 1.0
    hit: bool = True


class MemoryManager:
    """
    Single entry-point for all memory operations.
    Implements the ASFT learning priority hierarchy at the memory layer.
    """

    def __init__(self, config=None, session_id: Optional[str] = None):
        self._session_id = session_id or str(uuid.uuid4())
        self._config = config

        db_path = "./asft_data/memory.db"
        chroma_dir = "./asft_data/chroma"
        vector_backend = "chromadb"
        embedding_model = "all-MiniLM-L6-v2"

        if config:
            db_path = config.memory.sqlite_path
            chroma_dir = config.memory.chroma_persist_dir
            vector_backend = config.memory.vector_backend
            embedding_model = config.memory.embedding_model

        # Ensure dirs exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)

        # Instantiate all memory systems
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(db_path=db_path)
        self.vector = VectorMemory.from_config(config.memory) if config else VectorMemory(
            backend=vector_backend,
            embedding_model=embedding_model,
            persist_dir=chroma_dir,
            collection_name="asft_memory",
        )
        self.semantic = SemanticMemory(db_path=db_path, vector_memory=self.vector)
        self.long_term = LongTermMemory(db_path=db_path)
        self.consolidator = MemoryConsolidator(self.episodic, self.long_term, self.semantic)

        logger.info("MemoryManager initialized (session=%s)", self._session_id)

    # ------------------------------------------------------------------
    # Unified Query Interface
    # ------------------------------------------------------------------

    def query(self, query: str, top_k: int = 5) -> List[MemoryQueryResult]:
        """
        Query all memory systems in priority order.
        Returns results from the first system that has relevant data.
        Aggregates across systems when multiple sources have matches.
        """
        results: List[MemoryQueryResult] = []

        # 1. Working memory — exact key lookup
        wm_val = self.working.get(query)
        if wm_val is not None:
            results.append(MemoryQueryResult(source="working", content=wm_val, confidence=1.0))

        # 2. Semantic memory — structured fact lookup
        facts = self.semantic.semantic_search(query, top_k=top_k)
        for f in facts:
            results.append(MemoryQueryResult(
                source="semantic",
                content=f,
                confidence=f.get("confidence", 0.8),
            ))

        # 3. Long-term memory — category-based retrieval
        lt_results = self.long_term.retrieve(category="task_performance", key=query, limit=top_k)
        for r in lt_results:
            results.append(MemoryQueryResult(source="long_term", content=r, confidence=0.85))

        # 4. Vector memory — semantic nearest-neighbor
        if not results:
            vector_results = self.vector.search(query, top_k=top_k)
            for vr in vector_results:
                results.append(MemoryQueryResult(
                    source="vector",
                    content=vr.doc.text,
                    confidence=vr.score,
                ))

        if not results:
            results.append(MemoryQueryResult(source="none", content=None, confidence=0.0, hit=False))

        return results

    def can_answer_from_memory(self, query: str, threshold: float = 0.7) -> bool:
        """
        Determine whether memory retrieval alone can answer this query
        (before considering training or tool usage).
        """
        results = self.query(query, top_k=3)
        if not results or not results[0].hit:
            return False
        return any(r.confidence >= threshold for r in results)

    # ------------------------------------------------------------------
    # Store Interfaces
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any, tags: Optional[List[str]] = None) -> None:
        """Store something in working memory."""
        self.working.set(key, value, tags=tags)

    def learn_fact(self, subject: str, predicate: str, obj: str,
                   source: str = "asft", confidence: float = 1.0) -> str:
        """Store a semantic fact."""
        return self.semantic.add_fact(FactRecord(
            subject=subject, predicate=predicate, object=obj,
            source=source, confidence=confidence,
        ))

    def record_event(self, event_type: str, context: Dict, outcome: Dict,
                     success: bool = True, confidence: float = 1.0,
                     duration: float = 0.0, task_id: Optional[str] = None) -> int:
        """Record a task event to episodic memory."""
        return self.episodic.record(EventRecord(
            event_type=event_type,
            context=context,
            outcome=outcome,
            success=success,
            confidence=confidence,
            duration_seconds=duration,
            task_id=task_id,
            session_id=self._session_id,
        ))

    def index_text(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Index text in vector memory for semantic search."""
        self.vector.add_text(doc_id, text, metadata=metadata)

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def maybe_consolidate(self, interval_hours: float = 24.0,
                          min_events: int = 100) -> Optional[Dict]:
        """Run consolidation if due."""
        if self.consolidator.should_run(interval_hours, min_events):
            return self.consolidator.run()
        return None

    def force_consolidate(self) -> Dict:
        """Force-run consolidation immediately."""
        return self.consolidator.run()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "working_memory_items": len(self.working),
            "episodic_events": self.episodic.count(),
            "semantic_facts": self.semantic.count(),
            "long_term_entries": self.long_term.count(),
            "vector_documents": self.vector.count(),
            "session_id": self._session_id,
        }
