"""
Semantic Memory — Structured fact/concept store with vector-backed retrieval.
Stores named facts, concepts, and relationships for instant recall.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Column, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Fact(Base):
    __tablename__ = "semantic_facts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    subject = Column(String(256), nullable=False, index=True)
    predicate = Column(String(256), nullable=False, index=True)
    object_ = Column("object", Text, nullable=False)
    source = Column(String(256))
    confidence = Column(Float, default=1.0)
    timestamp = Column(Float, default=time.time, index=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(Float, default=time.time)
    tags = Column(Text, default="[]")  # JSON list


@dataclass
class FactRecord:
    subject: str
    predicate: str
    object: str
    source: str = "unknown"
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)


class SemanticMemory:
    """
    Structured semantic memory for facts and concepts.
    Combines SQLite persistence with optional vector search.
    """

    def __init__(
        self, db_path: str = "./asft_data/memory.db", vector_memory=None, max_facts: int = 50_000
    ):
        self._db_path = db_path
        self._vector_memory = vector_memory  # Optional VectorMemory
        self._max_facts = max_facts

        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("SemanticMemory initialized")

    def add_fact(self, record: FactRecord) -> str:
        """Store a fact triple. Returns its ID."""
        fact_id = str(uuid.uuid4())
        with self._Session() as session:
            fact = Fact(
                id=fact_id,
                subject=record.subject,
                predicate=record.predicate,
                object_=record.object,
                source=record.source,
                confidence=record.confidence,
                timestamp=time.time(),
                tags=json.dumps(record.tags),
            )
            session.add(fact)
            session.commit()

        # Also index in vector memory for semantic search
        if self._vector_memory:
            text = f"{record.subject} {record.predicate} {record.object}"
            self._vector_memory.add_text(
                doc_id=fact_id,
                text=text,
                metadata={
                    "subject": record.subject,
                    "predicate": record.predicate,
                    "source": record.source,
                    "confidence": record.confidence,
                },
            )
        self._maybe_prune()
        return fact_id

    def query_by_subject(self, subject: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._Session() as session:
            facts = (
                session.query(Fact)
                .filter(Fact.subject.ilike(f"%{subject}%"))
                .order_by(Fact.confidence.desc())
                .limit(limit)
                .all()
            )
            self._touch(session, facts)
            return [self._to_dict(f) for f in facts]

    def query_by_predicate(self, predicate: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._Session() as session:
            facts = (
                session.query(Fact)
                .filter(Fact.predicate.ilike(f"%{predicate}%"))
                .order_by(Fact.confidence.desc())
                .limit(limit)
                .all()
            )
            self._touch(session, facts)
            return [self._to_dict(f) for f in facts]

    def semantic_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search using vector similarity if vector_memory is available."""
        if not self._vector_memory:
            # Fallback to keyword search
            return self.query_by_subject(query, limit=top_k)
        results = self._vector_memory.search(query, top_k=top_k)
        ids = [r.doc.id for r in results]
        if not ids:
            return []
        with self._Session() as session:
            facts = session.query(Fact).filter(Fact.id.in_(ids)).all()
            return [self._to_dict(f) for f in facts]

    def count(self) -> int:
        with self._Session() as session:
            return session.query(func.count(Fact.id)).scalar()

    def delete(self, fact_id: str) -> bool:
        with self._Session() as session:
            f = session.get(Fact, fact_id)
            if f:
                session.delete(f)
                session.commit()
                return True
        return False

    def _touch(self, session: Session, facts: list[Fact]) -> None:
        now = time.time()
        for f in facts:
            f.access_count += 1
            f.last_accessed = now
        session.commit()

    def _to_dict(self, f: Fact) -> dict[str, Any]:
        return {
            "id": f.id,
            "subject": f.subject,
            "predicate": f.predicate,
            "object": f.object_,
            "source": f.source,
            "confidence": f.confidence,
            "timestamp": f.timestamp,
            "access_count": f.access_count,
            "tags": json.loads(f.tags or "[]"),
        }

    def _maybe_prune(self) -> None:
        with self._Session() as session:
            total = session.query(func.count(Fact.id)).scalar()
            if total > self._max_facts:
                # Prune least-accessed, lowest-confidence first
                old = (
                    session.query(Fact)
                    .order_by(Fact.access_count.asc(), Fact.confidence.asc())
                    .limit(total - self._max_facts)
                    .all()
                )
                for f in old:
                    session.delete(f)
                session.commit()
