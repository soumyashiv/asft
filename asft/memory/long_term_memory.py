"""
Long-Term Memory — Consolidated, summarized knowledge store.
Periodically summarizes episodic events into durable long-term knowledge.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class LongTermEntry(Base):
    __tablename__ = "long_term_memory"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    category = Column(String(128), nullable=False, index=True)
    key = Column(String(512), nullable=False, index=True)
    content = Column(Text, nullable=False)
    summary = Column(Text)
    source_events = Column(Text, default="[]")  # JSON list of episodic event IDs
    confidence = Column(Float, default=1.0)
    importance = Column(Float, default=0.5)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)
    access_count = Column(Integer, default=0)
    version = Column(Integer, default=1)


class LongTermMemory:
    """
    Durable consolidated knowledge store.
    Stores patterns, insights, and summaries derived from episodic events.
    Updated by the Consolidator, not by raw event logging.
    """

    def __init__(self, db_path: str = "./asft_data/memory.db"):
        self._engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def store(self, category: str, key: str, content: str,
              summary: Optional[str] = None, confidence: float = 1.0,
              importance: float = 0.5, source_events: Optional[List[int]] = None) -> str:
        """Create or update a long-term memory entry. Returns ID."""
        with self._Session() as session:
            # Check if entry with same category+key exists
            existing = (
                session.query(LongTermEntry)
                .filter(LongTermEntry.category == category, LongTermEntry.key == key)
                .first()
            )
            if existing:
                existing.content = content
                existing.summary = summary or existing.summary
                existing.confidence = max(existing.confidence, confidence)
                existing.importance = max(existing.importance, importance)
                existing.source_events = json.dumps(source_events or [])
                existing.updated_at = time.time()
                existing.version += 1
                entry_id = existing.id
            else:
                entry = LongTermEntry(
                    id=str(uuid.uuid4()),
                    category=category,
                    key=key,
                    content=content,
                    summary=summary,
                    source_events=json.dumps(source_events or []),
                    confidence=confidence,
                    importance=importance,
                )
                session.add(entry)
                entry_id = entry.id
            session.commit()
        return entry_id

    def retrieve(self, category: str, key: Optional[str] = None,
                 limit: int = 20) -> List[Dict[str, Any]]:
        with self._Session() as session:
            q = session.query(LongTermEntry).filter(LongTermEntry.category == category)
            if key:
                q = q.filter(LongTermEntry.key.ilike(f"%{key}%"))
            q = q.order_by(LongTermEntry.importance.desc()).limit(limit)
            results = q.all()
            for r in results:
                r.access_count += 1
            session.commit()
            return [self._to_dict(r) for r in results]

    def get_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        with self._Session() as session:
            e = session.get(LongTermEntry, entry_id)
            return self._to_dict(e) if e else None

    def top_by_importance(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._Session() as session:
            entries = (
                session.query(LongTermEntry)
                .order_by(LongTermEntry.importance.desc())
                .limit(n)
                .all()
            )
            return [self._to_dict(e) for e in entries]

    def delete(self, entry_id: str) -> bool:
        with self._Session() as session:
            e = session.get(LongTermEntry, entry_id)
            if e:
                session.delete(e)
                session.commit()
                return True
        return False

    def count(self) -> int:
        with self._Session() as session:
            return session.query(func.count(LongTermEntry.id)).scalar()

    def categories(self) -> List[str]:
        with self._Session() as session:
            rows = session.query(LongTermEntry.category).distinct().all()
            return [r[0] for r in rows]

    def _to_dict(self, e: LongTermEntry) -> Dict[str, Any]:
        return {
            "id": e.id,
            "category": e.category,
            "key": e.key,
            "content": e.content,
            "summary": e.summary,
            "confidence": e.confidence,
            "importance": e.importance,
            "created_at": e.created_at,
            "updated_at": e.updated_at,
            "access_count": e.access_count,
            "version": e.version,
            "source_events": json.loads(e.source_events or "[]"),
        }
