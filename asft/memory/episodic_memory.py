"""
Episodic Memory — SQLite-backed event store with full temporal indexing.
Records task events, outcomes, context, and chains of reasoning.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class EpisodicEvent(Base):
    __tablename__ = "episodic_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False, index=True)
    task_id = Column(String(128), index=True)
    session_id = Column(String(128), index=True)
    timestamp = Column(Float, nullable=False, default=time.time, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Payload stored as JSON
    context = Column(Text, default="{}")    # input context
    outcome = Column(Text, default="{}")    # result/output
    metadata_ = Column("metadata", Text, default="{}")

    # Scoring
    success = Column(Integer, default=1)    # 1=success, 0=failure
    confidence = Column(Float, default=1.0)
    duration_seconds = Column(Float, default=0.0)

    __table_args__ = (
        Index("ix_event_type_ts", "event_type", "timestamp"),
        Index("ix_task_success", "task_id", "success"),
    )


@dataclass
class EventRecord:
    event_type: str
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    success: bool = True
    confidence: float = 1.0
    duration_seconds: float = 0.0


class EpisodicMemory:
    """
    SQLite-backed episodic memory.
    Stores the full history of task events for analysis and learning.
    """

    def __init__(self, db_path: str = "./asft_data/memory.db", max_events: int = 10_000):
        self._db_path = db_path
        self._max_events = max_events
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("EpisodicMemory initialized: %s", db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, record: EventRecord) -> int:
        """Store an event. Returns the new event ID."""
        with self._Session() as session:
            event = EpisodicEvent(
                event_type=record.event_type,
                task_id=record.task_id,
                session_id=record.session_id,
                timestamp=time.time(),
                context=json.dumps(record.context),
                outcome=json.dumps(record.outcome),
                metadata_=json.dumps(record.metadata),
                success=int(record.success),
                confidence=record.confidence,
                duration_seconds=record.duration_seconds,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            event_id = event.id
        self._maybe_prune()
        return event_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, event_id: int) -> Optional[Dict[str, Any]]:
        with self._Session() as session:
            e = session.get(EpisodicEvent, event_id)
            return self._to_dict(e) if e else None

    def query(
        self,
        event_type: Optional[str] = None,
        task_id: Optional[str] = None,
        success: Optional[bool] = None,
        since_timestamp: Optional[float] = None,
        limit: int = 100,
        order_desc: bool = True,
    ) -> List[Dict[str, Any]]:
        with self._Session() as session:
            q = session.query(EpisodicEvent)
            if event_type:
                q = q.filter(EpisodicEvent.event_type == event_type)
            if task_id:
                q = q.filter(EpisodicEvent.task_id == task_id)
            if success is not None:
                q = q.filter(EpisodicEvent.success == int(success))
            if since_timestamp:
                q = q.filter(EpisodicEvent.timestamp >= since_timestamp)
            if order_desc:
                q = q.order_by(EpisodicEvent.timestamp.desc())
            else:
                q = q.order_by(EpisodicEvent.timestamp.asc())
            return [self._to_dict(e) for e in q.limit(limit).all()]

    def failure_rate(self, event_type: Optional[str] = None, window_hours: float = 24.0) -> float:
        """Return fraction of failed events in the given window."""
        since = time.time() - window_hours * 3600
        with self._Session() as session:
            q = session.query(EpisodicEvent).filter(EpisodicEvent.timestamp >= since)
            if event_type:
                q = q.filter(EpisodicEvent.event_type == event_type)
            total = q.count()
            if total == 0:
                return 0.0
            failures = q.filter(EpisodicEvent.success == 0).count()
            return failures / total

    def count(self) -> int:
        with self._Session() as session:
            return session.query(func.count(EpisodicEvent.id)).scalar()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _to_dict(self, e: EpisodicEvent) -> Dict[str, Any]:
        return {
            "id": e.id,
            "event_type": e.event_type,
            "task_id": e.task_id,
            "session_id": e.session_id,
            "timestamp": e.timestamp,
            "context": json.loads(e.context or "{}"),
            "outcome": json.loads(e.outcome or "{}"),
            "metadata": json.loads(e.metadata_ or "{}"),
            "success": bool(e.success),
            "confidence": e.confidence,
            "duration_seconds": e.duration_seconds,
        }

    def _maybe_prune(self) -> None:
        with self._Session() as session:
            total = session.query(func.count(EpisodicEvent.id)).scalar()
            if total > self._max_events:
                cutoff = total - self._max_events
                oldest = (
                    session.query(EpisodicEvent)
                    .order_by(EpisodicEvent.timestamp.asc())
                    .limit(cutoff)
                    .all()
                )
                for e in oldest:
                    session.delete(e)
                session.commit()
                logger.debug("Pruned %d old episodic events", cutoff)
