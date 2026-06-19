"""
ASFT Job Store — Persistent, async-safe training and compression job state.

REPLACES the original in-memory `_jobs: Dict[str, Dict]` in server.py.

Why the original was broken:
  - Server restart = complete loss of all job state
  - No concurrent access protection (race conditions in FastAPI)
  - No job history or audit trail
  - No ability to query by status or type

This implementation:
  - Uses aiosqlite for async-safe writes from FastAPI
  - Survives server restarts
  - Provides full job history
  - Is replaceable with Redis for distributed deployments
    (implements IJobStore ABC)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from asft.core.interfaces import IJobStore, JobRecord
from asft.core.exceptions import JobNotFoundError

logger = logging.getLogger(__name__)


class SQLiteJobStore(IJobStore):
    """
    SQLite-backed persistent job store using aiosqlite.

    Thread-safe and coroutine-safe. Each write uses a short-lived
    connection to avoid holding the SQLite WAL open across event loop cycles.
    """

    def __init__(self, db_path: str = "./asft_data/jobs.db"):
        self._db_path = str(Path(db_path))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_init(self) -> None:
        """Create the jobs table if it does not exist."""
        if self._initialized:
            return
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id      TEXT PRIMARY KEY,
                        job_type    TEXT NOT NULL,
                        status      TEXT NOT NULL DEFAULT 'queued',
                        created_at  REAL NOT NULL,
                        updated_at  REAL NOT NULL,
                        payload     TEXT NOT NULL DEFAULT '{}',
                        result      TEXT,
                        error       TEXT
                    )
                """)
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type)"
                )
                await db.commit()
            self._initialized = True
            logger.debug("Job store initialised: %s", self._db_path)
        except ImportError:
            logger.error("aiosqlite not installed — job store unavailable. pip install aiosqlite")
            raise

    async def create(
        self,
        job_id: Optional[str] = None,
        job_type: str = "training",
        payload: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        await self._ensure_init()
        import aiosqlite

        if job_id is None:
            job_id = str(uuid.uuid4())[:12]
        now = time.time()
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            status="queued",
            created_at=now,
            updated_at=now,
            payload=payload or {},
        )

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO jobs
                   (job_id, job_type, status, created_at, updated_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    record.job_id,
                    record.job_type,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.payload),
                ),
            )
            await db.commit()

        logger.info("Job created | id=%s type=%s", job_id, job_type)
        return record

    async def get(self, job_id: str) -> Optional[JobRecord]:
        await self._ensure_init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None
        return self._row_to_record(row)

    async def update_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        await self._ensure_init()
        import aiosqlite

        now = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """UPDATE jobs
                   SET status = ?, updated_at = ?, result = ?, error = ?
                   WHERE job_id = ?""",
                (
                    status,
                    now,
                    json.dumps(result) if result is not None else None,
                    error,
                    job_id,
                ),
            )
            if db.total_changes == 0:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            await db.commit()

        logger.debug("Job updated | id=%s status=%s", job_id, status)

    async def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[JobRecord]:
        await self._ensure_init()
        import aiosqlite

        where_clauses = []
        params: list = []
        if job_type:
            where_clauses.append("job_type = ?")
            params.append(job_type)
        if status:
            where_clauses.append("status = ?")
            params.append(status)

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ) as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_record(r) for r in rows]

    async def cancel(self, job_id: str) -> bool:
        """Attempt to cancel a queued job. Returns True if cancelled."""
        record = await self.get(job_id)
        if record is None:
            return False
        if record.status not in ("queued",):
            return False
        await self.update_status(job_id, "cancelled")
        return True

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            job_type=row["job_type"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            payload=json.loads(row["payload"] or "{}"),
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
        )


# ---------------------------------------------------------------------------
# In-memory fallback (tests / no aiosqlite)
# ---------------------------------------------------------------------------


class InMemoryJobStore(IJobStore):
    """
    Pure in-memory job store for unit tests and development.
    Not suitable for production (state lost on restart).
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}

    async def create(self, job_id: Optional[str] = None,
                     job_type: str = "training",
                     payload: Optional[Dict[str, Any]] = None) -> JobRecord:
        if job_id is None:
            job_id = str(uuid.uuid4())[:12]
        now = time.time()
        record = JobRecord(
            job_id=job_id, job_type=job_type, status="queued",
            created_at=now, updated_at=now, payload=payload or {}
        )
        self._jobs[job_id] = record
        return record

    async def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    async def update_status(self, job_id: str, status: str,
                            result: Optional[Dict] = None,
                            error: Optional[str] = None) -> None:
        if job_id not in self._jobs:
            raise JobNotFoundError(f"Job '{job_id}' not found.")
        r = self._jobs[job_id]
        r.status = status
        r.updated_at = time.time()
        if result is not None:
            r.result = result
        if error is not None:
            r.error = error

    async def list_jobs(self, job_type: Optional[str] = None,
                        status: Optional[str] = None,
                        limit: int = 50) -> List[JobRecord]:
        records = list(self._jobs.values())
        if job_type:
            records = [r for r in records if r.job_type == job_type]
        if status:
            records = [r for r in records if r.status == status]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_job_store(db_path: Optional[str] = None) -> IJobStore:
    """
    Create the best available job store.
    Falls back to InMemoryJobStore if aiosqlite is unavailable.
    """
    try:
        import aiosqlite  # noqa: F401
        return SQLiteJobStore(db_path or "./asft_data/jobs.db")
    except ImportError:
        logger.warning("aiosqlite not available — using in-memory job store (dev only)")
        return InMemoryJobStore()
