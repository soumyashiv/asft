"""
Memory Consolidator — Runs periodic consolidation jobs:
  - Summarizes episodic events → long-term memory
  - Deduplicates semantic facts
  - Prunes stale working memory
  - Compresses memory stores
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Analyzes recent episodic events and extracts patterns into long-term memory.
    Should be run periodically (e.g., every 24h or after N events).
    """

    def __init__(self, episodic_memory, long_term_memory, semantic_memory=None):
        self._episodic = episodic_memory
        self._long_term = long_term_memory
        self._semantic = semantic_memory
        self._last_run: float = 0.0

    def should_run(self, interval_hours: float = 24.0, min_events: int = 100) -> bool:
        elapsed = (time.time() - self._last_run) / 3600
        if elapsed < interval_hours:
            return False
        count = self._episodic.count()
        return count >= min_events

    def run(self, window_hours: float = 168.0) -> Dict[str, Any]:
        """
        Run consolidation over the past `window_hours`.
        Returns a summary of what was consolidated.
        """
        start = time.time()
        logger.info("Starting memory consolidation (window=%.0fh)", window_hours)
        since = time.time() - window_hours * 3600

        # Fetch recent events
        events = self._episodic.query(since_timestamp=since, limit=5000, order_desc=False)

        results = {
            "events_analyzed": len(events),
            "patterns_stored": 0,
            "failure_patterns": 0,
            "success_patterns": 0,
        }

        if not events:
            logger.info("No events to consolidate")
            self._last_run = time.time()
            return results

        # Extract failure patterns
        failure_patterns = self._extract_failure_patterns(events)
        for pattern in failure_patterns:
            self._long_term.store(
                category="failure_pattern",
                key=pattern["event_type"],
                content=pattern["description"],
                summary=pattern["summary"],
                importance=min(1.0, pattern["count"] / 10),
                source_events=pattern["event_ids"],
            )
            results["failure_patterns"] += 1
            results["patterns_stored"] += 1

        # Extract success patterns
        success_patterns = self._extract_success_patterns(events)
        for pattern in success_patterns:
            self._long_term.store(
                category="success_pattern",
                key=pattern["event_type"],
                content=pattern["description"],
                summary=pattern["summary"],
                importance=min(1.0, pattern["count"] / 10),
                source_events=pattern["event_ids"],
            )
            results["success_patterns"] += 1
            results["patterns_stored"] += 1

        # Extract task performance summaries
        task_summaries = self._summarize_by_task(events)
        for task_type, summary in task_summaries.items():
            self._long_term.store(
                category="task_performance",
                key=task_type,
                content=str(summary),
                summary=f"Task '{task_type}': success_rate={summary['success_rate']:.2%}, count={summary['count']}",
                importance=0.6,
            )
            results["patterns_stored"] += 1

        duration = time.time() - start
        results["duration_seconds"] = round(duration, 2)
        self._last_run = time.time()
        logger.info("Consolidation complete: %s", results)
        return results

    def _extract_failure_patterns(self, events: List[Dict]) -> List[Dict]:
        failures = [e for e in events if not e.get("success", True)]
        type_counter: Counter = Counter(e["event_type"] for e in failures)
        patterns = []
        for event_type, count in type_counter.most_common(20):
            matching = [e for e in failures if e["event_type"] == event_type]
            event_ids = [e["id"] for e in matching]
            patterns.append({
                "event_type": event_type,
                "count": count,
                "description": f"Recurring failure pattern in '{event_type}' ({count} occurrences)",
                "summary": f"{event_type} failed {count} times",
                "event_ids": event_ids[:20],
            })
        return patterns

    def _extract_success_patterns(self, events: List[Dict]) -> List[Dict]:
        successes = [e for e in events if e.get("success", True)]
        type_counter: Counter = Counter(e["event_type"] for e in successes)
        patterns = []
        for event_type, count in type_counter.most_common(20):
            matching = [e for e in successes if e["event_type"] == event_type]
            event_ids = [e["id"] for e in matching]
            patterns.append({
                "event_type": event_type,
                "count": count,
                "description": f"High-performing pattern in '{event_type}' ({count} successes)",
                "summary": f"{event_type} succeeded {count} times",
                "event_ids": event_ids[:20],
            })
        return patterns

    def _summarize_by_task(self, events: List[Dict]) -> Dict[str, Dict]:
        by_type: Dict[str, List[Dict]] = {}
        for e in events:
            t = e.get("event_type", "unknown")
            by_type.setdefault(t, []).append(e)

        summaries = {}
        for task_type, task_events in by_type.items():
            total = len(task_events)
            successes = sum(1 for e in task_events if e.get("success", True))
            avg_confidence = sum(e.get("confidence", 1.0) for e in task_events) / total
            avg_duration = sum(e.get("duration_seconds", 0) for e in task_events) / total
            summaries[task_type] = {
                "count": total,
                "success_rate": successes / total,
                "avg_confidence": round(avg_confidence, 3),
                "avg_duration_seconds": round(avg_duration, 3),
            }
        return summaries
