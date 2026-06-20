"""
Working Memory — Fast in-session scratch space.
Holds current task context, intermediate results, and active state.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryItem:
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    ttl: float | None = None  # seconds; None = no expiry

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


class WorkingMemory:
    """
    Fast key-value in-memory store for the current session.
    Also maintains a bounded history deque of recent items.
    """

    def __init__(self, max_items: int = 1000, history_size: int = 200):
        self._store: dict[str, MemoryItem] = {}
        self._history: deque[MemoryItem] = deque(maxlen=history_size)
        self._max_items = max_items

    def set(self, key: str, value: Any, tags: list[str] | None = None,
            ttl: float | None = None) -> None:
        """Store or overwrite a value."""
        item = MemoryItem(key=key, value=value, tags=tags or [], ttl=ttl)
        if len(self._store) >= self._max_items and key not in self._store:
            # Evict oldest non-pinned item
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = item
        self._history.append(item)

    def get(self, key: str, default: Any = None) -> Any:
        item = self._store.get(key)
        if item is None:
            return default
        if item.is_expired:
            del self._store[key]
            return default
        return item.value

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    def purge_expired(self) -> int:
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        return len(expired)

    def search_by_tag(self, tag: str) -> list[MemoryItem]:
        return [item for item in self._store.values() if tag in item.tags]

    def recent(self, n: int = 20) -> list[MemoryItem]:
        """Return N most recently added items from history."""
        return list(self._history)[-n:]

    def all_keys(self) -> list[str]:
        return list(self._store.keys())

    def snapshot(self) -> dict[str, Any]:
        """Return a plain-dict snapshot of current state (excluding expired)."""
        self.purge_expired()
        return {k: v.value for k, v in self._store.items()}

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store and not self._store[key].is_expired
