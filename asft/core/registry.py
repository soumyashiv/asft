"""
ASFT Registry — Central plugin/component registry.
Tracks all skill packs, memory backends, learning strategies, and tools.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


class RegistryEntry:
    def __init__(self, name: str, obj: Any, metadata: dict):
        self.name = name
        self.obj = obj
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<RegistryEntry name={self.name!r} type={type(self.obj).__name__}>"


class Registry:
    """
    Thread-safe central registry for ASFT components.

    Supports namespaced registration:
      - skill_packs   → skill pack instances
      - memory        → memory backend implementations
      - strategies    → learning strategy implementations
      - tools         → callable tools
      - models        → loaded model adapters
    """

    _instance: Optional["Registry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "Registry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._store: Dict[str, Dict[str, RegistryEntry]] = {}
        return cls._instance

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def register(self, namespace: str, name: str, obj: Any, metadata: Optional[dict] = None) -> None:
        """Register a component under a namespace."""
        if namespace not in self._store:
            self._store[namespace] = {}
        entry = RegistryEntry(name=name, obj=obj, metadata=metadata or {})
        self._store[namespace][name] = entry
        logger.debug("Registered %s/%s", namespace, name)

    def get(self, namespace: str, name: str) -> Any:
        """Retrieve a component. Raises KeyError if not found."""
        try:
            return self._store[namespace][name].obj
        except KeyError:
            raise KeyError(f"Component '{name}' not found in namespace '{namespace}'")

    def get_or_none(self, namespace: str, name: str) -> Optional[Any]:
        try:
            return self._store[namespace][name].obj
        except KeyError:
            return None

    def unregister(self, namespace: str, name: str) -> bool:
        """Remove a component. Returns True if removed."""
        try:
            del self._store[namespace][name]
            logger.debug("Unregistered %s/%s", namespace, name)
            return True
        except KeyError:
            return False

    def list(self, namespace: str) -> list[str]:
        """List all registered names in a namespace."""
        return list(self._store.get(namespace, {}).keys())

    def list_all(self) -> Dict[str, list[str]]:
        """List everything in all namespaces."""
        return {ns: list(entries.keys()) for ns, entries in self._store.items()}

    def exists(self, namespace: str, name: str) -> bool:
        return name in self._store.get(namespace, {})

    def get_metadata(self, namespace: str, name: str) -> dict:
        try:
            return self._store[namespace][name].metadata
        except KeyError:
            return {}

    # ------------------------------------------------------------------
    # Decorator-based registration
    # ------------------------------------------------------------------

    def skill(self, name: str, **metadata):
        """Decorator: register as a skill pack."""
        def decorator(cls_or_fn):
            self.register("skill_packs", name, cls_or_fn, metadata)
            return cls_or_fn
        return decorator

    def tool(self, name: str, **metadata):
        """Decorator: register as a tool."""
        def decorator(fn: Callable):
            self.register("tools", name, fn, metadata)
            return fn
        return decorator

    def strategy(self, name: str, **metadata):
        """Decorator: register as a learning strategy."""
        def decorator(cls_or_fn):
            self.register("strategies", name, cls_or_fn, metadata)
            return cls_or_fn
        return decorator

    # ------------------------------------------------------------------
    # Namespace shortcuts
    # ------------------------------------------------------------------

    def register_skill(self, name: str, pack, metadata: Optional[dict] = None) -> None:
        self.register("skill_packs", name, pack, metadata)

    def get_skill(self, name: str) -> Any:
        return self.get("skill_packs", name)

    def register_model(self, name: str, model, metadata: Optional[dict] = None) -> None:
        self.register("models", name, model, metadata)

    def get_model(self, name: str) -> Any:
        return self.get("models", name)

    def register_tool(self, name: str, fn: Callable, metadata: Optional[dict] = None) -> None:
        self.register("tools", name, fn, metadata)

    def get_tool(self, name: str) -> Callable:
        return self.get("tools", name)

    def summary(self) -> str:
        lines = ["=== ASFT Registry ==="]
        for ns, names in self.list_all().items():
            lines.append(f"  [{ns}] ({len(names)} entries)")
            for n in names:
                lines.append(f"    - {n}")
        return "\n".join(lines)


# Module-level singleton
registry = Registry()
