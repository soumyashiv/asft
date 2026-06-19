"""
ASFT Events — Typed internal event bus.

Implements a lightweight, async pub/sub system for internal decoupling.
For example, when a job completes, it emits an event that the memory
consolidator can subscribe to without hard coupling the modules.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Set, Type

logger = logging.getLogger(__name__)


class BaseEvent:
    """Base class for all internal events."""
    pass


@dataclass
class JobCompletedEvent(BaseEvent):
    job_id: str
    job_type: str
    result: Dict[str, Any]


@dataclass
class JobFailedEvent(BaseEvent):
    job_id: str
    job_type: str
    error: str


@dataclass
class MemoryStoredEvent(BaseEvent):
    key: str
    source_tier: str


@dataclass
class SecurityViolationEvent(BaseEvent):
    violation_type: str
    details: str
    ip_address: str


# Type alias for event handlers
EventHandler = Callable[[BaseEvent], Coroutine[Any, Any, None]]


class EventBus:
    """
    In-memory async event bus.
    
    Handlers run in the background via asyncio.create_task to avoid
    blocking the publisher.
    """
    
    def __init__(self):
        self._subscribers: Dict[Type[BaseEvent], Set[EventHandler]] = {}
        
    def subscribe(self, event_type: Type[BaseEvent], handler: EventHandler) -> None:
        """Register a coroutine handler for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(handler)
        logger.debug("Subscribed %s to %s", handler.__name__, event_type.__name__)
        
    def publish(self, event: BaseEvent) -> None:
        """
        Publish an event to all registered subscribers.
        Executes handlers concurrently in the background.
        """
        event_type = type(event)
        handlers = self._subscribers.get(event_type, set())
        
        if not handlers:
            return
            
        for handler in handlers:
            # Schedule the coroutine to run in the background
            asyncio.create_task(self._safe_execute(handler, event))
            
    async def _safe_execute(self, handler: EventHandler, event: BaseEvent) -> None:
        try:
            await handler(event)
        except Exception:
            logger.exception("Event handler %s failed for event %s", 
                             handler.__name__, type(event).__name__)

# Singleton instance
bus = EventBus()
