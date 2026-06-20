"""
ASFT Observability — Prometheus-compatible metrics collector.

Provides a lightweight metric collection interface that can be scraped
by Prometheus or exported to OpenTelemetry.
"""
import logging
from collections import defaultdict
from threading import Lock

from asft.core.interfaces import IMetricsCollector

logger = logging.getLogger(__name__)


class InMemoryMetricsCollector(IMetricsCollector):
    """
    In-memory metrics collector suitable for single-process deployments
    or as a fallback before true OpenTelemetry/Prometheus integration.
    """
    
    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()
        
    def _format_name(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        # Sort keys to ensure consistent formatting
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def increment(self, name: str, value: float = 1.0,
                  labels: dict[str, str] | None = None) -> None:
        full_name = self._format_name(name, labels)
        with self._lock:
            self._counters[full_name] += value
            
    def gauge(self, name: str, value: float,
              labels: dict[str, str] | None = None) -> None:
        full_name = self._format_name(name, labels)
        with self._lock:
            self._gauges[full_name] = value
            
    def histogram(self, name: str, value: float,
                  labels: dict[str, str] | None = None) -> None:
        full_name = self._format_name(name, labels)
        with self._lock:
            # Keep only the last 1000 observations to prevent memory leaks
            if len(self._histograms[full_name]) >= 1000:
                self._histograms[full_name].pop(0)
            self._histograms[full_name].append(value)
            
    def dump_metrics(self) -> str:
        """Dump metrics in Prometheus exposition format."""
        lines = []
        with self._lock:
            for k, v in self._counters.items():
                # Strip {} for TYPE declaration if present
                base_name = k.split("{")[0]
                lines.append(f"# TYPE {base_name} counter")
                lines.append(f"{k} {v}")
                
            for k, v in self._gauges.items():
                base_name = k.split("{")[0]
                lines.append(f"# TYPE {base_name} gauge")
                lines.append(f"{k} {v}")
                
            # Naive histogram representation for debug
            for k, vals in self._histograms.items():
                if vals:
                    base_name = k.split("{")[0]
                    lines.append(f"# TYPE {base_name}_count counter")
                    lines.append(f"{k}_count {len(vals)}")
                    lines.append(f"# TYPE {base_name}_sum counter")
                    lines.append(f"{k}_sum {sum(vals)}")
                    
        return "\n".join(lines) + "\n"

# Singleton instance
metrics = InMemoryMetricsCollector()
