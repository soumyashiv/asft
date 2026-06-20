"""
ASFT Observability — Structured logging setup.

Provides JSON-structured logging suitable for ingestion into
ELK, Datadog, Splunk, etc. Uses standard library logging wrapped
with formatting functions for minimal dependencies, while providing
a clean API.
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for log records.
    """

    def __init__(self, service_name: str = "asft"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = "".join(traceback.format_exception(*record.exc_info))

        # Add extra fields (e.g. from LoggerAdapter or extra={} arg)
        # We filter out standard LogRecord attributes
        standard_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "id",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_obj[key] = value

        return json.dumps(log_obj)


def configure_logging(log_level: int = logging.INFO, json_format: bool = True) -> None:
    """
    Configure the root logger.

    Args:
        log_level: The logging level (e.g., logging.INFO)
        json_format: If True, output as JSON. If False, output standard text.
    """
    root_logger = logging.getLogger()

    # Remove existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        # Standard format for local development
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
