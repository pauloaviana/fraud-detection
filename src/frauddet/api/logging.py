"""Structured JSON logging for the service.

One JSON object per line on stdout. Prediction logs carry operational metadata only — request id,
latency, decision, probability, policy, model/bundle identity, a truncated hash of the row id — never
the transaction payload (amounts, merchants, coordinates, card numbers, dates of birth ...).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sys
from typing import Any

_RESERVED = {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info",
             "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread",
             "threadName", "processName", "process", "message", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname, "logger": record.name, "event": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                doc[k] = v
        if record.exc_info:
            doc["exception"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(doc, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel("WARNING")     # access lines carry query strings; we log our own
    return logging.getLogger("frauddet.api")


def safe_id(value: Any) -> str | None:
    """Truncated SHA-256 of an identifier: correlatable across logs, not the raw id."""
    if value is None:
        return None
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]
