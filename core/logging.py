from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[arg-type]
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO") -> None:
    cee_logger = logging.getLogger("cee")
    if not cee_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        cee_logger.addHandler(handler)
    cee_logger.setLevel(level.upper())
    cee_logger.propagate = False


def ensure_trace_id(trace_id: str | None = None) -> str:
    tid = trace_id or str(uuid4())
    trace_id_var.set(tid)
    return tid


@contextmanager
def traced_span(name: str, **fields: object):
    logger = logging.getLogger("cee.trace")
    start = perf_counter()
    logger.info("span.start", extra={"extra_fields": {"span": name, **fields}})
    try:
        yield
        logger.info(
            "span.end",
            extra={"extra_fields": {"span": name, "duration_ms": round((perf_counter() - start) * 1000, 3), **fields}},
        )
    except Exception as exc:
        logger.exception(
            "span.error",
            extra={"extra_fields": {"span": name, "error": str(exc), **fields}},
        )
        raise
