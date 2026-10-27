"""Logging and per-stage wall-time tracking.

    with log.timings() as t:          # one dict per run (contextvar, thread-safe)
        with log.stage("ingest"):
            ...
    t == {"ingest": 1.23}
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_timings: ContextVar[dict[str, float] | None] = ContextVar("montage_timings", default=None)
_configured = False


def get_logger(name: str = "montage") -> logging.Logger:
    global _configured
    if not _configured:
        root = logging.getLogger("montage")
        if not root.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
            root.addHandler(h)
        root.setLevel(logging.INFO)
        root.propagate = False
        _configured = True
    return logging.getLogger(name if name.startswith("montage") else f"montage.{name}")


@contextmanager
def timings() -> Iterator[dict[str, float]]:
    d: dict[str, float] = {}
    token = _timings.set(d)
    try:
        yield d
    finally:
        _timings.reset(token)


def current_timings() -> dict[str, float] | None:
    return _timings.get()


@contextmanager
def stage(name: str, into: dict[str, float] | None = None) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        target = into if into is not None else _timings.get()
        if target is not None:
            target[name] = round(target.get(name, 0.0) + dt, 3)
        get_logger("stage").info("%s: %.2fs", name, dt)
