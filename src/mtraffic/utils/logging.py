"""Lightweight logging setup. No analytics, no telemetry, no external sinks."""

from __future__ import annotations

import logging
import sys
from typing import Literal

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure(
    level: Literal["debug", "info", "warning", "error"] = "info",
    *,
    stream=None,
) -> None:
    """Configure the root logger once."""
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.setLevel(_LEVELS[level])
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(_LEVELS[level])
