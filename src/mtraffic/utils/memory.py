"""Peak resident set sampling for memory measurement."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil


@dataclass(slots=True)
class MemorySample:
    peak_rss_mb: float
    end_rss_mb: float
    duration_s: float


class PeakRSSMonitor:
    """Background sampler. Use as a context manager."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._start = 0.0
        self._end_rss = 0
        self._proc = psutil.Process()

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            self._stop.wait(self._interval)

    def __enter__(self) -> PeakRSSMonitor:
        self._peak = self._proc.memory_info().rss
        self._start = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._end_rss = self._proc.memory_info().rss

    @property
    def sample(self) -> MemorySample:
        return MemorySample(
            peak_rss_mb=self._peak / (1024**2),
            end_rss_mb=self._end_rss / (1024**2),
            duration_s=time.perf_counter() - self._start,
        )
