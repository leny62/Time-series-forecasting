"""Context manager based timers used by the timing harness."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class TimerResult:
    label: str
    elapsed_s: float


class Timer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed_s = 0.0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_s = time.perf_counter() - self._start

    @property
    def result(self) -> TimerResult:
        return TimerResult(self.label, self.elapsed_s)
