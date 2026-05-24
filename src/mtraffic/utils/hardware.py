"""Capture hardware and library metadata for reproducibility."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import asdict, dataclass

import psutil


@dataclass(slots=True)
class HardwareInfo:
    os: str
    os_release: str
    machine: str
    processor: str
    python: str
    cpu_logical: int
    cpu_physical: int
    ram_gb: float
    torch_version: str | None
    tf_version: str | None
    has_cuda: bool
    has_mps: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _maybe_version(module: str) -> str | None:
    try:
        m = importlib.import_module(module)
    except ImportError:
        return None
    return getattr(m, "__version__", None)


def collect() -> HardwareInfo:
    has_cuda = False
    has_mps = False
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        pass
    vm = psutil.virtual_memory()
    return HardwareInfo(
        os=platform.system(),
        os_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        python=sys.version.split()[0],
        cpu_logical=psutil.cpu_count(logical=True) or 0,
        cpu_physical=psutil.cpu_count(logical=False) or 0,
        ram_gb=round(vm.total / (1024**3), 2),
        torch_version=_maybe_version("torch"),
        tf_version=_maybe_version("tensorflow"),
        has_cuda=has_cuda,
        has_mps=has_mps,
    )
