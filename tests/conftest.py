"""Shared fixtures and import path setup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from mtraffic.utils import seed as _seed


@pytest.fixture(scope="session", autouse=True)
def _seed_everything() -> None:
    _seed.set_all(20251201)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
