"""Path utilities and filename parsing."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

FILENAME_PATTERN = re.compile(r"sms-call-internet-mi-(\d{4})-(\d{2})-(\d{2})\.txt$")


def date_from_filename(path: Path | str) -> date | None:
    """Return the calendar date encoded in a daily TSV filename, or None."""
    m = FILENAME_PATTERN.search(str(path))
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def partition_path(interim_dir: Path, day: date) -> Path:
    return interim_dir / f"year_month={day.year:04d}-{day.month:02d}" / f"day={day.isoformat()}.parquet"


def manifest_path(interim_dir: Path) -> Path:
    return interim_dir / "_manifest.json"


def missing_path(interim_dir: Path) -> Path:
    return interim_dir / "_missing.json"


def list_daily_files(raw_dir: Path) -> list[tuple[date, Path]]:
    """Return (date, path) tuples sorted by date for files that match the pattern."""
    items: list[tuple[date, Path]] = []
    for p in sorted(raw_dir.iterdir() if raw_dir.is_dir() else []):
        d = date_from_filename(p)
        if d is not None and p.is_file():
            items.append((d, p))
    return items
