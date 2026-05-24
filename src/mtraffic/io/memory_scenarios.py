"""Three memory measurement scenarios used in Task 1 reporting.

Scenario A. Naive pandas: read every TSV column with default dtypes (object/int64/float64),
            no aggregation.
Scenario B. Selective pandas: read only the 3 columns of interest with explicit dtypes,
            but without column-wise downcast and without country code aggregation.
Scenario C. Optimized streaming: pyarrow streaming reader, column pruning at parse time,
            uint16 / int64 timestamp / float32 traffic, group by (square_id, ts) sum.

Each scenario returns the in-memory size of the resulting DataFrame or Table as well as
sample wall-clock and peak RSS via the memory monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from mtraffic.io.readers import SOURCE_COLUMNS, read_one_file
from mtraffic.utils.memory import PeakRSSMonitor


@dataclass(slots=True)
class ScenarioResult:
    scenario: str
    file: str
    rows: int
    final_df_mb: float
    peak_rss_mb: float
    end_rss_mb: float
    duration_s: float


def _df_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024**2)


def _table_memory_mb(table: pa.Table) -> float:
    return float(table.nbytes) / (1024**2)


def scenario_naive(path: Path) -> ScenarioResult:
    with PeakRSSMonitor() as mon:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=list(SOURCE_COLUMNS),
            na_values=[""],
            keep_default_na=True,
            low_memory=False,
        )
    rows = len(df)
    size = _df_memory_mb(df)
    del df
    return ScenarioResult("A_naive_pandas", str(path), rows, size, mon.sample.peak_rss_mb, mon.sample.end_rss_mb, mon.sample.duration_s)


def scenario_selective(path: Path) -> ScenarioResult:
    with PeakRSSMonitor() as mon:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[0, 1, 7],
            names=["square_id", "time_interval", "internet"],
            dtype={"square_id": "int64", "time_interval": "int64", "internet": "float64"},
            na_values=[""],
            keep_default_na=True,
        )
    rows = len(df)
    size = _df_memory_mb(df)
    del df
    return ScenarioResult("B_selective_pandas", str(path), rows, size, mon.sample.peak_rss_mb, mon.sample.end_rss_mb, mon.sample.duration_s)


def scenario_optimized(path: Path, tz: str = "Europe/Rome") -> ScenarioResult:
    with PeakRSSMonitor() as mon:
        table = read_one_file(path, tz=tz)
    rows = table.num_rows
    size = _table_memory_mb(table)
    return ScenarioResult("C_optimized_streaming", str(path), rows, size, mon.sample.peak_rss_mb, mon.sample.end_rss_mb, mon.sample.duration_s)


def measure_all(path: Path) -> list[ScenarioResult]:
    return [scenario_naive(path), scenario_selective(path), scenario_optimized(path)]
