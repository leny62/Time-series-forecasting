"""CLI for the ingestion stage."""

from __future__ import annotations

import csv
import json
from datetime import date as _date
from pathlib import Path

import typer
from tqdm import tqdm

from mtraffic.config import Config
from mtraffic.data.catalog import (
    PartitionRecord,
    discover_raw,
    expected_dates,
    scan_existing_partitions,
    write_manifest,
    write_missing,
)
from mtraffic.io.memory_scenarios import measure_all
from mtraffic.io.paths import partition_path
from mtraffic.io.readers import read_one_file
from mtraffic.io.writers import sha256_of_file, write_daily_parquet
from mtraffic.utils import logging as mtlog
from mtraffic.utils.hardware import collect as hw_collect

ingest_app = typer.Typer(add_completion=False, help="Build the Parquet store from raw daily TSVs.")
log = mtlog.get_logger(__name__)


def _write_memory_report(report_dir: Path, results: list, day_file: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "tables" / "memory_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "file", "rows", "final_df_mb", "peak_rss_mb", "end_rss_mb", "duration_s"])
        for r in results:
            w.writerow([r.scenario, r.file, r.rows, f"{r.final_df_mb:.3f}", f"{r.peak_rss_mb:.3f}", f"{r.end_rss_mb:.3f}", f"{r.duration_s:.3f}"])
    return csv_path


def _ingest_one(raw_path: Path, interim_dir: Path, day: _date, tz: str) -> PartitionRecord:
    table = read_one_file(raw_path, tz=tz)
    out_path = write_daily_parquet(table, interim_dir, day)
    return PartitionRecord(
        day=day.isoformat(),
        path=str(out_path.relative_to(interim_dir)),
        rows=table.num_rows,
        bytes=out_path.stat().st_size,
        sha256=sha256_of_file(out_path),
    )


@ingest_app.callback(invoke_without_command=True)
def run(
    raw_dir: Path = typer.Option(None, "--raw-dir", help="Directory containing the daily TSVs."),
    out_dir: Path = typer.Option(None, "--out-dir", help="Output Parquet store directory."),
    report_dir: Path = typer.Option(None, "--report-dir", help="Directory for memory and hardware reports."),
    measure_memory: bool = typer.Option(True, "--measure-memory/--no-measure-memory", help="Run 3-scenario memory comparison on the first available file."),
    limit_days: int = typer.Option(0, "--limit-days", help="Process only the first N daily files (0 for all)."),
    force: bool = typer.Option(False, "--force", help="Re-ingest days even if their Parquet partition already exists."),
    config: Path = typer.Option(None, "--config", help="Override config YAML path."),
) -> None:
    """Build the canonical Parquet store from raw TSVs."""
    cfg = Config.load(config) if config else Config.load()
    raw = (raw_dir or cfg.paths.raw_dir).resolve()
    interim = (out_dir or cfg.paths.interim_dir).resolve()
    reports = (report_dir or cfg.paths.reports_dir).resolve()
    tz = cfg.ingest.tz
    interim.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    files = discover_raw(raw)
    if not files:
        typer.secho(f"No daily TSV files found under {raw}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    start = min(files)
    end = max(files)
    log.info("Found %d raw files spanning %s to %s in %s", len(files), start, end, raw)

    if measure_memory:
        sample_day = sorted(files)[0]
        sample_path = files[sample_day]
        log.info("Running 3-scenario memory benchmark on %s", sample_path.name)
        results = measure_all(sample_path)
        out_csv = _write_memory_report(reports, results, sample_path)
        log.info("Memory report written to %s", out_csv)

    expected = expected_dates(start, end)
    existing_days = {p.day: p for p in scan_existing_partitions(interim, expected)}
    to_process: list[tuple[_date, Path]] = []
    for day in sorted(files):
        if not force and partition_path(interim, day).exists():
            continue
        to_process.append((day, files[day]))
    if limit_days > 0:
        to_process = to_process[:limit_days]

    log.info("Ingesting %d daily file(s); %d already present", len(to_process), len(existing_days))
    for day, path in tqdm(to_process, desc="ingest", unit="day"):
        try:
            rec = _ingest_one(path, interim, day, tz=tz)
            existing_days[rec.day] = rec
            log.debug("Wrote %s (%d rows, %.2f MB)", rec.path, rec.rows, rec.bytes / 1024 / 1024)
        except Exception as exc:
            log.error("Failed to ingest %s: %s", path, exc)
            raise

    records = sorted(existing_days.values(), key=lambda r: r.day)
    write_manifest(interim, records, expected_start=start, expected_end=end)
    present_dates = {_date.fromisoformat(r.day) for r in records}
    missing = [d for d in expected if d not in present_dates]
    write_missing(interim, missing)
    hw = hw_collect().as_dict()
    (reports / "hardware.json").write_text(json.dumps(hw, indent=2), encoding="utf-8")
    log.info("Ingestion complete: %d partitions present, %d missing", len(records), len(missing))
    typer.echo(json.dumps({"partitions": len(records), "missing": len(missing), "interim_dir": str(interim)}))
