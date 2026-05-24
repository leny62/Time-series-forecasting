"""CLI for the EDA stage."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from mtraffic.config import Config
from mtraffic.data.loaders import area_totals, load_area_series
from mtraffic.eda import anomalies, correlation, decomposition, distributions, spatial, stationarity, timeseries
from mtraffic.utils import logging as mtlog

eda_app = typer.Typer(add_completion=False, help="Exploratory data analysis and characterization.")
log = mtlog.get_logger(__name__)


def _resolve_areas(cfg: Config, areas_arg: str | None) -> dict[str, int]:
    """Resolve `top,4159,4556` to {label: int square_id}, with `top` picked from the store."""
    spec = areas_arg or ",".join(str(a) for a in cfg.eda.areas)
    out: dict[str, int] = {}
    top_value: int | None = None
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "top":
            if top_value is None:
                totals = area_totals(cfg.paths.interim_dir)
                top_value = int(totals.idxmax())
                (cfg.paths.reports_dir / "tables").mkdir(parents=True, exist_ok=True)
                (cfg.paths.reports_dir / "tables" / "area_top.txt").write_text(f"{top_value}\n", encoding="utf-8")
            out[f"top({top_value})"] = top_value
        else:
            out[token] = int(token)
    return out


@eda_app.command(name="all")
def run_all(
    interim_dir: Path = typer.Option(None, "--interim-dir"),
    geojson: Path = typer.Option(None, "--geojson"),
    areas: str = typer.Option(None, "--areas", help="Comma separated, e.g. top,4159,4556."),
    report_dir: Path = typer.Option(None, "--report-dir"),
    config: Path = typer.Option(None, "--config"),
    skip_overview: bool = typer.Option(False, "--skip-overview", help="Skip city PDF, spatial heatmap and three-area preview."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip per-area figures whose PNG already exists."),
) -> None:
    cfg = Config.load(config) if config else Config.load()
    if interim_dir is not None:
        cfg.paths.interim_dir = interim_dir.resolve()
    if report_dir is not None:
        cfg.paths.reports_dir = report_dir.resolve()
    if geojson is not None:
        cfg.paths.geojson = geojson.resolve()
    figs = cfg.paths.reports_dir / "figures"
    tabs = cfg.paths.reports_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    selected = _resolve_areas(cfg, areas)
    log.info("EDA on areas: %s", selected)
    summary: dict[str, object] = {"areas": selected}

    if not skip_overview:
        # T2.I distribution
        log.info("Distribution: computing area totals and plotting city PDF")
        totals = area_totals(cfg.paths.interim_dir)
        stats_d = distributions.plot_city_pdf(totals, figs / "city_pdf.png", annotate_areas=list(selected.values()))
        summary["city_pdf"] = stats_d

        # T2.VI spatial
        log.info("Spatial: plotting heatmap")
        sp = spatial.plot_spatial(totals, figs / "spatial_heatmap.png")
        summary["spatial"] = sp

        # T2.II three-area first 14 days
        log.info("Time series: three areas, first 14 days")
        series_start = datetime(2013, 11, 1)
        timeseries.plot_three_areas_two_weeks(
            cfg.paths.interim_dir,
            list(selected.values()),
            figs / "three_area_2weeks.png",
            start=series_start,
            days=cfg.eda.preview_days,
            labels=list(selected.keys()),
        )

    def _skip_fig(p: Path) -> bool:
        return skip_existing and p.exists()

    per_area_summary: dict[str, dict] = {}
    for label, area in selected.items():
        log.info("Per-area EDA for %s (square %d)", label, area)
        s = load_area_series(cfg.paths.interim_dir, area)
        per_area: dict[str, object] = {"square_id": area, "n_obs": int(len(s))}

        # Stationarity: always compute ADF and KPSS; only skip the PNG render.
        rolling_png = figs / f"rolling_{area}.png"
        if not _skip_fig(rolling_png):
            per_area["stationarity"] = stationarity.plot_rolling(
                s, window=cfg.eda.rolling_window_steps, out_png=rolling_png,
                title=f"Rolling mean/std, area {area}",
            )
        else:
            per_area["stationarity"] = {**stationarity.adf_summary(s), **{f"kpss_{k}": v for k, v in stationarity.kpss_summary(s).items()}}

        for lag, name in [(1, "adf_after_diff"), (144, "adf_after_seasonal_diff")]:
            png = figs / f"diff{lag}_{area}.png"
            if not _skip_fig(png):
                per_area[name] = stationarity.plot_diff(
                    s, lag=lag, out_png=png,
                    title=("First difference" if lag == 1 else "Seasonal difference (lag 144)") + f", area {area}",
                )
            else:
                per_area[name] = {"adf_stat_diff": stationarity.adf_summary(s.diff(lag).dropna())["statistic"], "adf_p_diff": stationarity.adf_summary(s.diff(lag).dropna())["pvalue"]}

        for period, tag in [(cfg.eda.rolling_window_steps, "daily"), (1008, "weekly")]:
            dec_png = figs / f"decompose_{area}_{tag}.png"
            try:
                dec = decomposition.stl_decompose(s, period=period)
                per_area[f"strengths_{tag}"] = decomposition.strengths(dec)
                if not _skip_fig(dec_png):
                    decomposition.plot_decomposition(dec, dec_png, f"STL decomposition, area {area}, period={period}")
            except Exception as exc:
                log.warning("STL decomposition failed for area %d period %d: %s", area, period, exc)

        acf_png = figs / f"acf_{area}.png"
        pacf_png = figs / f"pacf_{area}.png"
        if not _skip_fig(acf_png) or not _skip_fig(pacf_png):
            per_area["correlation"] = correlation.plot_acf_pacf(
                s, out_acf=acf_png, out_pacf=pacf_png,
                acf_lags=cfg.eda.acf_max_lag, pacf_lags=cfg.eda.pacf_max_lag,
                title_prefix=f"area {area}:",
            )

        anom_png = figs / f"anomalies_{area}.png"
        outliers = anomalies.stl_residual_outliers(s, period=cfg.eda.rolling_window_steps)
        drops = anomalies.seasonal_naive_drops(s)
        if not _skip_fig(anom_png):
            anomalies.plot_anomalies(s, outliers, drops, anom_png, f"Anomalies, area {area}")
        outliers.to_csv(tabs / f"anomalies_{area}.csv", index=False)
        per_area["outliers_count"] = int(len(outliers))
        per_area["drop_runs_count"] = int(len(drops))

        per_area_summary[str(area)] = per_area

    summary["per_area"] = per_area_summary

    summary_path = cfg.paths.reports_dir / "eda_summary.json"
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        prev_areas = previous.get("per_area", {}) if isinstance(previous, dict) else {}
        merged = previous.copy() if isinstance(previous, dict) else {}
        merged_areas = dict(prev_areas)
        merged_areas.update(summary["per_area"])
        merged["per_area"] = merged_areas
        # Overview keys overwrite only when this run actually produced them.
        for key in ("city_pdf", "spatial"):
            if key in summary:
                merged[key] = summary[key]
        merged["areas"] = dict({**previous.get("areas", {}), **summary["areas"]}) if isinstance(previous.get("areas"), dict) else summary["areas"]
        summary = merged
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    typer.echo(json.dumps({"figures_dir": str(figs), "summary": str(summary_path)}))
