"""Command line entry point for the mtraffic pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from mtraffic import __version__
from mtraffic.config import Config
from mtraffic.utils import hardware, logging as mtlog, seed

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Milan mobile traffic pipeline.")


def _load_config(config: Path | None) -> Config:
    return Config.load(config) if config else Config.load()


@app.callback()
def main(
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config; defaults to configs/default.yaml."),
    log_level: str = typer.Option("info", "--log-level", help="debug, info, warning, error."),
) -> None:
    mtlog.configure(level=log_level)  # type: ignore[arg-type]
    cfg = _load_config(config)
    seed.set_all(cfg.seed)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command(name="hwinfo")
def hwinfo(out: Path | None = typer.Option(None, "--out", help="Write hardware JSON to this path.")) -> None:
    """Print and optionally save hardware and library metadata."""
    info = hardware.collect().as_dict()
    payload = json.dumps(info, indent=2)
    typer.echo(payload)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")


# Subcommand groups for ingest, eda, train, forecast, eval are registered by their modules in later
# milestones. Each module appends to this `app` via `app.add_typer(...)`.

try:
    from mtraffic.io.cli_ingest import ingest_app

    app.add_typer(ingest_app, name="ingest")
except ImportError:
    pass

try:
    from mtraffic.eda.cli_eda import eda_app

    app.add_typer(eda_app, name="eda")
except ImportError:
    pass

try:
    from mtraffic.models.cli_train import train_app

    app.add_typer(train_app, name="train")
except ImportError:
    pass

try:
    from mtraffic.eval.cli_eval import forecast_app, eval_app

    app.add_typer(forecast_app, name="forecast")
    app.add_typer(eval_app, name="eval")
except ImportError:
    pass


if __name__ == "__main__":
    app()
