"""One-shot generator for the three orchestration notebooks.

This file lives under outputs/ (a temporary scratchpad). It is not part of the
repo and is not invoked by the pipeline. Run it once, copy the .ipynb files
into notebooks/, then discard.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def code(source: str) -> dict:
    lines = source.splitlines()
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": NB_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --- 01 Task 1 memory ---------------------------------------------------------

nb1 = make_notebook([
    md(
        "# Task 1. Data handling and memory management\n"
        "\n"
        "Thin orchestrator over `mtraffic.io.memory_scenarios` and the partitioned Parquet store.\n"
        "All heavy logic lives under `src/mtraffic/`; this notebook only calls into it and\n"
        "renders the headline numbers used in the report.\n"
        "\n"
        "Run `make ingest` first to populate `data/interim/`."
    ),
    code(
        "from pathlib import Path\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from mtraffic.config import Config\n"
        "from mtraffic.io.memory_scenarios import (\n"
        "    scenario_naive,\n"
        "    scenario_selective,\n"
        "    scenario_optimized,\n"
        ")\n"
        "\n"
        "cfg = Config.load()\n"
        "raw_dir = cfg.paths.raw_dir\n"
        "interim_dir = cfg.paths.interim_dir\n"
        "report_dir = cfg.paths.reports_dir"
    ),
    md(
        "## A representative daily file\n"
        "\n"
        "The three scenarios are measured on the same daily TSV so the comparison is fair.\n"
        "Each scenario is wrapped by `PeakRSSMonitor` which samples resident set size while\n"
        "the read runs."
    ),
    code(
        "candidates = sorted(raw_dir.glob('sms-call-internet-mi-*.txt'))\n"
        "assert candidates, f'No daily TSVs under {raw_dir}'\n"
        "day_file = candidates[len(candidates) // 2]\n"
        "print(f'Using {day_file.name} ({day_file.stat().st_size / 2**20:.1f} MB on disk)')"
    ),
    md(
        "## Run the three scenarios\n"
        "\n"
        "A. Naive pandas: every column at default dtypes.\n"
        "B. Selective pandas: 3 columns, explicit dtypes, no aggregation.\n"
        "C. Optimized streaming: pyarrow batches, column pruning, uint16/float32, group-by-sum.\n"
        "\n"
        "Note that scenario A allocates roughly 1.3 GB of RSS for a single day. On the 5 GB\n"
        "raw archive, the naive path would not fit in memory on most laptops."
    ),
    code(
        "results = [scenario_naive(day_file), scenario_selective(day_file), scenario_optimized(day_file)]\n"
        "df = pd.DataFrame([r.__dict__ for r in results])\n"
        "df[['scenario', 'rows', 'final_df_mb', 'peak_rss_mb', 'duration_s']].round(2)"
    ),
    md(
        "## Persisted report\n"
        "\n"
        "`make ingest` writes the same table to `reports/tables/memory_report.csv` so the\n"
        "report and the notebook stay in sync."
    ),
    code(
        "csv_path = report_dir / 'tables' / 'memory_report.csv'\n"
        "if csv_path.exists():\n"
        "    pd.read_csv(csv_path)\n"
        "else:\n"
        "    print(f'{csv_path} not found. Run: make ingest')"
    ),
    md(
        "## Parquet partition store\n"
        "\n"
        "Each daily file becomes one Parquet partition under\n"
        "`data/interim/year_month=YYYY-MM/day=YYYY-MM-DD/part.parquet`. The manifest\n"
        "records sha256 and row counts."
    ),
    code(
        "manifest = interim_dir / '_manifest.json'\n"
        "if manifest.exists():\n"
        "    import json\n"
        "    m = json.loads(manifest.read_text())\n"
        "    print(f\"Partitions: {len(m.get('partitions', []))}\")\n"
        "    print(f\"Total rows: {sum(p['rows'] for p in m.get('partitions', [])):,}\")\n"
        "    print(f\"Total bytes: {sum(p['bytes'] for p in m.get('partitions', [])) / 2**20:.1f} MB\")\n"
        "else:\n"
        "    print(f'{manifest} not found. Run: make ingest')"
    ),
])

# --- 02 Task 2 EDA -----------------------------------------------------------

nb2 = make_notebook([
    md(
        "# Task 2. Exploratory data analysis and time series characterisation\n"
        "\n"
        "All seven mandated EDA items: distribution, multi-area preview, rolling stats with\n"
        "stationarity tests, differencing, STL decomposition with strength scores, ACF/PACF,\n"
        "spatial heatmap, and anomalies. Each cell imports a function from `mtraffic.eda` and\n"
        "renders the resulting figure or summary table.\n"
        "\n"
        "Run `make ingest` first."
    ),
    code(
        "from datetime import datetime\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "\n"
        "from mtraffic.config import Config\n"
        "from mtraffic.data.loaders import area_totals, load_area_series\n"
        "from mtraffic.eda import (\n"
        "    anomalies,\n"
        "    correlation,\n"
        "    decomposition,\n"
        "    distributions,\n"
        "    spatial,\n"
        "    stationarity,\n"
        "    timeseries,\n"
        ")\n"
        "\n"
        "cfg = Config.load()\n"
        "INTERIM = cfg.paths.interim_dir\n"
        "FIG = cfg.paths.reports_dir / 'figures'\n"
        "FIG.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "# Focus area: square 5161 (the city-centre top area).\n"
        "AREA = 5161\n"
        "series = load_area_series(INTERIM, AREA)\n"
        "print(f'Area {AREA}: {len(series):,} observations from {series.index.min()} to {series.index.max()}')"
    ),
    md("## I. City-wide distribution"),
    code(
        "totals = area_totals(INTERIM)\n"
        "stats = distributions.plot_city_pdf(totals, FIG / 'city_pdf.png', annotate_areas=[5161, 4159, 4556])\n"
        "stats"
    ),
    code("display(Image(FIG / 'city_pdf.png'))"),
    md("## II. Three areas, first two weeks"),
    code(
        "timeseries.plot_three_areas_two_weeks(\n"
        "    INTERIM,\n"
        "    [5161, 4159, 4556],\n"
        "    FIG / 'three_area_2weeks.png',\n"
        "    start=datetime(2013, 11, 1),\n"
        "    days=14,\n"
        "    labels=['top (5161)', '4159', '4556'],\n"
        ")\n"
        "display(Image(FIG / 'three_area_2weeks.png'))"
    ),
    md("## III. Rolling mean and standard deviation, ADF and KPSS"),
    code(
        "rolling_stats = stationarity.plot_rolling(\n"
        "    series,\n"
        "    window=cfg.eda.rolling_window_steps,\n"
        "    out_png=FIG / f'rolling_{AREA}.png',\n"
        "    title=f'Rolling mean and std, area {AREA}',\n"
        ")\n"
        "rolling_stats"
    ),
    code("display(Image(FIG / f'rolling_{AREA}.png'))"),
    md("## IV. Differencing"),
    code(
        "diff1 = stationarity.plot_diff(series, lag=1, out_png=FIG / f'diff1_{AREA}.png', title='First difference')\n"
        "diff144 = stationarity.plot_diff(series, lag=144, out_png=FIG / f'diff144_{AREA}.png', title='Seasonal difference, lag 144')\n"
        "{'first': diff1, 'seasonal_144': diff144}"
    ),
    md("## V. STL decomposition with seasonal and trend strength"),
    code(
        "dec_daily = decomposition.stl_decompose(series, period=144)\n"
        "strengths_daily = decomposition.strengths(dec_daily)\n"
        "decomposition.plot_decomposition(dec_daily, FIG / f'decompose_{AREA}_daily.png', f'STL daily, area {AREA}')\n"
        "strengths_daily"
    ),
    code("display(Image(FIG / f'decompose_{AREA}_daily.png'))"),
    md("## VI. ACF and PACF"),
    code(
        "acf_summary = correlation.plot_acf_pacf(\n"
        "    series,\n"
        "    out_acf=FIG / f'acf_{AREA}.png',\n"
        "    out_pacf=FIG / f'pacf_{AREA}.png',\n"
        "    acf_lags=cfg.eda.acf_max_lag,\n"
        "    pacf_lags=cfg.eda.pacf_max_lag,\n"
        "    title_prefix=f'area {AREA}:',\n"
        ")\n"
        "acf_summary"
    ),
    code(
        "display(Image(FIG / f'acf_{AREA}.png'))\n"
        "display(Image(FIG / f'pacf_{AREA}.png'))"
    ),
    md("## VII. Spatial distribution"),
    code(
        "spatial_stats = spatial.plot_spatial(totals, FIG / 'spatial_heatmap.png')\n"
        "spatial_stats"
    ),
    code("display(Image(FIG / 'spatial_heatmap.png'))"),
    md("## VIII. Anomaly detection"),
    code(
        "outliers = anomalies.stl_residual_outliers(series, period=144)\n"
        "drops = anomalies.seasonal_naive_drops(series)\n"
        "anomalies.plot_anomalies(series, outliers, drops, FIG / f'anomalies_{AREA}.png', f'Anomalies, area {AREA}')\n"
        "print(f'STL residual outliers: {len(outliers)}')\n"
        "print(f'Seasonal-naive drop runs: {len(drops)}')\n"
        "outliers.head()"
    ),
    code("display(Image(FIG / f'anomalies_{AREA}.png'))"),
])

# --- 03 Task 3 models --------------------------------------------------------

nb3 = make_notebook([
    md(
        "# Task 3. Forecasting model design, training and evaluation\n"
        "\n"
        "Thin orchestrator: load the area series, train SARIMA (Fourier-ARIMA dynamic\n"
        "harmonic regression), an LSTM, and a dilated 1D CNN, then run a shared\n"
        "walk-forward evaluation across the test week (Dec 16 to 22, 2013).\n"
        "\n"
        "The full sweep across all three areas is run by `make train && make forecast && make eval`.\n"
        "This notebook demonstrates the pipeline on a single area for narrative purposes."
    ),
    code(
        "from datetime import datetime\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "\n"
        "from mtraffic.config import Config\n"
        "from mtraffic.data.loaders import load_area_series\n"
        "from mtraffic.eval import metrics, walkforward\n"
        "from mtraffic.eval.plots import plot_combined, plot_forecast\n"
        "from mtraffic.models.classical.sarima import SarimaModel\n"
        "from mtraffic.models.neural.cnn_torch import CnnConfig, CnnForecaster\n"
        "from mtraffic.models.neural.lstm_torch import LstmConfig, LstmForecaster\n"
        "from mtraffic.utils.seed import seed_all\n"
        "\n"
        "seed_all(42)\n"
        "\n"
        "cfg = Config.load()\n"
        "AREA = 5161\n"
        "TRAIN_END = cfg.eval.train_end\n"
        "TEST_START = cfg.eval.test_start\n"
        "TEST_END = cfg.eval.test_end\n"
        "\n"
        "series = load_area_series(cfg.paths.interim_dir, AREA)\n"
        "train_series = series.loc[:TRAIN_END]\n"
        "print(f'Area {AREA}: {len(series):,} obs, train ends {TRAIN_END}, test {TEST_START} to {TEST_END}')"
    ),
    md(
        "## SARIMA\n"
        "\n"
        "ARIMA(2,0,2) endogenous plus K=4 daily and K=3 weekly Fourier exogenous regressors.\n"
        "This is the dynamic harmonic regression approach from FPP3 chapter 9, chosen over a\n"
        "seasonal state space (s=144) which is intractable on CPU for our train length."
    ),
    code(
        "sarima = SarimaModel(order=(2, 0, 2), daily_terms=4, weekly_terms=3)\n"
        "sarima.fit(train_series)\n"
        "print('SARIMA fitted')"
    ),
    md(
        "## LSTM\n"
        "\n"
        "Two-layer LSTM, hidden size 64, sequence length 144 (one day). Time features\n"
        "(sin/cos of time-of-day, day-of-week, is_weekend) concatenated with the standardised\n"
        "traffic value."
    ),
    code(
        "lstm_cfg = LstmConfig(\n"
        "    seq_len=cfg.lstm.seq_len,\n"
        "    hidden=cfg.lstm.hidden,\n"
        "    layers=cfg.lstm.layers,\n"
        "    epochs=cfg.lstm.epochs,\n"
        "    batch_size=cfg.lstm.batch_size,\n"
        "    lr=cfg.lstm.lr,\n"
        ")\n"
        "lstm = LstmForecaster(lstm_cfg)\n"
        "lstm.fit(train_series)\n"
        "print('LSTM fitted')"
    ),
    md(
        "## Dilated 1D CNN (TCN-style)\n"
        "\n"
        "Dilated causal convolutions with dilation set {1, 2, 4, 8, 16, 32}, kernel 3,\n"
        "residual blocks. Receptive field reaches ~190 timesteps, enough to span a full day."
    ),
    code(
        "cnn_cfg = CnnConfig(\n"
        "    seq_len=cfg.cnn.seq_len,\n"
        "    channels=cfg.cnn.channels,\n"
        "    dilations=tuple(cfg.cnn.dilations),\n"
        "    kernel_size=cfg.cnn.kernel_size,\n"
        "    epochs=cfg.cnn.epochs,\n"
        "    batch_size=cfg.cnn.batch_size,\n"
        "    lr=cfg.cnn.lr,\n"
        ")\n"
        "cnn = CnnForecaster(cnn_cfg)\n"
        "cnn.fit(train_series)\n"
        "print('CNN fitted')"
    ),
    md(
        "## Walk-forward evaluation\n"
        "\n"
        "At each test timestamp t every model sees the actual history up to t (no leakage)\n"
        "and emits a one-step-ahead forecast. The protocol is identical across models."
    ),
    code(
        "results = {}\n"
        "for name, model in [('SARIMA', sarima), ('LSTM', lstm), ('CNN', cnn)]:\n"
        "    res = walkforward.run(model, series, TEST_START, TEST_END)\n"
        "    results[name] = res\n"
        "    m = metrics.compute_all(res.y_true, res.y_pred)\n"
        "    print(f'{name:7s}  MAE {m[\"mae\"]:8.2f}  RMSE {m[\"rmse\"]:8.2f}  MAPE {m[\"mape\"]:6.2f}%  sMAPE {m[\"smape\"]:6.2f}%')"
    ),
    md("## Metrics table (persisted by `make eval`)"),
    code(
        "tabs = cfg.paths.reports_dir / 'tables'\n"
        "metrics_csv = tabs / 'task3_metrics_all.csv'\n"
        "if metrics_csv.exists():\n"
        "    pd.read_csv(metrics_csv)\n"
        "else:\n"
        "    print(f'{metrics_csv} not found. Run: make eval')"
    ),
    md("## Forecast figures"),
    code(
        "figs = cfg.paths.reports_dir / 'figures'\n"
        "combined = figs / f'forecast_{AREA}_combined.png'\n"
        "if combined.exists():\n"
        "    display(Image(combined))\n"
        "else:\n"
        "    print(f'{combined} not found. Run: make forecast')"
    ),
    md(
        "## Failure analysis\n"
        "\n"
        "`reports/tables/failure_cases.csv` lists windows where every model simultaneously\n"
        "exceeded its own average MAE by 2x or more. Two of three areas show their worst\n"
        "window on Sunday Dec 22, the last day before Christmas week."
    ),
    code(
        "failure_csv = tabs / 'failure_cases.csv'\n"
        "if failure_csv.exists():\n"
        "    pd.read_csv(failure_csv)\n"
        "else:\n"
        "    print(f'{failure_csv} not found. Run: make eval')"
    ),
])


def write_notebook(path: Path, nb: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {path}")


import os
ROOT = Path(os.environ.get("PROJECT_ROOT", "/sessions/youthful-zealous-hopper/mnt/Time-series-forecasting"))
write_notebook(ROOT / "notebooks" / "01_task1_memory.ipynb", nb1)
write_notebook(ROOT / "notebooks" / "02_task2_eda.ipynb", nb2)
write_notebook(ROOT / "notebooks" / "03_task3_models.ipynb", nb3)
