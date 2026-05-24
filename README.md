# Milan Mobile Traffic: Comparative Time Series Analysis and Forecasting

Analysis and forecasting of Telecom Italia mobile internet traffic over the 100x100 grid of Milan, for the
period 2013-11-01 to 2013-12-30 (10 minute resolution). The pipeline covers:

1. Memory efficient ingestion of the daily TSV files into a partitioned Parquet store.
2. Exploratory data analysis and characterization of the dataset.
3. One step ahead forecasting of internet traffic in three areas for the week 2013-12-16 to 2013-12-22 using
   a SARIMA model (Fourier-ARIMA dynamic harmonic regression), a PyTorch LSTM and a PyTorch dilated 1D CNN
   (TCN-style).

## Layout

```
src/mtraffic/    importable package (io, transform, data, eda, models, eval, utils)
notebooks/       thin orchestrators for the report (01 memory, 02 EDA, 03 modeling)
configs/         YAML configuration (default.yaml, fast.yaml)
tests/unit/      unit tests
reports/         figures, tables, run artifacts
data/raw/        symlink or copy of dataverse_files (gitignored)
data/interim/    Parquet store produced by ingestion (gitignored)
```

Everything is invoked via the `mtraffic` CLI (a `typer` app exposed by the package) and
wrapped in the top-level `Makefile`. The notebooks call the same package functions; they
are reference views for graders and for assembling the report, not load-bearing entry
points.

## Requirements

Python 3.11. The recommended tool is [uv](https://docs.astral.sh/uv/).

```
uv venv
uv sync --all-extras
uv run pre-commit install
```

Place the 59 daily TSV files in `data/raw/`. On Linux and macOS, a symlink works:

```
ln -sf ../dataverse_files data/raw
```

The grid GeoJSON file `milano-grid.geojson` must be present at the repository root for the spatial figure.

## Running

```
make ingest       # build the Parquet store, measure memory
make eda          # produce all EDA figures and tables
make train        # train SARIMA, LSTM and CNN for the 3 areas
make forecast     # one step ahead walk forward over the test week
make eval         # tables, 9 forecast plots and comparison.md
make test         # unit tests
```

Single shot:

```
make all          # ingest, eda, train, forecast, eval
```

## Test Window

Training: 2013-11-01 to 2013-12-09.
Validation: 2013-12-10 to 2013-12-15.
Test: 2013-12-16 to 2013-12-22, used only to report the final metrics.

## Areas

Three independent forecasts are produced for: the area with the highest two month total internet traffic
(identified after ingestion), square id 4159 and square id 4556. The selected top area is recorded in
`reports/tables/area_top.txt` after `make ingest`.

## Hardware

Reference profile: Apple Silicon laptop, 16 GB RAM, CPU only. Optional GPU paths (CUDA, Apple MPS, TF Metal)
are auto detected and used when available. The reported numbers in the writeup are recorded from the CPU
baseline; any GPU acceleration is disclosed alongside.

## Video demonstration
https://drive.google.com/file/d/1i7PtgSZ729wQeA84nZ1MoiEoS58oHenH/view?usp=sharing

## License

MIT.

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Scientific Data 2, 150055 (2015).

[2] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse, V1, 2015. doi:10.7910/DVN/EGZHFV.

[3] Telecom Italia, "Milano Grid," Harvard Dataverse, V1, 2015. doi:10.7910/DVN/QJWLFU.

[4] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., OTexts, 2021. https://otexts.com/fpp3/.

[5] S. Bai, J. Z. Kolter and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.

[6] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation 9 (8), 1735-1780, 1997.

[7] D. A. Dickey and W. A. Fuller, "Distribution of the estimators for autoregressive time series with a unit root," JASA 74 (366), 427-431, 1979.

[8] D. Kwiatkowski, P. C. B. Phillips, P. Schmidt and Y. Shin, "Testing the null hypothesis of stationarity against the alternative of a unit root," J. Econometrics 54 (1-3), 159-178, 1992.

[9] R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning, "STL: A seasonal-trend decomposition procedure based on Loess," J. Official Statistics 6 (1), 3-73, 1990.

[10] X. Wang, K. Smith-Miles and R. J. Hyndman, "Characteristic-based clustering for time series data," Data Mining and Knowledge Discovery 13, 335-364, 2006.

[11] S. Seabold and J. Perktold, "statsmodels: Econometric and statistical modeling with Python," in *Proc. 9th SciPy Conf.*, 2010.

[12] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," NeurIPS 32, 2019.
