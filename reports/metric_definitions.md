# Evaluation Metrics: Formulas and Interpretation

All metrics are computed on the original (un-standardized) scale. Let `y_t` be the actual
internet activity at 10 minute bin `t` and `yhat_t` the corresponding one-step-ahead
prediction. `N` is the number of test points (1008 = 7 days x 144).

## Mean Absolute Error (MAE)

```
MAE = (1 / N) * sum_{t=1..N} |y_t - yhat_t|
```

MAE expresses the average error in the original unit of the data (number of call detail
records per 10 minute bin). It is robust to large but rare errors because it does not
square them. A drop in MAE means the typical step-by-step error is smaller.

## Root Mean Squared Error (RMSE)

```
RMSE = sqrt( (1 / N) * sum_{t=1..N} (y_t - yhat_t)^2 )
```

RMSE penalizes large errors more heavily than small ones because of the squared term. When
RMSE is much larger than MAE, a few unusually bad predictions dominate the error budget,
which often correlates with anomaly periods or regime shifts. We report both because the
gap between them is itself an indicator of how concentrated the errors are.

## Mean Absolute Percentage Error (MAPE)

```
MAPE = (1 / N) * sum_{t=1..N} ( |y_t - yhat_t| / max(|y_t|, epsilon) ) * 100
```

MAPE expresses the error as a percentage of the actual value. It is scale-free, which lets
us compare across areas with very different traffic magnitudes (square 5161 has roughly 5x
the activity of square 4159). We protect the denominator with `epsilon = 1e-3` to avoid
divisions by near-zero during late-night quiet minutes. MAPE is asymmetric: it punishes
over-prediction less than under-prediction in absolute terms because the denominator uses
the actual value only.

## Symmetric Mean Absolute Percentage Error (sMAPE)

```
sMAPE = (1 / N) * sum_{t=1..N} ( |y_t - yhat_t| / ((|y_t| + |yhat_t|) / 2) ) * 100
```

sMAPE corrects MAPE's asymmetry by averaging the actual and predicted in the denominator.
It is bounded in [0, 200] percent (Hyndman, FPP3, chapter 5). We report it alongside MAPE
when MAPE could be misleading near zero values; if sMAPE and MAPE roughly agree, the MAPE
number is trustworthy.

## How metrics are computed in this pipeline

1. After each model finishes training, `src/mtraffic/eval/walkforward.py` rolls a one-step
   ahead forecast across the test window (Dec 16 to Dec 22). At each timestamp `t` the model
   sees actual history up to `t - 1` and predicts `y_t`. There is no lookahead and the test
   window never enters training, validation, or scaler fitting.
2. The driver returns `(timestamps, y_true, y_pred)` as a `WalkForwardResult`. Forecasts are
   persisted to `reports/tables/forecasts/<model>_<area>.parquet`.
3. `src/mtraffic/eval/metrics.compute_all` returns `MAE`, `RMSE`, `MAPE_percent`,
   `sMAPE_percent`. The function code is the canonical formula; the table above mirrors it.
4. Per-area CSVs (`task3_metrics_<area>.csv`) and a combined CSV
   (`task3_metrics_all.csv`) are written by `mtraffic eval task3`.

## Reading the numbers in context

For this dataset, MAE alone is the cleanest model selection signal because the series is
strictly non-negative and the magnitude is meaningful (CDR counts). MAPE is the cleanest
cross-area comparator because it normalizes for traffic volume. RMSE is the cleanest
indicator of "blow-up" risk: when RMSE / MAE is large, the model has rare but severe
errors that show up in the failure-window analysis.
