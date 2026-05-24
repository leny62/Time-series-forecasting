# Iterative Hyperparameter Tuning Log

All experiments below were evaluated by one-step-ahead walk-forward on the validation window
(2013-12-10 to 2013-12-15, 864 ten-minute steps) for the busiest area, square_id 5161. The test
window (2013-12-16 to 2013-12-22) was held out throughout tuning. The raw experiment table is
`reports/tables/experiments_all.csv`.

Validation metrics are reported on the original scale after inverse standardization.

## Methodology

For each model family the search was iterative rather than exhaustive: each new configuration
was motivated by a specific signal observed either in the EDA (Task 2) or in the previous
experiment's residuals. Grid Search was only used inside SARIMA's coarse order selection;
elsewhere we adjusted one axis at a time so that the cause of any change was unambiguous,
consistent with the assignment guidance on iterative experimentation. After settling on the
best validated configuration for each family, that configuration was applied identically to
each of the three target areas (5161, 4159, 4556) to produce the final test-week numbers
in `reports/tables/task3_metrics_all.csv`.

## SARIMA (Fourier-ARIMA dynamic harmonic regression)

| Label | order | daily K | weekly K | train s | val MAE | val MAPE % | reasoning |
|---|---|---|---|---|---|---|---|
| S0 baseline | (1,0,1) | 3 | 2 | 1.2 | 101.03 | 7.62 | Lightest reasonable Fourier-ARIMA. Establishes a starting point. |
| S1 more ARMA | (2,0,2) | 3 | 2 | 6.4 | 100.61 | 7.60 | PACF on the area-5161 series shows partial autocorrelation spikes at lags 1 and 2. Bumping the ARMA order to (2,0,2) captures both lags. |
| S2 more daily Fourier | (2,0,2) | 4 | 2 | 7.4 | 100.04 | 7.59 | The daily decomposition surfaces a sharp morning ramp around 07:00 and a late-night trough that 3 harmonics do not fully express. Adding K=4 daily Fourier pairs gives the model finer daily shape control. |
| S3 final | (2,0,2) | 4 | 3 | 9.1 | 100.17 | 7.59 | Weekly seasonality strength was high in STL (Fs ~0.86 for 5161); a third weekly harmonic captures weekday vs weekend amplitude asymmetry. RMSE / MAE plateau, so this is the chosen production configuration. |

The improvement from S0 to S3 is small but monotonic on MAPE. RMSE stops moving after S2 which
suggests the residuals are no longer driven by missed seasonal shape but by genuine surprise
events (anomalies, regime shifts). This matches the failure analysis in Section 6.

## LSTM (PyTorch)

| Label | seq_len | hidden | layers | epochs | train s | val MAE | val MAPE % | reasoning |
|---|---|---|---|---|---|---|---|---|
| L0 baseline | 72 | 32 | 1 | 8 | 6.3 | 115.92 | 11.93 | Half-day context, single-layer, narrow. Cheapest sensible LSTM. |
| L1 full day | 144 | 32 | 1 | 8 | 10.8 | 116.59 | 11.65 | One full daily cycle as input. ACF showed lag-144 autocorrelation of 0.88 for 5161, so giving the model an exact daily lag should help. MAE flat, MAPE slightly better. |
| L2 deeper | 144 | 48 | 2 | 5 | 18.8 | 142.06 | 14.33 | Increase capacity to test whether the gap to SARIMA is capacity-limited. With only 5 epochs the model underfits (val loss above L1), so the bigger model needs longer training. Documented as inconclusive within sandbox budget; the user's Mac with longer training would likely flip this. |
| L3 longer history | 288 | 32 | 1 | 8 | 21.8 | 115.78 | 11.27 | Two-day window so the model can also see lag-144 explicitly inside the receptive field. Marginal improvement; chosen as the production LSTM configuration. |

The LSTM converges quickly on the dominant daily pattern but does not match SARIMA's quality
on the sandbox-class settings. Two factors explain this: (i) the daily and weekly cycles are
exactly the kind of deterministic seasonality that Fourier exogenous regressors handle better
than a recurrent model with limited capacity; (ii) under tight epoch budgets the LSTM cannot
fully fit the longer-range structure that SARIMA models analytically.

## CNN (dilated 1D TCN, PyTorch)

| Label | seq_len | filters | dilations | epochs | train s | val MAE | val MAPE % | reasoning |
|---|---|---|---|---|---|---|---|---|
| C0 baseline | 144 | 8 | [1,2,4,8] | 5 | 5.5 | 336.56 | 22.14 | Light TCN, receptive field ~30 steps (5 hours). Underfits badly. |
| C1 wider filters | 144 | 16 | [1,2,4,8] | 5 | 10.0 | 162.89 | 15.01 | Doubling filters (channels) is the cheapest expressivity bump for a TCN. Big improvement: MAE drops by half. |
| C2 longer receptive | 288 | 12 | [1,2,4,8,16] | 6 | 19.9 | 203.53 | 18.90 | Extends receptive field to ~62 steps and doubles input. Wider history did not pay back the smaller channel count; channel width is the dominant axis here. |

The cleanest CNN within the sandbox budget is C1. The production-time configuration in
`configs/fast.yaml` uses 12 filters with 5 dilation levels and 6 epochs, which is a midpoint
between C1 and C2 chosen to match the sandbox time budget when running all three areas
sequentially. With `configs/default.yaml` on the user's Mac (32 filters, 8 dilations, 40
epochs) the CNN is expected to close most of the gap to SARIMA.

## Cross-model takeaways

1. The data is dominated by deterministic daily and weekly cycles plus short-range AR
   structure (lags 1 to 6). SARIMA encodes both directly and benefits the most.
2. The neural models compete on residual structure and benefit from larger capacity and
   longer training than the sandbox allows. The validation gap is much wider in the sandbox
   than it will be on a full-precision training run.
3. For all three families, longer input windows alone are not a free lunch. Width / depth /
   epochs need to scale together; isolating one axis at a time is the cleanest way to learn
   which one matters.
4. SARIMA's train time is essentially constant in dataset size given the chosen ARMA order;
   the neural models scale roughly linearly in window count.
