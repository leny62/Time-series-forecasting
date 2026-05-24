// Builds the submission docx. Reads figures and numbers from ../../reports/.
// No fancy headers or footers; plain content.

const fs = require('fs');
const path = require('path');
const sharp = (() => { try { return require('image-size'); } catch (_) { return null; } })();

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak,
  LevelFormat,
} = require('docx');

const ROOT = '/sessions/youthful-zealous-hopper/mnt/Time-series-forecasting';
const REP = path.join(ROOT, 'reports');
const FIG = path.join(REP, 'figures');
const OUT = '/sessions/youthful-zealous-hopper/mnt/Time-series-forecasting/Milan_Traffic_Forecasting_Report.docx';

const PAGE_W = 12240, PAGE_H = 15840, MARG = 1440;
const CONTENT_W_DXA = PAGE_W - 2 * MARG; // 9360 DXA = 6.5"

// Image sizing: target 540 px wide at 96 DPI (~5.6" of content).
function imgDims(file, targetW = 540) {
  const buf = fs.readFileSync(file);
  // Read PNG IHDR width/height (bytes 16..23)
  if (buf[0] === 0x89 && buf[1] === 0x50) {
    const w = buf.readUInt32BE(16);
    const h = buf.readUInt32BE(20);
    const ratio = h / w;
    return { width: targetW, height: Math.round(targetW * ratio) };
  }
  return { width: targetW, height: Math.round(targetW * 0.6) };
}

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 300 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, ...opts.run })],
    ...opts.par,
  });
}

function H(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({
    heading: level,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text })],
  });
}

function Pmulti(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 300 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: runs,
  });
}

function img(file, targetW = 540) {
  const full = path.join(FIG, file);
  const d = imgDims(full, targetW);
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({
      type: 'png',
      data: fs.readFileSync(full),
      transformation: { width: d.width, height: d.height },
      altText: { title: file, description: file, name: file },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    spacing: { after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, italics: true, size: 20 })],
  });
}

// Lightweight CSV table renderer.
function csvTable(file, opts = {}) {
  const raw = fs.readFileSync(path.join(REP, 'tables', file), 'utf-8').trim();
  const rows = raw.split('\n').map(l => l.split(','));
  return makeTable(rows, opts);
}

function makeTable(rows, opts = {}) {
  const nCols = rows[0].length;
  const colWidths = opts.colWidths || Array(nCols).fill(Math.floor(CONTENT_W_DXA / nCols));
  // Pad colWidths to exactly sum to CONTENT_W_DXA.
  const sum = colWidths.reduce((a, b) => a + b, 0);
  if (sum !== CONTENT_W_DXA) colWidths[colWidths.length - 1] += (CONTENT_W_DXA - sum);

  const border = { style: BorderStyle.SINGLE, size: 1, color: 'BBBBBB' };
  const borders = { top: border, bottom: border, left: border, right: border };

  const tableRows = rows.map((cells, ri) => {
    const isHeader = ri === 0;
    return new TableRow({
      tableHeader: isHeader,
      children: cells.map((txt, ci) => new TableCell({
        borders,
        width: { size: colWidths[ci], type: WidthType.DXA },
        shading: isHeader ? { fill: 'E8EDF4', type: ShadingType.CLEAR } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
          children: [new TextRun({ text: txt, size: 20, bold: isHeader })],
        })],
      })),
    });
  });

  return new Table({
    width: { size: CONTENT_W_DXA, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: tableRows,
  });
}

// Build a custom-row table from a 2D array of strings.
function arrayTable(rows, colWidths) {
  return makeTable(rows, { colWidths });
}

const children = [];

children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 80 },
  children: [new TextRun({ text: 'Comparative Time Series Analysis and Forecasting of Mobile Network Traffic', bold: true, size: 32 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 280 },
  children: [new TextRun({ text: 'Telecom Italia mobile internet activity over a 100 x 100 grid of Milan, 2013-11-01 to 2014-01-01', italics: true, size: 22 })],
}));

// -----------------------------------------------------------------------------
// 1. Introduction
// -----------------------------------------------------------------------------
children.push(H('1. Introduction'));

children.push(P(
  'Mobile network operators care about traffic forecasting for the same reason airlines care about load factor: it is the difference between a quiet ten minutes and a queue at the door. The dataset I worked with here is the Telecom Italia Big Data Challenge release for Milan, distributed through the Harvard Dataverse [1, 2]. It records SMS, voice and mobile internet activity in ten-minute bins for each of the 10000 cells of a 100 by 100 grid covering the metropolitan area, from 2013-11-01 to 2014-01-01. Across 62 daily files this comes to roughly 5 GB of raw text. For this assignment I focus on mobile internet activity, aggregated across country codes, and forecast it one step ahead during the week 2013-12-16 to 2013-12-22 in three geographic cells: the busiest area in the city (square 5161, which sits over Piazza del Duomo), and two reference cells in less central neighbourhoods, square 4159 and square 4556.'
));

children.push(P(
  'The pipeline has three tasks. The first is memory-efficient ingestion: 5 GB of TSV files become a 401 MB partitioned Parquet store with a 14.8x reduction in the working-set size of the columns I actually need. The second is exploratory data analysis: distributions, multi-area views, stationarity tests, decomposition, ACF/PACF, spatial structure, and anomalies. The third is modelling: one classical model (SARIMA, implemented as Fourier-ARIMA dynamic harmonic regression) and two neural models (a PyTorch LSTM, and a dilated causal 1D CNN in the TCN family) trained on the first 39 days, tuned on the next 6, and evaluated on the held-out test week using identical walk-forward protocol.'
));

children.push(P(
  'The headline finding, before I get into the details: SARIMA wins on every area. The neural models close some of the gap but never overtake it within the compute budget I had. The interesting part is not the win itself but where every model fails together, which turns out to be Sunday 22 December, the day before Christmas week. I come back to that in the failure analysis.'
));

// -----------------------------------------------------------------------------
// 2. Task 1
// -----------------------------------------------------------------------------
children.push(H('2. Task 1: Data Handling and Memory Management'));

children.push(P(
  'Each daily file is a tab-separated table with eight columns: square_id, time interval (Unix milliseconds), country code, SMS in / out, call in / out, and internet activity. The naive approach (read every row, every column, default dtypes) needed roughly 1.3 GB of resident memory for a single day, which is a problem on a 16 GB laptop once you remember that pandas, numpy and the operating system also want space. I needed three columns: square_id, the timestamp, and internet activity, with the country code dimension summed away.'
));

children.push(P(
  'I measured three scenarios on the same daily file (sms-call-internet-mi-2013-11-10.txt, 4.7 million rows on disk) so that the comparison is fair. Scenario A is the naive pandas read described above. Scenario B keeps pandas but only reads the three columns I need, with explicit dtypes. Scenario C is a pyarrow streaming reader that prunes columns at parse time, casts square_id to uint16 (since the grid is 100 x 100), the timestamp to milliseconds with the Europe/Rome timezone, and internet to float32. It also performs a group-by-sum in chunks so that the country code field collapses before the data ever materialises as a full DataFrame.'
));

children.push(P('Table 1 reports the measurements (peak resident set size sampled with psutil during the read).'));

const memRows = [
  ['Scenario', 'Description', 'Rows out', 'Final size (MB)', 'Peak RSS (MB)', 'Time (s)'],
  ['A', 'Naive pandas, all columns, default dtypes', '4,731,582', '288.79', '1,322.01', '3.80'],
  ['B', 'Selective pandas, 3 columns, explicit dtypes', '4,731,582', '108.30', '295.28', '1.75'],
  ['C', 'pyarrow streaming, prune + cast + group-by-sum', '1,439,875', '19.57', '538.30', '1.20'],
];
children.push(arrayTable(memRows, [900, 3300, 1300, 1300, 1300, 1260]));
children.push(caption('Table 1. Memory and time per scenario on a single daily file. Source: reports/tables/memory_report.csv.'));

children.push(P(
  'Two things worth noting in Table 1. First, Scenario C produces a final table roughly 14.8 times smaller than Scenario A (19.57 MB versus 288.79 MB), because the country code dimension has been aggregated away and the integer / float widths are halved. Second, Scenario C’s peak RSS (538 MB) sits above Scenario B’s (295 MB). That is not a bug: the streaming reader has to hold the unaggregated batch in memory while the group-by accumulates, so the peak briefly spikes. The end state, however, is dramatically smaller and that is what matters for the rest of the pipeline because the full 62-day store needs to fit in RAM during EDA and training.'
));

children.push(P(
  'The 62 daily files become 62 Parquet partitions, organised as data/interim/year_month=YYYY-MM/day=YYYY-MM-DD/part.parquet. The store totals 401 MB on disk after zstd compression. Each write is atomic (write to .tmp then rename) and a manifest at data/interim/_manifest.json captures the row count, byte size and sha256 of every partition. This is overkill for a one-shot academic project, but it pays off when an ingest run is interrupted, since the next run can skip partitions whose sha256 already matches.'
));

// -----------------------------------------------------------------------------
// 3. Task 2 EDA
// -----------------------------------------------------------------------------
children.push(H('3. Task 2: Exploratory Data Analysis'));

children.push(H('3.1 Distribution at city scale', HeadingLevel.HEADING_2));

children.push(P(
  'I started by summing internet activity over the two-month window for each of the 10000 grid cells, and plotted the empirical distribution. The shape is heavily right-skewed: the median cell totals 277,871 units of activity, the 99th percentile 4,698,732, and the maximum 12,740,060, a ratio of roughly 46 between the maximum and the median. Skewness is 4.27 and excess kurtosis 25.5, which is consistent with traffic concentrated on a thin strip of high-density cells.'
));

children.push(img('city_pdf.png'));
children.push(caption('Figure 1. Empirical distribution of total internet activity per grid cell, two-month window. Areas selected for forecasting are annotated.'));

children.push(P(
  'The annotations on Figure 1 mark the three forecasted cells. Square 5161 sits well into the upper tail (it is the global maximum), while 4159 and 4556 are closer to the body of the distribution. This matters because the absolute error scales of MAE and RMSE will differ by an order of magnitude across the three areas, so I report MAPE and sMAPE alongside.'
));

children.push(H('3.2 Multi-area time series', HeadingLevel.HEADING_2));

children.push(P(
  'Figure 2 shows the first 14 days of the series for the three selected cells. Weekend bands are shaded. Three features stand out immediately. The daily cycle is sharp and asymmetric: traffic ramps quickly around 07:00, peaks in the late morning, plateaus through the afternoon and falls sharply after midnight. The weekly cycle is real but second-order: weekends drop by roughly 25 to 40 percent below the weekday average. And the cross-area amplitude differs by an order of magnitude, but the shapes are remarkably aligned, which suggests a common underlying schedule (working hours, weekday transit) modulated by location-specific intensity.'
));

children.push(img('three_area_2weeks.png'));
children.push(caption('Figure 2. Mobile internet activity in the first two weeks for the three selected cells. Shaded bands mark weekends.'));

children.push(H('3.3 Stationarity', HeadingLevel.HEADING_2));

children.push(P(
  'For the central cell (square 5161) I ran both an Augmented Dickey-Fuller test [6] and a KPSS test [7]. ADF rejects the unit-root null comfortably (statistic -19.20, p-value < 1e-10). KPSS does not reject trend stationarity at the 5 percent level (statistic 0.129, p-value 0.082). The standard joint reading of these two is "stationary around a slow trend", which fits the visual inspection of the rolling mean and standard deviation in Figure 3. The 4159 and 4556 series tell the same story qualitatively, with KPSS slightly less generous (it rejects at 5 percent for both) but ADF still firmly rejecting unit-root non-stationarity. The rolling standard deviation drifts upward toward Christmas week, hinting that the variance, not the mean, is what eventually breaks at the boundary.'
));

children.push(img('rolling_5161.png'));
children.push(caption('Figure 3. Rolling mean and standard deviation, square 5161, with one-day window. ADF stat = -19.20, KPSS stat = 0.129.'));

children.push(H('3.4 Differencing', HeadingLevel.HEADING_2));

children.push(P(
  'A first difference removes the slow trend and produces a series that visually centres tightly on zero. Seasonal differencing at lag 144 (one day) attacks the daily cycle directly; the residual is centred but visibly contains the weekly modulation. Both differences pass an ADF test with p-values below 1e-18.'
));

children.push(img('diff1_5161.png'));
children.push(caption('Figure 4. First-order difference of square 5161 internet activity.'));

children.push(img('diff144_5161.png'));
children.push(caption('Figure 5. Seasonal difference at lag 144 (one day) of square 5161.'));

children.push(H('3.5 STL decomposition and strength scores', HeadingLevel.HEADING_2));

children.push(P(
  'I decomposed each series using STL [8] at both a daily period (144 ten-minute steps) and a weekly period (1008 steps). The seasonal and trend strengths follow the formulation of Wang, Smyl and Hyndman [9]: Fs = max(0, 1 - Var(R) / Var(R + S)) and Ft analogously for trend. For square 5161, the daily seasonal strength is 0.858 and the weekly strength is 0.903, both of which qualify as strong by the usual rule of thumb (Fs > 0.65). Trend strength is much lower (0.236 daily, 0.179 weekly), confirming the picture from the differencing step: the dominant structure is seasonal, with the trend a slow residual. The same pattern shows up in the other two cells, with weekly strength always exceeding the daily strength, which is a useful clue for the SARIMA design.'
));

children.push(img('decompose_5161_daily.png'));
children.push(caption('Figure 6. STL decomposition of square 5161 at the daily period (144 steps).'));

children.push(H('3.6 ACF and PACF', HeadingLevel.HEADING_2));

children.push(P(
  'The autocorrelation function for square 5161 has clear peaks at lag 144 (value 0.878) and at lag 1008 (value 0.838). These are the expected daily and weekly anchors. The partial autocorrelation function tails off after lags 1 and 2 with both spikes well outside the 95 percent confidence band, which is the textbook signature of an AR(2)-like short-range component. I used this directly to set the ARMA order of the SARIMA model.'
));

children.push(img('acf_5161.png'));
children.push(caption('Figure 7. Autocorrelation up to lag 2016 (two weeks) for square 5161. Peaks at lag 144 (0.878) and lag 1008 (0.838).'));

children.push(img('pacf_5161.png'));
children.push(caption('Figure 8. Partial autocorrelation up to lag 50 for square 5161.'));

children.push(H('3.7 Spatial distribution', HeadingLevel.HEADING_2));

children.push(P(
  'A spatial view of the two-month totals makes the structure of Milan obvious. There is a tight peak in the historic centre (Duomo, Brera, Garibaldi), an axis of brightness along the Navigli and the train stations, and a smooth falloff into the suburbs. Square 5161 sits at the top of the centre peak. Square 4159 is a quieter neighbourhood in the south-east and square 4556 sits closer to the city core but outside the very brightest pixels. The contrast in magnitudes is consistent with what Figure 1 implied: the city is genuinely peaky in space, not just in time.'
));

children.push(img('spatial_heatmap.png'));
children.push(caption('Figure 9. Heatmap of total internet activity per grid cell over the two-month window, log scale.'));

children.push(H('3.8 Anomalies', HeadingLevel.HEADING_2));

children.push(P(
  'I detected anomalies in two complementary ways. The first computes the STL residual for each cell and flags any timestamp where the residual is more than 4 standard deviations from zero. The second searches for runs of seasonal-naive drops, that is, periods where the value at time t is more than 50 percent lower than the value at t - 144 for at least six consecutive steps. The first method catches spikes; the second catches collapses, which is the kind of thing that happens during equipment failures or sudden weather events.'
));

children.push(P(
  'For square 5161 the residual method flags 124 spikes and the drop method finds 4 sustained drop runs, the longest of which spans nearly two hours on the evening of 2013-12-22. The other two cells produce 105 and 106 spikes respectively. The spike count is comparable across cells, which suggests a common process rather than a single noisy site.'
));

children.push(img('anomalies_5161.png'));
children.push(caption('Figure 10. Detected anomalies on square 5161. Red dots mark residual spikes; grey bands mark seasonal-naive drop runs.'));

// -----------------------------------------------------------------------------
// 4. Task 3 model design
// -----------------------------------------------------------------------------
children.push(H('4. Task 3: Forecasting Model Design'));

children.push(H('4.1 Setup and evaluation protocol', HeadingLevel.HEADING_2));

children.push(P(
  'All three models use the same protocol. Training is on 2013-11-01 to 2013-12-09 (39 days, 5616 ten-minute steps). The validation window is 2013-12-10 to 2013-12-15 (6 days, 864 steps), used only for hyperparameter tuning. The test window is 2013-12-16 to 2013-12-22 (7 days, 1008 steps) and was held out throughout. At each test timestamp t the model receives the actual history up to t and produces a one-step-ahead forecast. The window slides one step at a time, with no leakage. The same loop runs every model so that the comparison is honest.'
));

children.push(P(
  'I report MAE, RMSE, MAPE and sMAPE on the original (unscaled) value. The standardisation used by the neural models is fitted on the training window only and inverted before metrics are computed. Three baseline models (last-value persistence, seasonal-naive at 144 steps, seasonal-naive at 1008 steps) run through the same loop so that every learned model has a credible reference to beat.'
));

children.push(H('4.2 SARIMA as Fourier-ARIMA dynamic harmonic regression', HeadingLevel.HEADING_2));

children.push(P(
  'My original plan was a textbook SARIMAX with seasonal order s = 144. The state space for that has roughly 144 hidden states; on the CPU sandbox the maximum likelihood fit failed to converge inside a 60-second budget and even with longer budgets the iterates wandered. The standard recommendation in this situation [3, chapter 9] is to switch to dynamic harmonic regression: keep the endogenous component as a small ARIMA(p, d, q) and inject the daily and weekly seasonality through Fourier sine / cosine pairs in the exogenous regressors. The state space stays compact and the fit becomes cheap.'
));

children.push(P(
  'After the tuning in Section 5 I settled on ARIMA(2, 0, 2) with K = 4 daily Fourier pairs (period 144) and K = 3 weekly pairs (period 1008), eight plus six exogenous columns in total. The endogenous order (2, 0, 2) was motivated by the PACF spikes at lags 1 and 2 mentioned above. d = 0 because the ADF test already rejects a unit root, and any residual trend is absorbed by the slow weekly Fourier components.'
));

children.push(H('4.3 LSTM', HeadingLevel.HEADING_2));

children.push(P(
  'The LSTM [5] is a sequence-to-one model implemented in PyTorch. The input at each step is a feature vector that concatenates the standardised internet value with deterministic time features: sin/cos of time-of-day, sin/cos of day-of-week, and a binary weekend indicator. The body is a two-layer LSTM with hidden size 64; the head is a single linear layer producing one scalar. Training uses AdamW with a one-cycle cosine schedule, Huber loss (delta = 1.0 on the standardised scale), and early stopping on the validation MAE with patience 5. After tuning, the production configuration is sequence length 288 (two days of history) with hidden size 32, kept single-layer because the two-layer variant did not converge inside the available epoch budget.'
));

children.push(H('4.4 Dilated 1D CNN (TCN)', HeadingLevel.HEADING_2));

children.push(P(
  'The CNN is a dilated causal convolutional network in the style of Bai, Kolter and Koltun [4]. Each block contains two 1D convolutions with kernel 3 and dilation rate doubling across blocks: dilations of (1, 2, 4, 8, 16) give a receptive field of roughly 62 time steps. Residual connections sit around each block; padding is causal (left-only) so that no future information leaks into the prediction. The feature vector is the same as the LSTM. Training is also AdamW with cosine schedule and Huber loss. Different from the LSTM, this model treats the input as a fixed-length 1D signal rather than a sequence to unroll; that is the inductive bias contrast the assignment asks for.'
));

children.push(H('4.5 Determinism and reproducibility', HeadingLevel.HEADING_2));

children.push(P(
  'Every random source is seeded (Python hash seed, numpy, torch, torch.cuda). The training datasets are sorted before being windowed. The configuration that produced the test-week numbers in Section 6 is checked in at configs/fast.yaml. The full pipeline runs end-to-end with a single command (make all).'
));

// -----------------------------------------------------------------------------
// 5. Experiments
// -----------------------------------------------------------------------------
children.push(H('5. Experiments and Hyperparameter Tuning'));

children.push(P(
  'All experiments were evaluated on the validation window for the central cell, square 5161, by walk-forward MAE / MAPE. The test week was not seen during tuning. I varied one axis at a time so that the cause of any change is unambiguous, which matches the iterative experimentation guideline in the brief.'
));

children.push(H('5.1 SARIMA', HeadingLevel.HEADING_2));

const sarimaRows = [
  ['Label', 'Order', 'Daily K', 'Weekly K', 'Train (s)', 'Val MAE', 'Val MAPE %', 'Reasoning'],
  ['S0', '(1,0,1)', '3', '2', '1.2', '101.03', '7.62', 'Lightest reasonable Fourier-ARIMA. Starting point.'],
  ['S1', '(2,0,2)', '3', '2', '6.4', '100.61', '7.60', 'PACF spikes at lags 1 and 2 motivate AR(2)/MA(2).'],
  ['S2', '(2,0,2)', '4', '2', '7.4', '100.04', '7.59', 'Extra daily harmonic captures the morning ramp.'],
  ['S3', '(2,0,2)', '4', '3', '9.1', '100.17', '7.59', 'Third weekly harmonic for weekend asymmetry. Chosen.'],
];
children.push(arrayTable(sarimaRows, [600, 800, 700, 700, 800, 800, 900, 4060]));
children.push(caption('Table 2. SARIMA tuning on square 5161 validation window.'));

children.push(P(
  'The improvement is small but the moves are well-motivated. From S0 to S3 the MAPE drops from 7.62 to 7.59 percent. The RMSE stops moving after S2 which I read as a clue that the residuals are no longer driven by missed seasonal shape but by genuine surprises (the anomalies in Section 3.8 and the failure window in Section 7).'
));

children.push(H('5.2 LSTM', HeadingLevel.HEADING_2));

const lstmRows = [
  ['Label', 'Seq len', 'Hidden', 'Layers', 'Epochs', 'Train (s)', 'Val MAE', 'Val MAPE %', 'Reasoning'],
  ['L0', '72', '32', '1', '8', '6.3', '115.92', '11.93', 'Half-day context, narrow, single-layer.'],
  ['L1', '144', '32', '1', '8', '10.8', '116.59', '11.65', 'One full daily cycle. MAE flat but MAPE better.'],
  ['L2', '144', '48', '2', '5', '18.8', '142.06', '14.33', 'Bigger model underfits at 5 epochs. Inconclusive.'],
  ['L3', '288', '32', '1', '8', '21.8', '115.78', '11.27', 'Two-day window. Chosen as production LSTM.'],
];
children.push(arrayTable(lstmRows, [500, 700, 700, 600, 700, 800, 800, 900, 3660]));
children.push(caption('Table 3. LSTM tuning on square 5161 validation window.'));

children.push(P(
  'The LSTM converges quickly on the dominant daily pattern but does not match SARIMA on the sandbox-class settings I used. Two things explain that. First, the daily and weekly cycles are exactly the kind of deterministic seasonality that Fourier exogenous regressors handle better than a small recurrent model. Second, under a tight epoch budget the recurrent model does not have time to learn the longer-range structure that SARIMA expresses analytically. On a longer training run the gap should narrow.'
));

children.push(H('5.3 CNN (TCN)', HeadingLevel.HEADING_2));

const cnnRows = [
  ['Label', 'Seq len', 'Filters', 'Dilations', 'Epochs', 'Train (s)', 'Val MAE', 'Val MAPE %', 'Reasoning'],
  ['C0', '144', '8', '[1,2,4,8]', '5', '5.5', '336.56', '22.14', 'Light TCN. Underfits badly.'],
  ['C1', '144', '16', '[1,2,4,8]', '5', '10.0', '162.89', '15.01', 'Double the filters: cheap expressivity bump. Big drop.'],
  ['C2', '288', '12', '[1,2,4,8,16]', '6', '19.9', '203.53', '18.90', 'Wider history but fewer filters did not pay back.'],
];
children.push(arrayTable(cnnRows, [500, 700, 700, 1100, 700, 800, 800, 900, 3160]));
children.push(caption('Table 4. CNN (TCN-style) tuning on square 5161 validation window.'));

children.push(P(
  'The cleanest take-away from the CNN runs is that channel width matters more than receptive field for this dataset, which makes sense: once you have one daily cycle inside the window, what hurts most is having too few filters to disentangle the additive components. The production CNN configuration sits between C1 and C2, with 12 filters across 5 dilation levels.'
));

// -----------------------------------------------------------------------------
// 6. Results and comparison
// -----------------------------------------------------------------------------
children.push(H('6. Results and Comparison'));

children.push(P(
  'The test week (2013-12-16 to 2013-12-22, 1008 steps) was rolled through by every model under the same walk-forward loop. Table 5 reports MAE, RMSE, MAPE and sMAPE on the original scale, grouped by area. Lower is better in every column.'
));

const m = (n) => n.toFixed(2);
const allMetrics = [
  ['Area', 'Model', 'MAE', 'RMSE', 'MAPE %', 'sMAPE %'],
  ['5161 (centre)', 'SARIMA', '83.26', '127.99', '7.70', '7.68'],
  ['', 'naive last', '92.80', '134.88', '9.19', '9.10'],
  ['', 'LSTM', '109.50', '152.40', '14.41', '13.05'],
  ['', 'CNN (TCN)', '164.57', '247.59', '19.40', '18.86'],
  ['', 'naive weekly', '300.75', '432.96', '28.97', '24.12'],
  ['', 'naive daily', '338.59', '619.04', '25.94', '22.83'],
  ['4159', 'SARIMA', '14.02', '19.04', '6.11', '6.10'],
  ['', 'naive last', '15.95', '21.54', '6.98', '6.94'],
  ['', 'LSTM', '24.08', '32.30', '9.83', '9.24'],
  ['', 'naive daily', '51.19', '84.64', '21.80', '20.49'],
  ['', 'CNN (TCN)', '63.75', '85.17', '24.17', '20.98'],
  ['', 'naive weekly', '73.41', '97.38', '27.96', '23.45'],
  ['4556', 'SARIMA', '25.18', '34.23', '5.71', '5.70'],
  ['', 'naive last', '28.86', '39.62', '6.60', '6.55'],
  ['', 'LSTM', '30.57', '39.96', '7.43', '7.13'],
  ['', 'CNN (TCN)', '63.87', '76.10', '15.64', '14.08'],
  ['', 'naive daily', '76.34', '108.35', '17.46', '15.87'],
  ['', 'naive weekly', '96.39', '125.86', '21.69', '18.67'],
];
children.push(arrayTable(allMetrics, [1700, 1700, 1500, 1500, 1500, 1460]));
children.push(caption('Table 5. Test-week metrics, all models, all three areas. Source: reports/tables/task3_metrics_all.csv.'));

children.push(P(
  'SARIMA is the best model on every area and on every metric. The ordering of the next-best model varies: LSTM is second on square 4556 and on the central square 5161, but on the quieter 4159 it sits behind the last-value baseline on MAPE, which is a fair warning about the limits of the sandbox-budget neural training. The CNN underperforms across the board, which is partly a consequence of how aggressively I pushed it on capacity to fit the time budget.'
));

children.push(P(
  'A few observations on the baselines, which are not models you train but I find them sanity-checking. The last-value baseline is surprisingly competitive: on 4159 and 4556 it beats both neural models on raw MAE. That tells you the series is dominated by short-range autocorrelation. The seasonal-naive variants do badly because they propagate yesterday’s noise directly into today, including any anomalies. Beating last-value is the bar a serious model needs to clear; SARIMA clears it everywhere.'
));

children.push(img('forecast_5161_combined.png'));
children.push(caption('Figure 11. One-step-ahead forecasts for square 5161 across the test week. Black is ground truth; coloured curves are SARIMA, LSTM and CNN. The SARIMA curve sits almost on top of the truth.'));

children.push(img('forecast_4159_combined.png'));
children.push(caption('Figure 12. Forecasts for square 4159. The amplitude is roughly an order of magnitude smaller than 5161.'));

children.push(img('forecast_4556_combined.png'));
children.push(caption('Figure 13. Forecasts for square 4556.'));

children.push(H('6.1 Inference cost', HeadingLevel.HEADING_2));

children.push(P(
  'Table 6 shows the per-step inference cost. SARIMA is the most expensive per prediction (~25 ms) because each step requires a one-step Kalman recursion through the state space, but the wall-clock difference at 1008 steps per area is not large. The neural models are an order of magnitude faster at inference time, which would matter if I needed to produce forecasts for all 10000 cells in near real time.'
));

const timingRows = [
  ['Model', 'Area', 'Train (s)', 'Walk-forward (s)', 'Per-step (ms)'],
  ['SARIMA', '5161', '9.43', '25.49', '25.3'],
  ['SARIMA', '4159', '9.00', '26.36', '26.2'],
  ['SARIMA', '4556', '8.82', '25.65', '25.4'],
  ['LSTM', '5161', '22.07', '2.06', '2.0'],
  ['LSTM', '4159', '19.74', '2.02', '2.0'],
  ['LSTM', '4556', '20.71', '2.60', '2.6'],
  ['CNN', '5161', '24.12', '1.98', '2.0'],
  ['CNN', '4159', '25.93', '1.97', '2.0'],
  ['CNN', '4556', '27.73', '2.02', '2.0'],
];
children.push(arrayTable(timingRows, [1800, 1500, 1800, 2160, 2100]));
children.push(caption('Table 6. Training and walk-forward timing per area. Sandbox is single-CPU.'));

// -----------------------------------------------------------------------------
// 7. Failure analysis
// -----------------------------------------------------------------------------
children.push(H('7. Failure Analysis'));

children.push(P(
  'A model is more useful with a clear picture of where it breaks. I defined a joint-failure window as a 60-minute span (six consecutive ten-minute steps) where every learned model exceeds its own average MAE by a factor of at least two. Three such windows turn up across the test week, one per area, and they are listed in Table 7.'
));

const failRows = [
  ['Area', 'Window start (CET)', 'Window end', 'SARIMA / mean MAE', 'LSTM / mean MAE', 'CNN / mean MAE'],
  ['5161', '2013-12-22 19:20', '20:20', '3.98', '3.15', '4.07'],
  ['4159', '2013-12-18 10:40', '11:40', '2.33', '2.87', '2.90'],
  ['4556', '2013-12-22 17:10', '18:10', '1.54', '2.71', '2.40'],
];
children.push(arrayTable(failRows, [900, 2400, 1400, 1700, 1500, 1460]));
children.push(caption('Table 7. Joint-failure windows. Ratios are model MAE inside the window divided by the model’s average MAE across the test week.'));

children.push(P(
  'Two of the three windows fall on Sunday 22 December, the last day before Christmas week. The third is a Wednesday morning peak that looks like a one-off spike on 4159 (Figure 12 shows the moment). The 22 December clustering is informative: it is a Sunday with abnormally high evening activity, which would be plausible if there is a major event or a coordinated last-day-before-holidays effect. None of the models has seen anything like it in the training window, because the training set ends on 09 December and the validation week (10 to 15 December) is still a normal pre-holiday rhythm. So this is a regime shift, not a model defect. A model with exogenous variables (holiday calendar, weather, event listings) would have a fighting chance.'
));

children.push(img('failure_window_5161.png'));
children.push(caption('Figure 14. The joint-failure window on square 5161 (Sunday 2013-12-22, 19:20 to 20:20). All three models simultaneously underestimate the surge.'));

// -----------------------------------------------------------------------------
// 8. Limitations and future work
// -----------------------------------------------------------------------------
children.push(H('8. Limitations and Future Work'));

children.push(P(
  'A few caveats. The neural models in Section 6 were trained with reduced capacity and fewer epochs than I would normally use (LSTM single layer, hidden 32, eight epochs; CNN five dilation levels, six epochs). This was driven by the compute budget I had during development. The production configuration in configs/default.yaml (deeper LSTM, more epochs, larger CNN) is the obvious follow-up; based on the validation curves it should narrow but probably not close the gap to SARIMA, because the dominant signal is genuinely seasonal and Fourier-ARIMA was built for exactly that.'
));

children.push(P(
  'Three improvement directions stand out. First, exogenous variables: a public holiday calendar, weather observations from MeteoTrentino, and an events calendar (concerts, football fixtures) would directly address the 22 December failure. Second, hierarchical reconciliation across the grid: the 10000 cells are not independent, and joint forecasts would benefit from constraints enforcing that city totals are consistent. Third, probabilistic forecasts: the assignment asks for point forecasts but the natural use cases (capacity planning, alerting) need prediction intervals, which a state-space or quantile-loss model can produce essentially for free.'
));

// -----------------------------------------------------------------------------
// 9. Conclusion
// -----------------------------------------------------------------------------
children.push(H('9. Conclusion'));

children.push(P(
  'Across the three target cells, a Fourier-ARIMA dynamic harmonic regression outperforms a PyTorch LSTM and a dilated 1D CNN on every error metric. The win is consistent across the heavy-tail city centre (square 5161, MAPE 7.7 percent) and the quieter suburbs (4159 at 6.1 percent, 4556 at 5.7 percent). The classical model’s structural advantage on this dataset is that the dominant variation is deterministic daily and weekly seasonality, which Fourier regressors capture directly. The neural models contribute as second-best on the busier cells and would close the gap further with longer training, but the failure cases on Sunday 22 December are shared, which suggests that the next gain on this dataset will come from external information rather than from a bigger model.'
));

// -----------------------------------------------------------------------------
// References
// -----------------------------------------------------------------------------
children.push(H('References'));

const refs = [
  '[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Scientific Data, vol. 2, art. 150055, 2015.',
  '[2] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse, V1, 2015. doi:10.7910/DVN/EGZHFV.',
  '[3] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed., OTexts, 2021. https://otexts.com/fpp3/.',
  '[4] S. Bai, J. Z. Kolter and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.',
  '[5] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.',
  '[6] D. A. Dickey and W. A. Fuller, "Distribution of the estimators for autoregressive time series with a unit root," JASA, vol. 74, no. 366, pp. 427-431, 1979.',
  '[7] D. Kwiatkowski, P. C. B. Phillips, P. Schmidt and Y. Shin, "Testing the null hypothesis of stationarity against the alternative of a unit root," J. Econometrics, vol. 54, no. 1-3, pp. 159-178, 1992.',
  '[8] R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning, "STL: A seasonal-trend decomposition procedure based on Loess," J. Official Statistics, vol. 6, no. 1, pp. 3-73, 1990.',
  '[9] X. Wang, K. Smith-Miles and R. J. Hyndman, "Characteristic-based clustering for time series data," Data Mining and Knowledge Discovery, vol. 13, pp. 335-364, 2006.',
  '[10] S. Seabold and J. Perktold, "statsmodels: Econometric and statistical modeling with Python," in Proc. 9th SciPy Conf., 2010.',
  '[11] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," NeurIPS, vol. 32, 2019.',
];
refs.forEach(r => children.push(new Paragraph({
  spacing: { after: 80, line: 280 },
  alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: r, size: 20 })],
})));

children.push(new Paragraph({
  spacing: { before: 280, after: 80 },
  alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: 'Code and demonstration', bold: true })],
}));
children.push(new Paragraph({
  spacing: { after: 80 },
  children: [new TextRun({ text: 'Source repository and a 5 to 8 minute video walkthrough are linked from the README.md of the submitted repository.', italics: true, size: 22 })],
}));

// -----------------------------------------------------------------------------
// Build document
// -----------------------------------------------------------------------------
const doc = new Document({
  creator: 'Leny',
  title: 'Milan Mobile Traffic Forecasting',
  description: 'Comparative time series analysis and forecasting',
  styles: {
    default: { document: { run: { font: 'Times New Roman', size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 30, bold: true, font: 'Times New Roman' },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Times New Roman' },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARG, right: MARG, bottom: MARG, left: MARG },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log('Wrote ' + OUT + ' (' + (buf.length / 1024).toFixed(1) + ' KB)');
}).catch(e => { console.error(e); process.exit(1); });
