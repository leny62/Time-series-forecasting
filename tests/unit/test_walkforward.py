from datetime import datetime

import numpy as np
import pandas as pd

from mtraffic.eval.walkforward import run
from mtraffic.models.baselines.naive import LastValue, SeasonalNaive


def _make_series(n: int = 50) -> pd.Series:
    idx = pd.date_range("2013-12-15 00:00", periods=n, freq="10min", tz="Europe/Rome")
    return pd.Series(np.arange(n, dtype=float), index=idx, name="x")


def test_walkforward_last_value_is_correct() -> None:
    s = _make_series(30)
    tz = s.index.tz
    res = run(LastValue(), s, datetime(2013, 12, 15, 1, 0).replace(tzinfo=tz), datetime(2013, 12, 15, 2, 0).replace(tzinfo=tz))
    # Each y_pred at ts should equal the actual value at ts-10min.
    for i, ts in enumerate(res.timestamps):
        prev_ts = ts - pd.Timedelta(minutes=10)
        assert res.y_pred[i] == s.loc[prev_ts]


def test_walkforward_seasonal_naive_with_short_history_falls_back() -> None:
    s = _make_series(30)
    tz = s.index.tz
    res = run(SeasonalNaive(period=144), s, datetime(2013, 12, 15, 1, 0).replace(tzinfo=tz), datetime(2013, 12, 15, 1, 30).replace(tzinfo=tz))
    # period=144 not available in 30-step history; falls back to last value.
    for i, ts in enumerate(res.timestamps):
        prev_ts = ts - pd.Timedelta(minutes=10)
        assert res.y_pred[i] == s.loc[prev_ts]
