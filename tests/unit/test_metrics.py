import numpy as np

from mtraffic.eval.metrics import compute_all, mae, mape_percent, rmse, smape_percent


def test_mae_simple() -> None:
    y = np.array([1.0, 2.0, 3.0])
    yh = np.array([1.0, 2.0, 4.0])
    assert mae(y, yh) == 1 / 3


def test_rmse_simple() -> None:
    y = np.array([1.0, 1.0, 1.0])
    yh = np.array([1.0, 1.0, 4.0])
    assert rmse(y, yh) == np.sqrt(3.0)


def test_mape_protects_near_zero() -> None:
    y = np.array([0.0, 0.0, 1.0])
    yh = np.array([0.1, 0.2, 1.0])
    out = mape_percent(y, yh, epsilon=1e-3)
    assert np.isfinite(out)
    assert out > 0


def test_smape_symmetry() -> None:
    y = np.array([1.0, 2.0])
    yh = np.array([2.0, 1.0])
    assert smape_percent(y, yh) == smape_percent(yh, y)


def test_compute_all_returns_dict() -> None:
    out = compute_all(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert set(out.keys()) == {"MAE", "RMSE", "MAPE_percent", "sMAPE_percent"}
