from mtraffic.config import Config


def test_default_config_loads() -> None:
    cfg = Config.load()
    assert cfg.seed == 20251201
    assert cfg.eval.test_start.year == 2013
    assert cfg.eval.test_end.year == 2013
    assert cfg.models.sarima.order == (2, 0, 2)
    assert cfg.models.sarima.daily_terms == 4
    assert cfg.models.sarima.weekly_terms == 3
    assert cfg.models.lstm.seq_len == 288
    assert cfg.models.cnn.seq_len == 1008
