"""Tests for feature engineering (anomaly_detector.engineer_features)."""
import numpy as np

from anomaly_detector import engineer_features


EXPECTED_COLUMNS = {
    "return_pct", "log_return", "volume_zscore", "volume_multiple",
    "realized_vol", "vol_zscore", "return_zscore",
}


def test_adds_expected_columns(clean_ohlcv):
    out = engineer_features(clean_ohlcv)
    assert EXPECTED_COLUMNS.issubset(out.columns)


def test_drops_warmup_rows_with_no_nans(clean_ohlcv):
    out = engineer_features(clean_ohlcv)
    # engineer_features() ends with dropna(), so no NaNs should remain
    assert not out[list(EXPECTED_COLUMNS)].isna().any().any()
    # and it must have shed rows to do so (rolling windows need warmup)
    assert len(out) < len(clean_ohlcv)


def test_return_pct_matches_close_pct_change(clean_ohlcv):
    out = engineer_features(clean_ohlcv)
    recomputed = clean_ohlcv["Close"].pct_change() * 100
    aligned = recomputed.loc[out.index]
    assert np.allclose(out["return_pct"], aligned, atol=1e-9)


def test_realized_vol_is_nonnegative(clean_ohlcv):
    out = engineer_features(clean_ohlcv)
    assert (out["realized_vol"] >= 0).all()


def test_volume_multiple_is_positive(clean_ohlcv):
    out = engineer_features(clean_ohlcv)
    assert (out["volume_multiple"] > 0).all()
