"""Tests for the z-score + IQR anomaly ensemble (anomaly_detector.detect_anomalies)."""
import pandas as pd

from anomaly_detector import (
    engineer_features, detect_anomalies, top_anomalies, _iqr_flags,
)


def test_iqr_flags_marks_only_extreme_values():
    series = pd.Series([1, 2, 2, 3, 2, 3, 2, 100, -100, 2])
    flags = _iqr_flags(series)
    assert flags.iloc[7] and flags.iloc[8]        # the two extreme outliers
    assert not flags.iloc[0:7].any()               # everything else is quiet


def test_detect_anomalies_adds_expected_columns(clean_ohlcv):
    feat = engineer_features(clean_ohlcv)
    anom = detect_anomalies(feat)
    for col in ("flag_return", "flag_volume", "flag_vol", "severity",
                "is_anomaly", "signal_count"):
        assert col in anom.columns


def test_detect_anomalies_severity_zero_when_not_flagged(clean_ohlcv):
    feat = engineer_features(clean_ohlcv)
    anom = detect_anomalies(feat)
    quiet = anom[~anom["is_anomaly"]]
    assert (quiet["severity"] == 0).all()


def test_shocked_series_flags_more_anomalies_than_clean(clean_ohlcv, shocked_ohlcv):
    clean_anom = detect_anomalies(engineer_features(clean_ohlcv))
    shocked_anom = detect_anomalies(engineer_features(shocked_ohlcv))
    assert shocked_anom["is_anomaly"].sum() > clean_anom["is_anomaly"].sum()


def test_top_anomalies_respects_n_and_is_sorted_by_severity(shocked_ohlcv):
    anom = detect_anomalies(engineer_features(shocked_ohlcv))
    top = top_anomalies(anom, n=3)
    assert len(top) <= 3
    assert list(top["severity"]) == sorted(top["severity"], reverse=True)
    assert "Date" in top.columns


def test_signal_count_matches_number_of_true_flags(shocked_ohlcv):
    anom = detect_anomalies(engineer_features(shocked_ohlcv))
    flagged = anom[anom["is_anomaly"]]
    expected = flagged[["flag_return", "flag_volume", "flag_vol"]].sum(axis=1)
    assert (flagged["signal_count"] == expected).all()
