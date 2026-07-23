"""Tests for the rule-based explanation fallback (no ANTHROPIC_API_KEY required)."""

import pytest

from anomaly_detector import (
    engineer_features, detect_anomalies, top_anomalies, explain_anomalies,
)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Force the rule-based path so these tests never hit the network/LLM."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_explain_anomalies_uses_rule_based_fallback(shocked_ohlcv):
    feat = engineer_features(shocked_ohlcv)
    anom = detect_anomalies(feat)
    top = top_anomalies(anom, n=5)
    results = explain_anomalies(top, feat, "TEST")

    assert len(results) == len(top)
    for r in results:
        assert r["explanation"]
        assert isinstance(r["signals"], list) and r["signals"]
        assert r["signal_count"] >= 1


def test_explain_anomalies_sorted_by_date(shocked_ohlcv):
    feat = engineer_features(shocked_ohlcv)
    anom = detect_anomalies(feat)
    top = top_anomalies(anom, n=5)
    results = explain_anomalies(top, feat, "TEST")
    dates = [r["date"] for r in results]
    assert dates == sorted(dates)
