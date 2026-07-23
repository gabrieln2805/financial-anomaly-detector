"""Tests for dashboard_builder.py: payload serialization and safe HTML embedding."""
import json

from anomaly_detector import engineer_features, detect_anomalies, top_anomalies, explain_anomalies
from dashboard_builder import company_to_js_payload, build_multi_dashboard, _safe_json


def _payload(df, ticker="TEST", company="Test Co", sector="Technology"):
    feat = engineer_features(df)
    anom = detect_anomalies(feat)
    top = top_anomalies(anom, n=5)
    expls = explain_anomalies(top, feat, ticker)
    return company_to_js_payload(ticker, company, sector, feat, anom, expls)


def test_payload_has_expected_shape(shocked_ohlcv):
    payload = _payload(shocked_ohlcv)
    for key in ("ticker", "name", "sector", "color", "dates", "closes",
                "anomDates", "anomScatter", "sigCounts", "kpis", "explanations"):
        assert key in payload
    assert payload["kpis"]["nTotal"] == len(payload["dates"])
    assert payload["kpis"]["nAnom"] == len(payload["anomDates"])


def test_payload_is_json_serializable(shocked_ohlcv):
    payload = _payload(shocked_ohlcv)
    # will raise if anything (e.g. numpy types) leaked through un-serialized
    json.dumps(payload)


def test_safe_json_escapes_script_breakout():
    dangerous = {"name": "</script><script>alert(1)</script>"}
    out = _safe_json(dangerous)
    assert "</script>" not in out
    assert "\\u003c/script\\u003e" in out
    # still valid JSON once unescaped back through JS \uXXXX semantics
    assert json.loads(json.dumps(dangerous)) == dangerous


def test_build_multi_dashboard_embeds_safe_json_not_raw(shocked_ohlcv):
    payload = _payload(shocked_ohlcv, ticker="XSS", company="</script><script>alert(1)</script>")
    out = build_multi_dashboard([payload], period="1y", generated_at="today")
    # Neither the JSON payload nor the raw HTML pill markup should let the
    # untrusted company name break out of its context.
    assert "<script>alert(1)</script>" not in out
    assert payload["ticker"] in out


def test_build_multi_dashboard_replaces_all_placeholders(shocked_ohlcv):
    payload = _payload(shocked_ohlcv)
    html = build_multi_dashboard([payload], period="2y", generated_at="July 23, 2026")
    # Check the actual template placeholder tokens are gone -- NOT a blanket
    # "{{"/"}}"  search, since the template's CSS legitimately contains "}}"
    # from nested rules (e.g. `@media(...){.foo{bar:1fr}}`).
    for placeholder in (
        "{{COMPANIES_JSON}}", "{{FIRST_TICKER}}", "{{N_COMPANIES}}",
        "{{PERIOD}}", "{{GENERATED_AT}}", "{{PILLS_HTML}}",
        "{{ZSCORE_THRESHOLD}}", "{{LLM_MODEL}}",
    ):
        assert placeholder not in html
