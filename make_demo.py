"""
Generate multi-company anomaly demo (no network needed).
Company configs and explanations live in demo_companies.json.
Produces: anomaly_demo.html
"""
import json, datetime, os
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.pop("ANTHROPIC_API_KEY", None)

from anomaly_detector import (
    engineer_features, detect_anomalies, top_anomalies, explain_anomalies,
)
from dashboard_builder import company_to_js_payload, build_multi_dashboard


def gen_ohlcv(cfg: dict, n: int = 504, seed: int = 0) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-03", periods=n, freq="B", name="Date")
    price = cfg["price0"]
    drift = cfg["drift"] / 252
    dvol  = cfg["dvol"]
    shocks = {int(k): v for k, v in cfg["shocks"].items()}
    bvol  = cfg["base_vol"]

    closes, opens, highs, lows, vols = [], [], [], [], []
    for i in range(n):
        shock    = shocks.get(i, 0) / 100
        vm       = 5.0 if shock != 0 else 1.0
        daily_ret = shock + drift + rng.normal(0, dvol)
        close    = round(max(price * (1 + daily_ret), 0.10), 2)
        opens.append(round(price * (1 + rng.normal(0, dvol * 0.3)), 2))
        highs.append(round(max(close, price) * (1 + abs(rng.normal(0, dvol * 0.2))), 2))
        lows.append(round(min(close, price) * (1 - abs(rng.normal(0, dvol * 0.2))), 2))
        closes.append(close)
        vols.append(int(bvol * vm * rng.uniform(0.6, 1.4)))
        price = close

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=dates,
    )


# Load company roster from JSON
COMPANIES = json.loads(Path("demo_companies.json").read_text(encoding="utf-8"))

payloads = []
for idx, cfg in enumerate(COMPANIES):
    ticker = cfg["ticker"]
    print(f"[{ticker}] {cfg['name']}...")

    df_raw  = gen_ohlcv(cfg, seed=idx * 17)
    df      = engineer_features(df_raw)
    anom_df = detect_anomalies(df)
    top     = top_anomalies(anom_df, 10)
    expls   = explain_anomalies(top, df, ticker)

    curated = cfg["explanations"]
    for i, e in enumerate(expls):
        e["explanation"] = curated[i % len(curated)]

    payload = company_to_js_payload(
        ticker=ticker, company=cfg["name"], sector=cfg["sector"],
        df=df, anom_df=anom_df, explanations=expls,
    )
    payloads.append(payload)
    print(f"  anomalies={payload['kpis']['nAnom']}")

html = build_multi_dashboard(
    companies=payloads,
    period="2y",
    generated_at=datetime.datetime.now().strftime("%B %d, %Y %H:%M"),
)

out = Path("anomaly_demo.html")
out.write_text(html, encoding="utf-8")
print(f"\nWritten -> {out.resolve()}")
print(f"Companies: {[p['ticker'] for p in payloads]}")
