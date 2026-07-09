#!/usr/bin/env python3
"""
LLM-Powered Anomaly Detection on Financial Time Series
Usage: python anomaly_detector.py TSLA UPST AFRM [--period 2y] [--top 10]

Requires: pip install yfinance anthropic pandas numpy
Set ANTHROPIC_API_KEY for Claude explanations; falls back to rule-based otherwise.
"""

import os
import sys
import argparse
import datetime
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from dashboard_builder import company_to_js_payload, build_multi_dashboard

# ── constants ──────────────────────────────────────────────────────────────────
ZSCORE_THRESHOLD = 2.5
IQR_MULTIPLIER   = 1.75
ROLLING_WINDOW   = 20
TOP_ANOMALIES    = 12
LLM_MODEL        = "claude-haiku-4-5"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_pct"]   = out["Close"].pct_change() * 100
    out["log_return"]   = np.log(out["Close"] / out["Close"].shift(1))

    vol_mean = out["Volume"].rolling(ROLLING_WINDOW).mean()
    vol_std  = out["Volume"].rolling(ROLLING_WINDOW).std()
    out["volume_zscore"]   = (out["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
    out["volume_multiple"] = out["Volume"] / vol_mean

    out["realized_vol"] = out["log_return"].rolling(ROLLING_WINDOW).std() * np.sqrt(252) * 100

    rv_mean = out["realized_vol"].rolling(ROLLING_WINDOW * 3).mean()
    rv_std  = out["realized_vol"].rolling(ROLLING_WINDOW * 3).std()
    out["vol_zscore"] = (out["realized_vol"] - rv_mean) / rv_std.replace(0, np.nan)

    ret_mean = out["return_pct"].rolling(ROLLING_WINDOW).mean()
    ret_std  = out["return_pct"].rolling(ROLLING_WINDOW).std()
    out["return_zscore"] = (out["return_pct"] - ret_mean) / ret_std.replace(0, np.nan)

    return out.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# 2. ANOMALY DETECTION — Z-SCORE + IQR ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def _iqr_flags(series: pd.Series, k: float = IQR_MULTIPLIER) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    signals = {"return": "return_zscore", "volume": "volume_zscore", "vol": "vol_zscore"}
    anom = df.copy()
    triggered = []

    for name, zcol in signals.items():
        z    = anom[zcol].fillna(0)
        col  = f"flag_{name}"
        anom[col] = (z.abs() > ZSCORE_THRESHOLD) & _iqr_flags(z)
        triggered.append((col, zcol))

    anom["severity"]     = sum(anom[c].astype(float) * anom[z].abs() for c, z in triggered)
    anom["is_anomaly"]   = anom[["flag_return", "flag_volume", "flag_vol"]].any(axis=1)
    anom["signal_count"] = anom[["flag_return", "flag_volume", "flag_vol"]].sum(axis=1)
    return anom


def top_anomalies(anom: pd.DataFrame, n: int = TOP_ANOMALIES) -> pd.DataFrame:
    result = anom[anom["is_anomaly"]].nlargest(n, "severity").reset_index()
    for col in list(result.columns):
        if col.lower() in ("index", "date", "datetime"):
            result = result.rename(columns={col: "Date"})
            break
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. LLM EXPLANATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _build_context(row, df: pd.DataFrame, ticker: str) -> str:
    date     = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])
    ret_pct  = row.get("return_pct", 0)
    vol_mult = row.get("volume_multiple", 1)
    real_vol = row.get("realized_vol", 0)
    close    = row.get("Close", 0)

    fired = []
    if row.get("flag_return"): fired.append(f"price return {ret_pct:+.2f}%")
    if row.get("flag_volume"): fired.append(f"volume {vol_mult:.1f}x avg")
    if row.get("flag_vol"):    fired.append(f"realized vol {real_vol:.1f}%")

    try:
        idx = df.index.get_loc(row["Date"])
        win = df.iloc[max(0, idx-2): idx+3]
        ctx = "\n".join(
            f"  {d.date()} close={r['Close']:.2f} ret={r['return_pct']:+.2f}% vol={r['volume_multiple']:.1f}x"
            for d, r in win.iterrows()
        )
    except Exception:
        ctx = "(window unavailable)"

    return (
        f"Ticker: {ticker}\nDate: {date}\nClose: ${close:.2f}\n"
        f"Signals: {', '.join(fired) or 'composite'}\n5-day context:\n{ctx}"
    )


def _claude_explain(context: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL, max_tokens=180,
        system=(
            "You are a quantitative financial analyst. Given a detected anomaly in a stock's "
            "time series, write 2-3 sentences explaining the most likely market cause. "
            "Be specific about signal type. No bullet points."
        ),
        messages=[{"role": "user", "content": f"Explain this anomaly:\n\n{context}"}]
    )
    return msg.content[0].text.strip()


def _rule_explain(context: str) -> str:
    lo = context.lower()
    parts = []
    if "price return" in lo:
        parts.append("An unusually large price move suggests a significant news catalyst such as "
                     "an earnings surprise, analyst rating change, or macro event.")
    if "volume" in lo:
        parts.append("Elevated trading volume points to institutional repositioning or "
                     "an information event driving abnormal order flow.")
    if "realized vol" in lo and "volume" not in lo:
        parts.append("A volatility regime shift may reflect increased uncertainty around "
                     "earnings, sector rotation, or broader market stress.")
    if not parts:
        parts.append("Multiple signals breached statistical thresholds simultaneously, "
                     "consistent with a binary market event such as earnings, regulatory action, "
                     "or macro shock.")
    return " ".join(parts)


def explain_anomalies(top: pd.DataFrame, df: pd.DataFrame, ticker: str) -> list:
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    results = []

    for _, row in top.iterrows():
        context = _build_context(row, df, ticker)
        if use_llm:
            try:
                explanation = _claude_explain(context)
            except Exception as e:
                explanation = _rule_explain(context) + f" (LLM error: {str(e)[:40]})"
        else:
            explanation = _rule_explain(context)

        fired = []
        if row.get("flag_return"): fired.append("Return")
        if row.get("flag_volume"): fired.append("Volume")
        if row.get("flag_vol"):    fired.append("Volatility")

        results.append({
            "date":         row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"]),
            "close":        round(float(row.get("Close", 0)), 2),
            "return_pct":   round(float(row.get("return_pct", 0)), 2),
            "vol_mult":     round(float(row.get("volume_multiple", 1)), 1),
            "realized_vol": round(float(row.get("realized_vol", 0)), 1),
            "severity":     round(float(row.get("severity", 0)), 2),
            "signal_count": int(row.get("signal_count", 1)),
            "signals":      fired,
            "explanation":  explanation,
        })

    return sorted(results, key=lambda x: x["date"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLM-Powered Financial Anomaly Detector")
    parser.add_argument("tickers", nargs="+", help="One or more tickers e.g. TSLA UPST AFRM")
    parser.add_argument("--period",  default="2y",  help="yfinance period: 1y 2y 5y (default 2y)")
    parser.add_argument("--top",     type=int, default=TOP_ANOMALIES)
    parser.add_argument("--output",  default=".")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    out_dir      = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_llm      = bool(os.environ.get("ANTHROPIC_API_KEY"))
    generated_at = datetime.datetime.now().strftime("%B %d, %Y %H:%M")
    payloads     = []

    for raw in args.tickers:
        ticker = raw.upper()
        print(f"\n[{ticker}] Fetching OHLCV ({args.period})...")
        try:
            df_raw = fetch_ohlcv(ticker, args.period)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        print(f"  {len(df_raw):,} days")

        df      = engineer_features(df_raw)
        anom_df = detect_anomalies(df)
        n_anom  = int(anom_df["is_anomaly"].sum())
        print(f"  {n_anom} anomalies")

        top   = top_anomalies(anom_df, args.top)
        expls = explain_anomalies(top, df, ticker)

        try:
            info    = yf.Ticker(ticker).info
            company = info.get("longName", ticker)
            sector  = info.get("sector", "--")
        except Exception:
            company, sector = ticker, "--"

        payloads.append(company_to_js_payload(ticker, company, sector, df, anom_df, expls))
        print(f"  Done: {company} / {sector}")

    if not payloads:
        print("No valid tickers processed.")
        sys.exit(1)

    slug     = "_".join(c["ticker"] for c in payloads)
    out_path = out_dir / f"{slug}_anomalies.html"
    html     = build_multi_dashboard(payloads, args.period, generated_at,
                                     ZSCORE_THRESHOLD, LLM_MODEL)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard -> {out_path.resolve()}")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

    print("\n-- Summary --")
    for c in payloads:
        print(f"  {c['ticker']:<6} {c['name']:<40} anomalies={c['kpis']['nAnom']}")


if __name__ == "__main__":
    main()
