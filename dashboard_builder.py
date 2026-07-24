"""
dashboard_builder.py
Loads templates/dashboard.html and injects company data.
Imported by anomaly_detector.py and make_demo.py.
"""
import html
import json
from pathlib import Path

# Locate the template relative to THIS file
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"

SECTOR_COLORS = {
    "Technology":             "#6366f1",
    "Consumer Cyclical":      "#f59e0b",
    "Financial Services":     "#22c55e",
    "Healthcare":             "#ec4899",
    "Real Estate":            "#14b8a6",
    "Energy":                 "#f97316",
    "Industrials":            "#8b5cf6",
    "Communication Services": "#06b6d4",
    "Consumer Defensive":     "#84cc16",
    "Basic Materials":        "#a78bfa",
    "Utilities":              "#94a3b8",
}


def sector_color(sector: str) -> str:
    return SECTOR_COLORS.get(sector, "#6366f1")


def company_to_js_payload(
    ticker: str, company: str, sector: str,
    df, anom_df, explanations: list,
) -> dict:
    """Serialize one company into a JSON-safe dict for the dashboard."""

    anom_mask   = anom_df["is_anomaly"]
    anom_dates  = [d.strftime("%Y-%m-%d") for d in anom_df[anom_mask].index]
    anom_prices = {
        d.strftime("%Y-%m-%d"): round(float(r["Close"]), 2)
        for d, r in anom_df[anom_mask].iterrows()
    }

    n_total   = len(df)
    n_anom    = int(anom_mask.sum())
    multi_sig = sum(1 for e in explanations if e.get("signal_count", 0) > 1)
    ret_vals  = [abs(e["return_pct"]) for e in explanations if "Return" in e.get("signals", [])]

    sig_counts = {"Return": 0, "Volume": 0, "Volatility": 0}
    for e in explanations:
        for s in e.get("signals", []):
            sig_counts[s] = sig_counts.get(s, 0) + 1

    return {
        "ticker":   ticker,
        "name":     company,
        "sector":   sector,
        "color":    sector_color(sector),
        "dates":    [d.strftime("%Y-%m-%d") for d in df.index],
        "closes":   [round(float(v), 2) for v in df["Close"]],
        "volMult":  [round(float(v), 2) for v in df["volume_multiple"].fillna(1)],
        "rv":       [round(float(v), 2) for v in df["realized_vol"].fillna(0)],
        "anomDates":   anom_dates,
        "anomScatter": [{"x": d, "y": anom_prices[d]} for d in anom_dates if d in anom_prices],
        "sigCounts":   sig_counts,
        "kpis": {
            "nTotal":     n_total,
            "nAnom":      n_anom,
            "anomRate":   round(n_anom / n_total * 100, 1),
            "multiSig":   multi_sig,
            "avgRetAnom": round(sum(ret_vals) / len(ret_vals), 2) if ret_vals else 0,
        },
        "explanations": explanations,
    }


def _safe_json(obj) -> str:
    """json.dumps that is safe to embed inside a <script> block.

    Escapes </script>, <!--, and line separators so that ticker names,
    company names, or LLM-generated explanation text can never break out
    of the script context or terminate an HTML comment.
    """
    return (
        json.dumps(obj)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_multi_dashboard(
    companies: list,
    period: str,
    generated_at: str,
    zscore_threshold: float = 2.5,
    llm_model: str = "claude-haiku-4-5",
) -> str:
    """Render the multi-company dashboard HTML from the template."""
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    first    = companies[0]["ticker"]

    # Build company pills
    pills = []
    for c in companies:
        # Escape everything interpolated into raw HTML — ticker/name/sector
        # originate from Yahoo Finance and are not trusted input.
        ticker = html.escape(str(c["ticker"]))
        name   = html.escape(str(c["name"]))
        sector = html.escape(str(c["sector"]))
        color  = html.escape(str(c["color"]), quote=True)
        pills.append(
            f'<button class="co-pill" id="pill-{ticker}" '
            f'onclick="switchCompany(\'{ticker}\')">'
            f'<span class="pill-ticker">{ticker}</span>'
            f'<span class="pill-name">{name}</span>'
            f'<span class="pill-sector" style="color:{color}">{sector}</span>'
            f'</button>'
        )

    companies_map = {c["ticker"]: c for c in companies}

    rendered = (template
        .replace("{{COMPANIES_JSON}}",   _safe_json(companies_map))
        .replace("{{FIRST_TICKER}}",     first)
        .replace("{{N_COMPANIES}}",      str(len(companies)))
        .replace("{{PERIOD}}",           period)
        .replace("{{GENERATED_AT}}",     generated_at)
        .replace("{{PILLS_HTML}}",       "\n".join(pills))
        .replace("{{ZSCORE_THRESHOLD}}", str(zscore_threshold))
        .replace("{{LLM_MODEL}}",        llm_model)
    )
    return rendered


# Single-company backward-compat wrapper
def build_dashboard(ticker, company, sector, df, anom_df, explanations,
                    period, generated_at,
                    zscore_threshold=2.5, llm_model="claude-haiku-4-5"):
    payload = company_to_js_payload(ticker, company, sector, df, anom_df, explanations)
    return build_multi_dashboard([payload], period, generated_at,
                                 zscore_threshold, llm_model)