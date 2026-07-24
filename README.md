LLM-Powered Financial Anomaly Detection

> **LLM-powered anomaly detection on financial time series.**  
> Fetches OHLCV data via Yahoo Finance, runs a Z-score + IQR ensemble across three signals (returns, volume, realized volatility), then calls Claude to explain every flagged event in plain English with a risk score.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-6366f1?style=flat-square&logo=github)](https://gabrieln2805.github.io/financial-anomaly-detector)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Build Demo](https://github.com/gabrieln2805/financial-anomaly-detector/actions/workflows/build-demo.yml/badge.svg)](https://github.com/gabrieln2805/financial-anomaly-detector/actions/workflows/build-demo.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

<img width="2720" height="2960" alt="anomaly_detector_pipeline" src="https://github.com/user-attachments/assets/13ef89c9-69ba-420f-8e1b-e456db31bffd" />


## How it works

```
Yahoo Finance OHLCV
        │
        ▼
Feature Engineering
  ├─ Daily log returns
  ├─ Volume z-score (20-day rolling)
  └─ 20-day realized volatility
        │
        ▼
Z-score + IQR Ensemble
  Both |z| > 2.5 AND IQR fence broken
  → composite severity = Σ |z| across fired signals
        │
        ▼
Top-N Anomalies ranked by severity
        │
        ▼
Claude (claude-haiku-4-5)
  → 2-sentence market hypothesis
  → Risk tier: High / Medium / Low
        │
        ▼
Single-file HTML Dashboard
  ├─ Price chart with anomaly scatter overlay
  ├─ Volume × rolling mean (anomalies in red)
  ├─ Realized volatility regime
  ├─ Signal breakdown donut
  └─ LLM-explained anomaly cards
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. (Optional) Add your Anthropic key for LLM explanations
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run — opens the dashboard in your browser
python anomaly_detector.py TSLA UPST RIVN --period 2y
```

The dashboard is saved as `TSLA_UPST_RIVN_anomalies.html` — a single self-contained file you can share or host anywhere.

---

## Usage

```bash
# Single ticker
python anomaly_detector.py NVDA

# Multiple tickers → company filter built into one file
python anomaly_detector.py AAPL MSFT GOOGL META --period 1y

# Custom output directory, suppress auto-open
python anomaly_detector.py SOFI HIMS JOBY --output ./reports --no-open

# Generate the 8-company demo (synthetic data, no API key needed)
python make_demo.py
```

### Without an API key

If `ANTHROPIC_API_KEY` is not set, every anomaly card still renders with a rule-based explanation derived from the signal context (signal type, z-score magnitude, volume multiple). Set the key to upgrade to full Claude-powered hypotheses.

---

## Project structure

```
financial-anomaly-detector/
├── anomaly_detector.py     # CLI entry point: fetch → detect → explain → HTML
├── dashboard_builder.py    # Loads templates/dashboard.html, injects company data
├── make_demo.py            # Generates multi-company demo from synthetic data
├── demo_companies.json     # 8-company roster with curated explanations
├── templates/
│   └── dashboard.html      # Self-contained Chart.js dashboard template
├── requirements.txt
├── pyproject.toml
└── .github/
    └── workflows/
        └── build-demo.yml  # CI: rebuild + deploy to GitHub Pages on push
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(unset)* | Enable Claude explanations. Falls back to rule-based if missing. |
| `--period` | `2y` | yfinance lookback: `6mo`, `1y`, `2y`, `5y` |
| `--top` | `12` | Max anomalies explained per company |
| `ZSCORE_THRESHOLD` | `2.5` | Edit in `anomaly_detector.py` to tune sensitivity |
| `IQR_MULTIPLIER` | `1.75` | IQR fence width for ensemble |
| `ROLLING_WINDOW` | `20` | Days for rolling z-score and vol baseline |

---

## Adding companies to the demo

Edit `demo_companies.json`. Each entry needs:

```jsonc
{
  "ticker": "BIRD",
  "name": "Allbirds, Inc.",
  "sector": "Consumer Cyclical",
  "price0": 1.20,          // starting price for synthetic series
  "base_vol": 2000000,     // baseline daily volume
  "drift": -0.40,          // annual price drift (negative = declining)
  "dvol": 0.060,           // daily volatility
  "shocks": {              // day-index → % shock (+ or -)
    "45": -30, "120": 18, "210": -25
  },
  "explanations": [        // curated LLM-quality explanations (cycled)
    "Allbirds reported Q3 results showing...",
    "..."
  ]
}
```

Push to `main` → GitHub Actions rebuilds the live demo automatically.

---

## Detection algorithm

Each trading session is scored across three independent signals:

| Signal | Feature | Anomaly condition |
|---|---|---|
| **Return** | Daily % change vs 20-day rolling mean | `\|z\| > 2.5` AND outside IQR fence |
| **Volume** | Daily volume vs 20-day rolling mean | `\|z\| > 2.5` AND outside IQR fence |
| **Volatility** | 20-day realized vol vs 60-day baseline | `\|z\| > 2.5` AND outside IQR fence |

An event fires if **at least one** signal triggers. Composite severity = sum of `|z|` across all fired signals, used to rank events and determine which get Claude explanations.

---

## Use cases

- **KYC / transaction monitoring** — adapt signals to payment velocity and counterparty concentration
- **Trade reconciliation** — flag sessions where reported volume diverges from market data
- **Internal spend monitoring** — swap yfinance for an internal ledger time series
- **Research** — explore vol regime changes, earnings event detection, factor exposure breaks

---

