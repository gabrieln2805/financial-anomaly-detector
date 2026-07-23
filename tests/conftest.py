"""Shared fixtures: deterministic synthetic OHLCV data, no network required."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _gen_ohlcv(n: int = 300, seed: int = 0, shocks: dict | None = None) -> pd.DataFrame:
    """Synthetic daily OHLCV series with optional injected shocks (day-index -> % move).

    Mirrors the generator in make_demo.py so anomaly injection is realistic.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-03", periods=n, freq="B", name="Date")
    shocks = shocks or {}
    price = 50.0
    closes, opens, highs, lows, vols = [], [], [], [], []
    base_vol = 1_000_000

    for i in range(n):
        shock = shocks.get(i, 0) / 100
        vm = 6.0 if shock != 0 else 1.0
        daily_ret = shock + rng.normal(0, 0.015)
        close = round(max(price * (1 + daily_ret), 0.10), 2)
        opens.append(round(price * (1 + rng.normal(0, 0.004)), 2))
        highs.append(round(max(close, price) * (1 + abs(rng.normal(0, 0.003))), 2))
        lows.append(round(min(close, price) * (1 - abs(rng.normal(0, 0.003))), 2))
        closes.append(close)
        vols.append(int(base_vol * vm * rng.uniform(0.7, 1.3)))
        price = close

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=dates,
    )


@pytest.fixture
def clean_ohlcv() -> pd.DataFrame:
    """No injected shocks — should produce few/no anomalies."""
    return _gen_ohlcv(n=200, seed=1)


@pytest.fixture
def shocked_ohlcv() -> pd.DataFrame:
    """A few large price/volume shocks injected at known indices."""
    return _gen_ohlcv(n=200, seed=2, shocks={80: 35, 81: -5, 140: -40})
