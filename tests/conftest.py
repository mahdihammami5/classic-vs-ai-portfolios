"""Synthetic market data so tests are fast, deterministic and never touch the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfopt.config import BacktestConfig
from pfopt.data.market import MarketData
from pfopt.data.universe import Membership


def make_market(n_assets: int = 12, n_days: int = 700, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_assets)]
    vols = np.linspace(0.005, 0.03, n_assets)
    rets = rng.normal(0.0004, vols, size=(n_days, n_assets))
    returns = pd.DataFrame(rets, index=dates, columns=tickers)
    dollar_volume = pd.DataFrame(
        np.tile(np.arange(n_assets, 0, -1) * 1e6, (n_days, 1)), index=dates, columns=tickers
    )
    rf = pd.Series(0.0001, index=dates)
    membership = Membership.from_dict({"2000-01-01": tickers})
    benchmark = returns.mean(axis=1)
    vix = pd.Series(20.0 * np.exp(rng.normal(0, 0.05, n_days).cumsum() * 0.1), index=dates)
    return MarketData(returns, dollar_volume, rf, membership, benchmark, vix)


@pytest.fixture
def market() -> MarketData:
    return make_market()


@pytest.fixture
def bt_cfg() -> BacktestConfig:
    return BacktestConfig(
        start="2016-01-01", end=None, rebalance="M", lookback=120, cost_bps=10.0,
        universe_size=10, universe_refresh="Y", liquidity_window=20,
    )
