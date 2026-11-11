"""Walk-forward backtesting engine.

Timing convention (no look-ahead):
  * On rebalance date ``t`` the strategy sees returns up to and including the close of ``t``.
  * Trades execute at the close of ``t``; the new weights earn returns from ``t+1`` onward.
  * Between rebalances weights drift with prices (buy-and-hold), so turnover is measured
    against the drifted weights, as in a real portfolio.
  * Transaction cost = ``cost_bps`` x traded notional, deducted on the first day of the period.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pfopt.config import BacktestConfig
from pfopt.data.market import MarketData
from pfopt.data.universe import select_universe
from pfopt.strategies.base import RebalanceContext, Strategy

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series  # net daily returns
    gross_returns: pd.Series
    weights: pd.DataFrame  # target weights, one row per rebalance date
    turnover: pd.Series  # sum |w_target - w_drifted| at each rebalance (two-way)
    costs: pd.Series
    runtime_s: float = 0.0
    meta: dict = field(default_factory=dict)


def rebalance_dates(index: pd.DatetimeIndex, start, end, freq: str) -> list[pd.Timestamp]:
    """Last trading day before ``start``, then the last trading day of each period.

    The final trading day in the window is excluded (nothing to hold after it).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    before = index[index < start]
    if len(before) == 0:
        raise ValueError("Need at least one trading day before the backtest start.")
    window = index[(index >= start) & (index <= end)]
    if freq not in {"M", "Q"}:
        raise ValueError(f"Unknown rebalance frequency {freq!r}; use 'M' or 'Q'.")
    period_end = window.to_series().groupby(window.to_period(freq)).max()
    dates = [before[-1], *[d for d in period_end if d != window[-1]]]
    return dates


def _validate_weights(w: pd.Series, name: str) -> pd.Series:
    w = w.astype(float).fillna(0.0)
    if (w < -1e-9).any():
        raise ValueError(f"{name}: negative weights are not allowed (long-only).")
    w = w.clip(lower=0.0)
    total = w.sum()
    if total <= 0:
        raise ValueError(f"{name}: weights sum to zero.")
    return w / total


def run_backtest(
    strategy: Strategy,
    data: MarketData,
    cfg: BacktestConfig,
    seed: int = 42,
) -> BacktestResult:
    idx = data.returns.index
    end = pd.Timestamp(cfg.end) if cfg.end else idx[-1]
    reb = rebalance_dates(idx, cfg.start, end, cfg.rebalance)
    last_day = idx[idx <= end][-1]

    universe: list[str] = []
    universe_year = None
    w_prev = pd.Series(dtype=float)  # drifted weights just before the next rebalance
    net_parts, gross_parts = [], []
    weights_rows, turnover, costs = {}, {}, {}
    t_start = time.perf_counter()

    for i, t in enumerate(reb):
        refresh = cfg.universe_refresh == "R" or universe_year != t.year or not universe
        if refresh:
            universe = select_universe(
                t, data.membership, data.returns, data.dollar_volume,
                cfg.universe_size, cfg.lookback, cfg.liquidity_window,
            )
            universe_year = t.year
            if not universe:
                raise ValueError(f"Empty universe on {t.date()}: not enough price history.")

        past_returns = data.returns.loc[:t]
        past_volume = data.dollar_volume.loc[:t]
        strategy.on_rebalance(RebalanceContext(
            date=t, step=i, universe=list(universe), returns=past_returns,
            universe_at=lambda d, r=past_returns, v=past_volume: select_universe(
                d, data.membership, r, v, cfg.universe_size, cfg.lookback,
                cfg.liquidity_window),
            vix=None if data.vix is None else data.vix.loc[:t],
        ))
        hist = past_returns[universe].iloc[-cfg.lookback :].dropna(axis=1)
        prev = w_prev.reindex(hist.columns).fillna(0.0)
        rng = np.random.default_rng([seed, i])
        target = _validate_weights(strategy.allocate(hist, prev, rng), strategy.name)
        target = target[target > 0]

        names = target.index.union(w_prev.index)
        traded = (target.reindex(names, fill_value=0.0) - w_prev.reindex(names, fill_value=0.0))
        tw = float(traded.abs().sum())
        cost = tw * cfg.cost_bps / 1e4
        weights_rows[t], turnover[t], costs[t] = target, tw, cost

        next_t = reb[i + 1] if i + 1 < len(reb) else last_day
        period = data.returns.loc[(idx > t) & (idx <= next_t), target.index].fillna(0.0)
        if period.empty:
            continue
        growth = np.cumprod(1.0 + period.to_numpy(), axis=0)
        value = growth @ target.to_numpy()
        prev_value = np.concatenate([[1.0], value[:-1]])
        gross = value / prev_value - 1.0
        net = gross.copy()
        net[0] = (1.0 - cost) * (1.0 + gross[0]) - 1.0
        gross_parts.append(pd.Series(gross, index=period.index))
        net_parts.append(pd.Series(net, index=period.index))
        w_prev = pd.Series(target.to_numpy() * growth[-1] / value[-1], index=target.index)

    runtime = time.perf_counter() - t_start
    weights = pd.DataFrame(weights_rows).T.fillna(0.0).sort_index(axis=1)
    log.info("%s: %d rebalances in %.1fs", strategy.name, len(reb), runtime)
    return BacktestResult(
        name=strategy.name,
        returns=pd.concat(net_parts).rename(strategy.name),
        gross_returns=pd.concat(gross_parts).rename(strategy.name),
        weights=weights,
        turnover=pd.Series(turnover, name="turnover"),
        costs=pd.Series(costs, name="cost"),
        runtime_s=runtime,
        meta=strategy.report(),
    )
