"""Walk-forward backtesting engine.

Timing convention (no look-ahead):
  * On rebalance date ``t`` the strategy sees returns up to and including the close of ``t``.
  * Trades execute at the close of ``t``; the new weights earn returns from ``t+1`` onward.
  * Between rebalances weights drift with prices (buy-and-hold), so turnover is measured
    against the drifted weights, as in a real portfolio.
  * Transaction cost = ``cost_bps`` x traded notional, deducted on the first day of the period.

When to trade is set by a :class:`~pfopt.config.RebalancePolicy`: a calendar schedule, plus
optional event triggers (drift from target, VIX regime change) checked on extra dates, and an
optional no-trade band. The default policy is plain calendar rebalancing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pfopt.config import BacktestConfig, RebalancePolicy
from pfopt.data.market import MarketData
from pfopt.data.universe import select_universe
from pfopt.strategies.base import RebalanceContext, Strategy

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series  # net daily returns
    gross_returns: pd.Series
    weights: pd.DataFrame  # target weights, one row per executed rebalance
    turnover: pd.Series  # sum |w_target - w_drifted| at each executed rebalance (two-way)
    costs: pd.Series
    runtime_s: float = 0.0
    meta: dict = field(default_factory=dict)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)  # every rebalance decision
    contributions: pd.DataFrame = field(default_factory=pd.DataFrame)  # year x asset, gross


FREQS = {"D", "W", "M", "Q", "Y"}


def rebalance_dates(index: pd.DatetimeIndex, start, end, freq: str) -> list[pd.Timestamp]:
    """Last trading day before ``start``, then the last trading day of each period.

    The final trading day in the window is excluded (nothing to hold after it).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    before = index[index < start]
    if len(before) == 0:
        raise ValueError("Need at least one trading day before the backtest start.")
    window = index[(index >= start) & (index <= end)]
    if freq not in FREQS:
        raise ValueError(f"Unknown rebalance frequency {freq!r}; use one of {sorted(FREQS)}.")
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


def one_way_distance(a: pd.Series, b: pd.Series) -> float:
    """Share of the portfolio that must be bought (= sold) to go from ``b`` to ``a``."""
    if a.empty and b.empty:
        return 0.0
    names = a.index.union(b.index)
    diff = a.reindex(names, fill_value=0.0) - b.reindex(names, fill_value=0.0)
    return 0.5 * float(diff.abs().sum())


def run_backtest(
    strategy: Strategy,
    data: MarketData,
    cfg: BacktestConfig,
    seed: int = 42,
    policy: RebalancePolicy | None = None,
) -> BacktestResult:
    policy = policy or RebalancePolicy()
    idx = data.returns.index
    end = pd.Timestamp(cfg.end) if cfg.end else idx[-1]
    schedule = rebalance_dates(idx, cfg.start, end, policy.schedule or cfg.rebalance)
    checks = schedule
    if policy.check:
        checks = sorted(set(schedule) | set(rebalance_dates(idx, cfg.start, end, policy.check)))
    scheduled = set(schedule)
    if policy.vix_level is not None and data.vix is None:
        raise ValueError("A VIX trigger needs VIX data (vix.parquet).")
    last_day = idx[idx <= end][-1]
    pos = {d: k for k, d in enumerate(idx)}

    universe: list[str] = []
    universe_year = None
    w_hold = pd.Series(dtype=float)  # drifted holdings at the current date
    w_target = pd.Series(dtype=float)  # last executed target
    vix_state: bool | None = None
    last_trigger = -(10**9)
    n_calls = 0
    net_parts, gross_parts, events, contrib = [], [], [], []
    weights_rows, turnover, costs = {}, {}, {}
    t_start = time.perf_counter()

    for j, t in enumerate(checks):
        drift = one_way_distance(w_hold, w_target)
        can_trigger = pos[t] - last_trigger >= policy.min_gap
        reason = None
        if j == 0:
            reason = "initial"
        elif t in scheduled:
            reason = "schedule"
        elif can_trigger and policy.drift is not None and drift > policy.drift:
            reason = "drift"
        if policy.vix_level is not None:
            vix_t = data.vix.loc[:t]
            state = bool(vix_t.iloc[-1] >= policy.vix_level) if len(vix_t) else False
            if reason is None and can_trigger and vix_state is not None and state != vix_state:
                reason = "vix_up" if state else "vix_down"
            vix_state = state

        target, cost = w_hold, 0.0
        if reason is not None:
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
                date=t, step=n_calls, universe=list(universe), returns=past_returns,
                universe_at=lambda d, r=past_returns, v=past_volume: select_universe(
                    d, data.membership, r, v, cfg.universe_size, cfg.lookback,
                    cfg.liquidity_window),
                vix=None if data.vix is None else data.vix.loc[:t],
            ))
            hist = past_returns[universe].iloc[-cfg.lookback :].dropna(axis=1)
            prev = w_hold.reindex(hist.columns).fillna(0.0)
            rng = np.random.default_rng([seed, n_calls])
            n_calls += 1
            proposal = _validate_weights(strategy.allocate(hist, prev, rng), strategy.name)
            proposal = proposal[proposal > 0]
            gap = one_way_distance(proposal, w_hold)
            traded = not (policy.band is not None and reason != "initial" and gap < policy.band)
            events.append({"date": t, "reason": reason, "traded": traded,
                           "drift": drift, "proposal_distance": gap})
            if traded:
                target = w_target = proposal
                tw = 2.0 * gap
                cost = tw * cfg.cost_bps / 1e4
                weights_rows[t], turnover[t], costs[t] = target, tw, cost
                if reason not in {"initial", "schedule"}:
                    last_trigger = pos[t]

        next_t = checks[j + 1] if j + 1 < len(checks) else last_day
        period = data.returns.loc[(idx > t) & (idx <= next_t), target.index].fillna(0.0)
        if period.empty:
            continue
        r = period.to_numpy()
        w0 = target.to_numpy()
        growth = np.cumprod(1.0 + r, axis=0)
        value = growth @ w0
        prev_value = np.concatenate([[1.0], value[:-1]])
        prev_growth = np.vstack([np.ones((1, r.shape[1])), growth[:-1]])
        gross = value / prev_value - 1.0
        net = gross.copy()
        net[0] = (1.0 - cost) * (1.0 + gross[0]) - 1.0
        gross_parts.append(pd.Series(gross, index=period.index))
        net_parts.append(pd.Series(net, index=period.index))
        # Daily contribution of each asset to the gross return, summed per calendar year.
        daily = w0 * prev_growth * r / prev_value[:, None]
        contrib.append(pd.DataFrame(daily, index=period.index, columns=target.index)
                       .groupby(period.index.year).sum())
        w_hold = pd.Series(w0 * growth[-1] / value[-1], index=target.index)

    runtime = time.perf_counter() - t_start
    weights = pd.DataFrame(weights_rows).T.fillna(0.0).sort_index(axis=1)
    log.info("%s: %d rebalances (%d strategy calls, %d dates checked) in %.1fs",
             strategy.name, len(weights_rows), n_calls, len(checks), runtime)
    contributions = pd.DataFrame()
    if contrib:
        contributions = pd.concat(contrib).groupby(level=0).sum().fillna(0.0).sort_index(axis=1)
        contributions.index.name = "year"
    return BacktestResult(
        name=strategy.name,
        returns=pd.concat(net_parts).rename(strategy.name),
        gross_returns=pd.concat(gross_parts).rename(strategy.name),
        weights=weights,
        turnover=pd.Series(turnover, name="turnover", dtype=float),
        costs=pd.Series(costs, name="cost", dtype=float),
        runtime_s=runtime,
        meta=strategy.report(),
        events=pd.DataFrame(events),
        contributions=contributions,
    )
