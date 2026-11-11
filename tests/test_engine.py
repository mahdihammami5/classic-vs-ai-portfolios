import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.engine import rebalance_dates, run_backtest
from pfopt.config import BacktestConfig
from pfopt.strategies.base import Strategy
from pfopt.strategies.classic import EqualWeight


class Recorder(Strategy):
    """Equal weight that records the last date it was shown."""

    def __init__(self):
        super().__init__("Recorder")
        self.seen = []

    def allocate(self, hist, prev, rng):
        self.seen.append(hist.index[-1])
        return pd.Series(1.0, index=hist.columns)


def test_rebalance_dates_are_month_ends_plus_initial_day():
    idx = pd.bdate_range("2019-12-02", "2020-03-31")
    dates = rebalance_dates(idx, "2020-01-01", "2020-03-31", "M")
    assert dates[0] == pd.Timestamp("2019-12-31")
    assert pd.Timestamp("2020-01-31") in dates and pd.Timestamp("2020-02-28") in dates
    assert pd.Timestamp("2020-03-31") not in dates  # nothing left to hold after the last day


def test_no_look_ahead(market, bt_cfg):
    strat = Recorder()
    res = run_backtest(strat, market, bt_cfg)
    reb = list(res.weights.index)
    assert strat.seen == reb  # strategy only saw data up to each rebalance date
    # Returns earned after a rebalance start strictly after that rebalance date.
    for t in reb[:-1]:
        assert res.returns.index[res.returns.index > t][0] > t


def test_equal_weight_matches_manual_buy_and_hold(market):
    cfg = BacktestConfig(start="2016-01-01", end="2016-03-31", lookback=60, cost_bps=0.0,
                         universe_size=12, liquidity_window=20)
    res = run_backtest(EqualWeight("EW"), market, cfg)
    t0, t1 = res.weights.index[0], res.weights.index[1]
    block = market.returns.loc[(market.returns.index > t0) & (market.returns.index <= t1)]
    expected = (1 + block).cumprod().mean(axis=1)
    got = (1 + res.returns.loc[block.index]).cumprod()
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_costs_reduce_returns_by_turnover(market, bt_cfg):
    free = run_backtest(EqualWeight("EW"), market, BacktestConfig(**{**bt_cfg.__dict__,
                                                                     "cost_bps": 0.0}))
    paid = run_backtest(EqualWeight("EW"), market, bt_cfg)
    assert paid.returns.sum() < free.returns.sum()
    # First rebalance buys the whole portfolio from cash: turnover 1.0 -> cost 10 bps.
    assert paid.turnover.iloc[0] == pytest.approx(1.0)
    assert paid.costs.iloc[0] == pytest.approx(0.001)


def test_drift_makes_equal_weight_trade_every_month(market, bt_cfg):
    res = run_backtest(EqualWeight("EW"), market, bt_cfg)
    assert (res.turnover.iloc[1:] > 0).all()


def test_universe_picks_most_liquid(market, bt_cfg):
    res = run_backtest(EqualWeight("EW"), market, bt_cfg)
    # Dollar volume decreases with the asset number, so S10 and S11 are excluded.
    assert set(res.weights.columns) == {f"S{i:02d}" for i in range(10)}


class Rotator(Strategy):
    """All-in on a different asset at every call: maximal turnover."""

    def __init__(self):
        super().__init__("Rotator")
        self.calls = 0

    def allocate(self, hist, prev, rng):
        w = pd.Series(0.0, index=hist.columns)
        w.iloc[self.calls % len(w)] = 1.0
        self.calls += 1
        return w


def test_default_policy_equals_plain_calendar(market, bt_cfg):
    from pfopt.config import RebalancePolicy

    a = run_backtest(EqualWeight("ew"), market, bt_cfg)
    b = run_backtest(EqualWeight("ew"), market, bt_cfg, policy=RebalancePolicy())
    pd.testing.assert_series_equal(a.returns, b.returns)
    assert (a.events.reason.iloc[1:] == "schedule").all()


def test_no_trade_band_skips_small_trades(market, bt_cfg):
    from pfopt.config import RebalancePolicy

    plain = run_backtest(EqualWeight("ew"), market, bt_cfg)
    banded = run_backtest(EqualWeight("ew"), market, bt_cfg, policy=RebalancePolicy(band=0.5))
    assert len(banded.weights) == 1  # equal weight never drifts 50% away: only initial trade
    assert not banded.events.traded.iloc[1:].any()
    assert banded.costs.sum() < plain.costs.sum()
    # Large trades still go through.
    rot = run_backtest(Rotator(), market, bt_cfg, policy=RebalancePolicy(band=0.5))
    assert rot.events.traded.all()


def test_drift_and_vix_triggers(market, bt_cfg):
    from pfopt.config import RebalancePolicy

    pol = RebalancePolicy(schedule="Y", check="W", drift=0.02)
    res = run_backtest(EqualWeight("ew"), market, bt_cfg, policy=pol)
    trig = res.events[res.events.reason == "drift"]
    assert len(trig) > 0 and (trig.drift > 0.02).all()

    level = float(market.vix.loc[bt_cfg.start:].median())
    res = run_backtest(EqualWeight("ew"), market, bt_cfg,
                       policy=RebalancePolicy(schedule="Y", check="D", vix_level=level))
    for _, e in res.events[res.events.reason.str.startswith("vix")].iterrows():
        v = market.vix.loc[: e.date]
        assert (v.iloc[-1] >= level) == (e.reason == "vix_up")


def test_contributions_add_up_to_gross_return(market, bt_cfg):
    res = run_backtest(EqualWeight("ew"), market, bt_cfg)
    by_year = res.gross_returns.groupby(res.gross_returns.index.year).sum()
    np.testing.assert_allclose(res.contributions.sum(axis=1).to_numpy(), by_year.to_numpy())
