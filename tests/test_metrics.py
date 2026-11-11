import numpy as np
import pandas as pd
import pytest

from pfopt.backtest import metrics as m


def series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_total_return_and_cagr():
    r = series([0.001] * 252)
    assert m.total_return(r) == pytest.approx(1.001**252 - 1)
    assert m.cagr(r) == pytest.approx(1.001**252 - 1)


def test_max_drawdown_counts_from_initial_capital():
    # Loses 50% immediately, recovers partly: drawdown measured from starting value 1.0.
    r = series([-0.5, 0.2, 0.1])
    assert m.max_drawdown(r) == pytest.approx(-0.5)
    r = series([0.1, -0.5, 0.2])
    assert m.max_drawdown(r) == pytest.approx(-0.5)


def test_sharpe_uses_excess_returns():
    rng = np.random.default_rng(0)
    r = series(rng.normal(0.001, 0.01, 1000))
    rf = series(np.full(1000, 0.001))
    assert abs(m.sharpe_ratio(r, rf)) < abs(m.sharpe_ratio(r))


def test_zero_volatility_gives_nan_sharpe():
    assert np.isnan(m.sharpe_ratio(series([0.0] * 10)))


def test_cvar_is_mean_of_worst_tail():
    r = series(list(np.linspace(-0.10, 0.09, 20)))
    assert m.cvar(r, 0.05) == pytest.approx(-0.10)


def test_summarize_turnover_is_one_way_annualized():
    r = series([0.0] * 252)
    turnover = pd.Series([0.4, 0.4], index=[r.index[0] - pd.Timedelta(days=1), r.index[100]])
    out = m.summarize(r, turnover=turnover)
    assert out["turnover"] == pytest.approx(0.4)
