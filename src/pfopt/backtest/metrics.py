"""Performance and risk metrics computed from daily simple returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _excess(r: pd.Series, rf: pd.Series | None) -> pd.Series:
    if rf is None:
        return r
    return r - rf.reindex(r.index).fillna(0.0)


def total_return(r: pd.Series) -> float:
    return float(np.prod(1.0 + r.to_numpy()) - 1.0)


def cagr(r: pd.Series) -> float:
    if len(r) == 0:
        return np.nan
    return float(np.prod(1.0 + r.to_numpy()) ** (TRADING_DAYS / len(r)) - 1.0)


def ann_volatility(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(r: pd.Series, rf: pd.Series | None = None) -> float:
    ex = _excess(r, rf)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def sortino_ratio(r: pd.Series, rf: pd.Series | None = None) -> float:
    ex = _excess(r, rf)
    downside = np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0.0) ** 2))
    return float(ex.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1.0 + r).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min()) if len(r) else np.nan


def cvar(r: pd.Series, alpha: float = 0.05) -> float:
    """Expected daily return on the worst ``alpha`` share of days (negative number)."""
    q = r.quantile(alpha)
    return float(r[r <= q].mean())


def summarize(
    r: pd.Series,
    rf: pd.Series | None = None,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    r = r.dropna()
    mdd = max_drawdown(r)
    out = {
        "total_return": total_return(r),
        "cagr": cagr(r),
        "volatility": ann_volatility(r),
        "sharpe": sharpe_ratio(r, rf),
        "sortino": sortino_ratio(r, rf),
        "max_drawdown": mdd,
        "calmar": cagr(r) / abs(mdd) if mdd < 0 else np.nan,
        "cvar_95": cvar(r, 0.05),
    }
    if turnover is not None and len(r):
        # Rebalances whose holding period starts inside the window (the first one happens on
        # the last trading day before the window, hence the small calendar tolerance).
        first, last = r.index[0] - pd.Timedelta(days=7), r.index[-1]
        t = turnover[(turnover.index >= first) & (turnover.index < last)]
        years = len(r) / TRADING_DAYS
        out["turnover"] = float(t.sum() / 2.0 / years)  # one-way, annualized
    return out
