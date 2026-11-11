"""Forecast skill: the rank information coefficient (IC).

IC on a date = Spearman rank correlation, across stocks, between the forecast made on that
date and the return each stock actually earned over the next ``horizon`` trading days.
It measures whether a model ranks stocks correctly, independently of portfolio construction.
Rule of thumb in equity research: a mean IC of 0.02-0.05 that is stable over time is useful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(returns: pd.DataFrame, date, tickers, horizon: int) -> pd.Series | None:
    after = returns.loc[returns.index > pd.Timestamp(date), list(tickers)].iloc[:horizon]
    if len(after) < horizon:
        return None
    return (1.0 + after.fillna(0.0)).prod() - 1.0


def information_coefficients(
    predictions: pd.DataFrame, returns: pd.DataFrame, horizon: int
) -> pd.Series:
    """One rank IC per forecast date (dates without a full realized horizon are skipped)."""
    out = {}
    for date, pred in predictions.iterrows():
        pred = pred.dropna()
        realized = forward_returns(returns, date, pred.index, horizon)
        if realized is None or pred.nunique() < 3:
            continue
        out[date] = pred.rank().corr(realized.rank())
    return pd.Series(out, name="ic", dtype=float)


def summarize_ic(ic: pd.Series) -> dict[str, float]:
    ic = ic.dropna()
    n = len(ic)
    sd = ic.std(ddof=1) if n > 1 else np.nan
    return {
        "mean_ic": float(ic.mean()) if n else np.nan,
        "ic_std": float(sd),
        "t_stat": float(ic.mean() / sd * np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
        "hit_rate": float((ic > 0).mean()) if n else np.nan,
        "n_dates": n,
    }
