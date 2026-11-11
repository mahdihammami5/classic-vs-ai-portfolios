"""Point-in-time S&P 500 membership and investable-universe selection.

The universe on a given date only uses information available on that date:
index membership on that date, and trailing prices/volumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class Membership:
    """S&P 500 constituents as a step function of time."""

    def __init__(self, dates: pd.DatetimeIndex, members: list[frozenset[str]]):
        order = np.argsort(dates.values)
        self.dates = pd.DatetimeIndex(dates.values[order])
        self.members = [members[i] for i in order]

    @classmethod
    def from_csv(cls, path: str | Path) -> Membership:
        df = pd.read_csv(path)
        dates = pd.to_datetime(df["date"])
        members = [frozenset(t.strip() for t in s.split(",") if t.strip()) for s in df["tickers"]]
        return cls(pd.DatetimeIndex(dates), members)

    @classmethod
    def from_dict(cls, mapping: dict[str, list[str]]) -> Membership:
        dates = pd.DatetimeIndex(pd.to_datetime(list(mapping)))
        return cls(dates, [frozenset(v) for v in mapping.values()])

    def members_at(self, date) -> frozenset[str]:
        pos = self.dates.searchsorted(pd.Timestamp(date), side="right") - 1
        return self.members[pos] if pos >= 0 else frozenset()

    def ever_members(self, start, end) -> set[str]:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        out = set(self.members_at(start))
        for d, m in zip(self.dates, self.members, strict=True):
            if start < d <= end:
                out |= m
        return out


def select_universe(
    date,
    membership: Membership,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    size: int,
    lookback: int,
    liquidity_window: int,
) -> list[str]:
    """Pick the ``size`` most liquid index members with a complete return history.

    Liquidity = mean daily dollar volume over the last ``liquidity_window`` days.
    A complete ``lookback`` window is required so every strategy sees the same history.
    """
    date = pd.Timestamp(date)
    members = [t for t in membership.members_at(date) if t in returns.columns]
    if not members:
        return []
    window = returns.loc[:date, members].iloc[-lookback:]
    if len(window) < lookback:
        return []
    complete = window.columns[window.notna().all()]
    liq = dollar_volume.loc[:date, complete].iloc[-liquidity_window:].mean()
    return sorted(liq.nlargest(size).index)
