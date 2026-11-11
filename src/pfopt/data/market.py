"""Build the local dataset and load it back as a single ``MarketData`` object."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pfopt.config import DataConfig
from pfopt.data import sources
from pfopt.data.universe import Membership

log = logging.getLogger(__name__)

VIX = "^VIX"


@dataclass
class MarketData:
    returns: pd.DataFrame  # daily simple returns, NaN where a stock has no price
    dollar_volume: pd.DataFrame
    rf: pd.Series  # daily risk-free rate aligned to ``returns.index``
    membership: Membership
    benchmark: pd.Series  # daily benchmark (SPY) total returns

    @classmethod
    def load(cls, data_dir: str | Path) -> MarketData:
        processed = Path(data_dir) / "processed"
        close = pd.read_parquet(processed / "close.parquet")
        volume = pd.read_parquet(processed / "volume.parquet")
        bench = pd.read_parquet(processed / "benchmark.parquet")
        rf = pd.read_parquet(processed / "rf.parquet")["rf"]
        membership = Membership.from_csv(Path(data_dir) / "raw" / "sp500_components.csv")

        returns = close.pct_change(fill_method=None).iloc[1:]
        dollar_volume = (close * volume).iloc[1:]
        rf = rf.reindex(returns.index).ffill().fillna(0.0)
        benchmark = bench.iloc[:, 0].pct_change(fill_method=None).reindex(returns.index)
        return cls(returns, dollar_volume, rf, membership, benchmark.rename(bench.columns[0]))


def build_dataset(cfg: DataConfig) -> None:
    """Download every source and write ``data/processed/*.parquet`` plus a coverage report."""
    raw = Path(cfg.dir) / "raw"
    processed = Path(cfg.dir) / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    log.info("Downloading S&P 500 membership history")
    membership = Membership.from_csv(sources.download_membership(raw))
    end = cfg.price_end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sorted(membership.ever_members(cfg.price_start, end))
    log.info("%d tickers were index members between %s and %s", len(tickers), cfg.price_start, end)

    close, volume = sources.download_prices(tickers, cfg.price_start, cfg.price_end)
    close.to_parquet(processed / "close.parquet")
    volume.to_parquet(processed / "volume.parquet")
    log.info("Got prices for %d of %d tickers", close.shape[1], len(tickers))

    bench_close, _ = sources.download_prices([cfg.benchmark, VIX], cfg.price_start, cfg.price_end)
    bench_close[[cfg.benchmark]].to_parquet(processed / "benchmark.parquet")
    if VIX in bench_close:
        bench_close[[VIX]].rename(columns={VIX: "VIX"}).to_parquet(processed / "vix.parquet")

    log.info("Downloading risk-free rate")
    sources.download_risk_free(raw).to_frame().to_parquet(processed / "rf.parquet")

    coverage = coverage_report(membership, close)
    coverage.to_csv(processed / "coverage.csv")
    log.info("Membership coverage by year:\n%s", coverage.to_string())


def coverage_report(membership: Membership, close: pd.DataFrame) -> pd.DataFrame:
    """Share of index members on each year's first trading day that have a Yahoo price.

    Missing members are mostly companies that were later delisted or acquired:
    this is the residual survivorship bias of free data.
    """
    rows = []
    for year in sorted(close.index.year.unique()):
        day = close.index[close.index.year == year][0]
        members = membership.members_at(day)
        have = sum(1 for t in members if t in close.columns and pd.notna(close.at[day, t]))
        rows.append({"year": year, "members": len(members), "with_prices": have})
    df = pd.DataFrame(rows).set_index("year")
    df["coverage"] = (df["with_prices"] / df["members"]).round(3)
    return df
