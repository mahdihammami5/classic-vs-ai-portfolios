"""Experiment configuration loaded from YAML files in ``configs/``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    dir: str = "data"
    price_start: str = "2005-01-01"
    price_end: str | None = None  # None = today
    benchmark: str = "SPY"


@dataclass
class BacktestConfig:
    start: str = "2010-01-01"
    end: str | None = None  # None = last available date
    rebalance: str = "M"  # calendar schedule: "D", "W", "M", "Q" or "Y"
    lookback: int = 252  # trading days of history given to each strategy
    cost_bps: float = 10.0  # cost per unit of traded notional, in basis points
    universe_size: int = 100
    universe_refresh: str = "Y"  # "Y" re-select universe yearly, "R" at every rebalance
    liquidity_window: int = 63  # days used to rank stocks by dollar volume


@dataclass
class RebalancePolicy:
    """When a strategy trades. The default reproduces plain calendar rebalancing.

    * ``schedule`` overrides the backtest's calendar schedule for this strategy.
    * ``check`` adds extra dates ("D" or "W") on which the triggers below are evaluated.
    * ``drift``: rebalance when holdings have drifted more than this (one-way share of the
      portfolio) away from the last target.
    * ``vix_level``: rebalance when the VIX closes on the other side of this level than it
      was at the previous check (a change of market-stress regime).
    * ``band``: no-trade band. The strategy re-optimizes, but the trade is skipped when the
      new target is less than ``band`` (one-way) away from the current holdings.
    * ``min_gap``: minimum trading days between two trigger-driven rebalances.
    """

    schedule: str | None = None
    check: str | None = None
    drift: float | None = None
    vix_level: float | None = None
    band: float | None = None
    min_gap: int = 0

    @property
    def is_calendar_only(self) -> bool:
        return self.check is None and self.drift is None and self.vix_level is None


@dataclass
class StrategySpec:
    name: str
    type: str
    params: dict = field(default_factory=dict)
    rebalance: RebalancePolicy = field(default_factory=RebalancePolicy)

    def __post_init__(self):
        if isinstance(self.rebalance, dict):
            self.rebalance = RebalancePolicy(**self.rebalance)


@dataclass
class ExperimentConfig:
    experiment: str
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    periods: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    strategies: list[StrategySpec] = field(default_factory=list)
    output_dir: str = "results"
    seed: int = 42


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        experiment=raw["experiment"],
        data=DataConfig(**raw.get("data", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        periods={k: (v[0], v[1]) for k, v in raw.get("periods", {}).items()},
        strategies=[StrategySpec(**s) for s in raw.get("strategies", [])],
        output_dir=raw.get("output_dir", "results"),
        seed=raw.get("seed", 42),
    )
