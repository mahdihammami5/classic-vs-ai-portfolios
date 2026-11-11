"""Strategy registry: maps the ``type`` field of a config to a class."""

from __future__ import annotations

from pfopt.config import StrategySpec
from pfopt.strategies.base import Strategy
from pfopt.strategies.classic import (
    EqualWeight,
    InverseVolatility,
    MeanVariance,
    MinVariance,
    RiskParity,
)
from pfopt.strategies.genetic import GeneticWoC

REGISTRY: dict[str, type[Strategy]] = {
    "equal_weight": EqualWeight,
    "inverse_volatility": InverseVolatility,
    "min_variance": MinVariance,
    "mean_variance": MeanVariance,
    "risk_parity": RiskParity,
    "genetic_woc": GeneticWoC,
}


def build_strategy(spec: StrategySpec) -> Strategy:
    if spec.type not in REGISTRY:
        raise KeyError(f"Unknown strategy type {spec.type!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[spec.type](spec.name, **spec.params)


__all__ = ["REGISTRY", "Strategy", "build_strategy"]
