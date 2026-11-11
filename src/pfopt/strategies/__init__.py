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
from pfopt.strategies.forecast import ForecastMeanVariance
from pfopt.strategies.genetic import GeneticWoC

REGISTRY: dict[str, type[Strategy]] = {
    "forecast_mean_variance": ForecastMeanVariance,
    "equal_weight": EqualWeight,
    "inverse_volatility": InverseVolatility,
    "min_variance": MinVariance,
    "mean_variance": MeanVariance,
    "risk_parity": RiskParity,
    "genetic_woc": GeneticWoC,
}


# Strategies with heavy optional dependencies (PyTorch) are imported only when used.
LAZY = {"deep_portfolio": ("pfopt.strategies.deep", "DeepPortfolio")}


def build_strategy(spec: StrategySpec) -> Strategy:
    if spec.type in LAZY:
        import importlib

        module, cls = LAZY[spec.type]
        return getattr(importlib.import_module(module), cls)(spec.name, **spec.params)
    if spec.type not in REGISTRY:
        known = sorted([*REGISTRY, *LAZY])
        raise KeyError(f"Unknown strategy type {spec.type!r}. Available: {known}")
    return REGISTRY[spec.type](spec.name, **spec.params)


__all__ = ["REGISTRY", "Strategy", "build_strategy"]
