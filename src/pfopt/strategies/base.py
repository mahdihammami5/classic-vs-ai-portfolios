"""Strategy interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


class Strategy(ABC):
    """A long-only, fully invested allocation rule.

    ``allocate`` receives only past data: ``hist`` is a (days x assets) frame of daily
    returns ending on the rebalance date, with no missing values.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def allocate(
        self,
        hist: pd.DataFrame,
        prev: pd.Series,
        rng: np.random.Generator,
    ) -> pd.Series:
        """Return target weights indexed by ticker (non-negative, any positive scale)."""


def shrunk_covariance(hist: pd.DataFrame) -> np.ndarray:
    """Annualized Ledoit-Wolf covariance: stable even when assets ~ observations."""
    return LedoitWolf().fit(hist.to_numpy()).covariance_ * TRADING_DAYS


def cap_weights(w: np.ndarray, cap: float | None) -> np.ndarray:
    """Project each row onto {w >= 0, sum(w) = 1, w <= cap}, keeping proportions where possible.

    Works on a 1-D vector or a 2-D (rows = portfolios) array. Excess weight above the cap is
    redistributed in proportion to existing weights; if those cannot absorb it, it is spread
    equally across all uncapped assets.
    """
    single = w.ndim == 1
    W = np.atleast_2d(np.asarray(w, dtype=float)).copy()
    W = np.clip(W, 0.0, None)
    sums = W.sum(axis=1, keepdims=True)
    W = np.where(sums > 0, W / np.where(sums > 0, sums, 1.0), 1.0 / W.shape[1])
    if cap is None or cap >= 1.0:
        return W[0] if single else W
    if cap * W.shape[1] < 1.0 - 1e-12:
        raise ValueError(f"max_weight={cap} is infeasible for {W.shape[1]} assets.")
    for _ in range(200):
        excess = np.clip(W - cap, 0.0, None).sum(axis=1, keepdims=True)
        if excess.max() <= 1e-12:
            break
        W = np.minimum(W, cap)
        free = W < cap - 1e-12
        pos_free = free & (W > 0)
        room_pos = ((cap - W) * pos_free).sum(axis=1, keepdims=True)
        mass_pos = (W * pos_free).sum(axis=1, keepdims=True)
        proportional = W * pos_free / np.where(mass_pos > 0, mass_pos, 1.0)
        equal = free / np.maximum(free.sum(axis=1, keepdims=True), 1)
        share = np.where((room_pos >= excess) & (mass_pos > 0), proportional, equal)
        W = W + share * excess
    return W[0] if single else W
