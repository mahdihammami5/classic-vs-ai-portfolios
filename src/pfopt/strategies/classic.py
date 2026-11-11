"""Classic portfolio construction baselines."""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from pfopt.strategies.base import TRADING_DAYS, Strategy, cap_weights, shrunk_covariance

log = logging.getLogger(__name__)


def _cholesky(sigma: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(sigma + 1e-10 * np.eye(len(sigma)))


def _solve(problem: cp.Problem, var: cp.Variable, name: str) -> np.ndarray | None:
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        problem.solve(solver=cp.SCS)
    if var.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        log.warning("%s: solver status %s, falling back to equal weight", name, problem.status)
        return None
    return np.asarray(var.value).ravel()


class EqualWeight(Strategy):
    """1/N: a famously hard-to-beat baseline (DeMiguel, Garlappi & Uppal, 2009)."""

    def allocate(self, hist, prev, rng):
        return pd.Series(1.0 / hist.shape[1], index=hist.columns)


class InverseVolatility(Strategy):
    def allocate(self, hist, prev, rng):
        inv = 1.0 / hist.std(ddof=1).replace(0.0, np.nan)
        return inv.fillna(0.0) / inv.sum()


class MinVariance(Strategy):
    """Long-only global minimum variance with a per-asset cap."""

    def __init__(self, name: str, max_weight: float = 0.10):
        super().__init__(name)
        self.max_weight = max_weight

    def allocate(self, hist, prev, rng):
        n = hist.shape[1]
        L = _cholesky(shrunk_covariance(hist))
        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(L.T @ w)),
            [cp.sum(w) == 1, w >= 0, w <= self.max_weight],
        )
        sol = _solve(prob, w, self.name)
        sol = np.full(n, 1.0 / n) if sol is None else cap_weights(sol, self.max_weight)
        return pd.Series(sol, index=hist.columns)


class MeanVariance(Strategy):
    """Markowitz mean-variance utility: max  mu'w - (gamma/2) w'Sigma w.

    Expected returns are shrunk towards the cross-sectional average, because raw sample
    means are the main reason Markowitz portfolios fail out of sample.
    """

    def __init__(
        self,
        name: str,
        risk_aversion: float = 5.0,
        mean_shrinkage: float = 0.5,
        max_weight: float = 0.10,
    ):
        super().__init__(name)
        self.risk_aversion = risk_aversion
        self.mean_shrinkage = mean_shrinkage
        self.max_weight = max_weight

    def allocate(self, hist, prev, rng):
        mu_hat = hist.mean().to_numpy() * TRADING_DAYS
        w = solve_mean_variance(mu_hat, shrunk_covariance(hist), self.risk_aversion,
                                self.mean_shrinkage, self.max_weight, self.name)
        return pd.Series(w, index=hist.columns)


def solve_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    mean_shrinkage: float,
    max_weight: float,
    name: str = "mean_variance",
) -> np.ndarray:
    """Long-only, capped  max mu'w - (gamma/2) w'Sigma w  with mu shrunk to its average.

    ``mu`` and ``sigma`` are annualized. Shared by every predict-then-optimize strategy so
    that only the source of expected returns differs between them.
    """
    n = len(mu)
    mu = (1 - mean_shrinkage) * mu + mean_shrinkage * mu.mean()
    L = _cholesky(sigma)
    w = cp.Variable(n)
    utility = mu @ w - 0.5 * risk_aversion * cp.sum_squares(L.T @ w)
    prob = cp.Problem(cp.Maximize(utility), [cp.sum(w) == 1, w >= 0, w <= max_weight])
    sol = _solve(prob, w, name)
    return np.full(n, 1.0 / n) if sol is None else cap_weights(sol, max_weight)


class RiskParity(Strategy):
    """Equal risk contribution: every asset contributes the same share of portfolio variance.

    Solved through the convex reformulation  min 0.5 y'Sigma y - (1/n) sum log y,
    then w = y / sum(y)  (Spinu, 2013).
    """

    def allocate(self, hist, prev, rng):
        n = hist.shape[1]
        L = _cholesky(shrunk_covariance(hist))
        y = cp.Variable(n, pos=True)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(L.T @ y) - cp.sum(cp.log(y)) / n))
        sol = _solve(prob, y, self.name)
        sol = np.full(n, 1.0 / n) if sol is None else np.clip(sol, 0.0, None)
        return pd.Series(sol / sol.sum(), index=hist.columns)
