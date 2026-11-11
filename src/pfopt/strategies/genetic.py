"""Genetic Algorithm + Wisdom of Crowds (GA+WoC) portfolio optimizer.

This is the project's original method (see ``legacy/``), rewritten so it can be compared
fairly with the other strategies:

* fitness = Sharpe ratio on the trailing window, with a shrunk covariance matrix;
* vectorized population evaluation (the whole population is one matrix);
* tournament selection, blend crossover and sparse Gaussian mutation, which unlike
  single-point crossover do not depend on the arbitrary column order of tickers;
* the same per-asset cap as the other optimizers;
* the previous portfolio is seeded into the population, so the search starts from what
  is already held (lower turnover);
* Wisdom of Crowds: the final portfolio is the average of the ``crowd_size`` fittest
  individuals instead of the single best one, which averages out estimation noise;
* optional cost-aware fitness: ``Sharpe - turnover_penalty x one-way turnover`` from the
  current holdings, so the search only moves away from them when the gain is worth it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pfopt.strategies.base import TRADING_DAYS, Strategy, cap_weights, shrunk_covariance


class GeneticWoC(Strategy):
    def __init__(
        self,
        name: str,
        population: int = 100,
        generations: int = 150,
        tournament: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.05,
        mutation_scale: float = 0.02,
        elite: int = 2,
        init_assets: int = 20,
        crowd_size: int = 10,
        woc: bool = True,
        max_weight: float = 0.10,
        turnover_penalty: float = 0.0,
    ):
        super().__init__(name)
        self.population = population
        self.generations = generations
        self.tournament = tournament
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elite = elite
        self.init_assets = init_assets
        self.crowd_size = crowd_size
        self.woc = woc
        self.max_weight = max_weight
        self.turnover_penalty = turnover_penalty
        self.last_progress: list[float] = []

    @staticmethod
    def sharpe(W: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        ret = W @ mu
        vol = np.sqrt(np.einsum("ij,jk,ik->i", W, sigma, W))
        return ret / np.maximum(vol, 1e-12)

    def fitness(
        self, W: np.ndarray, mu: np.ndarray, sigma: np.ndarray, prev: np.ndarray
    ) -> np.ndarray:
        fit = self.sharpe(W, mu, sigma)
        if self.turnover_penalty > 0:
            # Holdings outside the current universe (prev sums to < 1) must be sold anyway.
            outside = max(0.0, 1.0 - float(prev.sum()))
            turnover = 0.5 * (np.abs(W - prev).sum(axis=1) + outside)
            fit = fit - self.turnover_penalty * turnover
        return fit

    def _init_population(self, n: int, prev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        P = self.population
        k = min(max(self.init_assets, int(np.ceil(1 / self.max_weight))), n)
        pop = np.zeros((P, n))
        for i in range(P):
            assets = rng.choice(n, size=k, replace=False)
            pop[i, assets] = rng.dirichlet(np.ones(k))
        pop[0] = 1.0 / n  # equal weight
        if prev.sum() > 0:
            pop[1] = prev
        return cap_weights(pop, self.max_weight)

    def evolve(
        self, mu: np.ndarray, sigma: np.ndarray, prev: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(mu)
        P = self.population
        pop = self._init_population(n, prev, rng)
        n_children = P - self.elite
        progress = []
        for _ in range(self.generations):
            fit = self.fitness(pop, mu, sigma, prev)
            progress.append(float(fit.max()))
            elites = pop[np.argsort(fit)[-self.elite :]]

            contenders = rng.integers(0, P, size=(2 * n_children, self.tournament))
            winners = contenders[np.arange(2 * n_children), fit[contenders].argmax(axis=1)]
            p1, p2 = pop[winners[:n_children]], pop[winners[n_children:]]

            alpha = rng.random((n_children, 1))
            do_cross = rng.random((n_children, 1)) < self.crossover_rate
            children = np.where(do_cross, alpha * p1 + (1 - alpha) * p2, p1)

            mask = rng.random(children.shape) < self.mutation_rate
            children = children + mask * rng.normal(0.0, self.mutation_scale, children.shape)
            pop = np.vstack([elites, cap_weights(children, self.max_weight)])

        fit = self.fitness(pop, mu, sigma, prev)
        self.last_progress = progress
        return pop, fit

    def allocate(self, hist, prev, rng):
        mu = hist.mean().to_numpy() * TRADING_DAYS
        sigma = shrunk_covariance(hist)
        pop, fit = self.evolve(mu, sigma, prev.to_numpy(), rng)
        if self.woc:
            crowd = pop[np.argsort(fit)[-self.crowd_size :]]
            w = cap_weights(crowd.mean(axis=0), self.max_weight)
        else:
            w = pop[fit.argmax()]
        return pd.Series(w, index=hist.columns)
