import numpy as np
import pandas as pd
import pytest

from pfopt.config import StrategySpec
from pfopt.strategies import REGISTRY, build_strategy
from pfopt.strategies.base import cap_weights, shrunk_covariance
from pfopt.strategies.classic import MinVariance, RiskParity
from pfopt.strategies.genetic import GeneticWoC

from .conftest import make_market


@pytest.fixture
def hist():
    return make_market(n_assets=15, n_days=300, seed=1).returns


@pytest.mark.parametrize("kind", sorted(REGISTRY))
def test_every_strategy_returns_valid_long_only_weights(kind, hist):
    params = {"generations": 20, "population": 30} if kind == "genetic_woc" else {}
    strat = build_strategy(StrategySpec(name=kind, type=kind, params=params))
    w = strat.allocate(hist, pd.Series(0.0, index=hist.columns), np.random.default_rng(0))
    w = w / w.sum()
    assert list(w.index) == list(hist.columns)
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0)


def test_cap_weights_respects_cap_and_budget():
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(30) * 0.2, size=50)
    capped = cap_weights(W, 0.1)
    assert np.all(capped <= 0.1 + 1e-9)
    np.testing.assert_allclose(capped.sum(axis=1), 1.0)


def test_cap_weights_fills_sparse_portfolios():
    w = np.zeros(20)
    w[:3] = [0.5, 0.3, 0.2]  # only 3 assets but cap 0.1 needs at least 10
    capped = cap_weights(w, 0.1)
    assert capped.max() <= 0.1 + 1e-9 and capped.sum() == pytest.approx(1.0)


def test_cap_weights_rejects_infeasible_cap():
    with pytest.raises(ValueError):
        cap_weights(np.ones(5) / 5, 0.1)


def test_min_variance_prefers_low_volatility_assets(hist):
    w = MinVariance("mv", max_weight=0.5).allocate(hist, None, None)
    vols = hist.std()
    assert w[vols.idxmin()] > w[vols.idxmax()]


def test_risk_parity_equalizes_risk_contributions(hist):
    w = RiskParity("rp").allocate(hist, None, None).to_numpy()
    sigma = shrunk_covariance(hist)
    contrib = w * (sigma @ w)
    contrib /= contrib.sum()
    np.testing.assert_allclose(contrib, 1 / len(w), atol=1e-4)


def test_genetic_algorithm_improves_fitness_and_is_reproducible(hist):
    ga = GeneticWoC("ga", population=40, generations=60, woc=False)
    prev = pd.Series(0.0, index=hist.columns)
    w1 = ga.allocate(hist, prev, np.random.default_rng(7))
    assert ga.last_progress[-1] >= ga.last_progress[0]  # elitism: never gets worse
    w2 = ga.allocate(hist, prev, np.random.default_rng(7))
    pd.testing.assert_series_equal(w1, w2)
