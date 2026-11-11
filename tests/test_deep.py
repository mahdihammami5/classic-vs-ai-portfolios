import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pfopt.backtest.engine import run_backtest  # noqa: E402
from pfopt.config import BacktestConfig  # noqa: E402
from pfopt.strategies.deep import (  # noqa: E402
    VOL_WINDOW,
    DeepPortfolio,
    PortfolioNet,
    make_features,
    portfolio_loss,
)

from .conftest import make_market  # noqa: E402


def tiny(**kw):
    params = dict(seq_len=20, horizon=5, sample_step=5, train_start="2015-01-01", hidden=8,
                  heads=2, epochs=3, batch_size=4, patience=2, n_models=2, max_weight=0.5,
                  device="cpu")
    return DeepPortfolio("deep", **{**params, **kw})


def test_features_shape_and_alignment():
    R = np.random.default_rng(0).normal(0, 0.01, (40 + VOL_WINDOW - 1, 7))
    f = make_features(R, 40)
    assert f.shape == (7, 40, 3)
    np.testing.assert_allclose(f[:, -1, 0], np.clip(R[-1] * 20, -5, 5), rtol=1e-6)


def test_network_outputs_long_only_weights_and_respects_mask():
    net = PortfolioNet(hidden=8, heads=2)
    x = torch.randn(3, 6, 10, 3)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[:, -2:] = False
    w = net(x, mask)
    assert torch.allclose(w.sum(-1), torch.ones(3))
    assert (w[:, -2:] == 0).all()


def test_network_is_permutation_equivariant():
    net = PortfolioNet(hidden=8, heads=2).eval()
    x = torch.randn(1, 5, 10, 3)
    perm = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        torch.testing.assert_close(net(x)[0, perm], net(x[:, perm])[0], rtol=1e-5, atol=1e-6)


def test_loss_prefers_the_better_asset():
    future = torch.zeros(1, 50, 2)
    future[0, :, 0] = torch.linspace(0.001, 0.003, 50)  # steady gains
    future[0, :, 1] = torch.linspace(-0.02, 0.02, 50)  # noisy
    good = portfolio_loss(torch.tensor([[0.9, 0.1]]), future, 0.1, 1.0, 0.0)
    bad = portfolio_loss(torch.tensor([[0.1, 0.9]]), future, 0.1, 1.0, 0.0)
    assert good < bad


def test_training_samples_never_use_future_returns():
    market = make_market(n_assets=6, n_days=500)
    strat = tiny()
    cfg = BacktestConfig(start="2016-01-01", lookback=60, universe_size=6, liquidity_window=20)
    run_backtest(strat, market, cfg)
    for entry in strat.training_log:
        trained_on = pd.Timestamp(entry["trained_on"])
        last_sample = pd.Timestamp(entry["last_val_sample"])
        # targets cover the `horizon` trading days after the last sample, all before training
        after = market.returns.index[market.returns.index > last_sample][: strat.horizon]
        assert after[-1] <= trained_on


def test_deep_strategy_runs_in_backtest_and_is_reproducible():
    market = make_market(n_assets=6, n_days=500)
    cfg = BacktestConfig(start="2016-01-01", lookback=60, universe_size=6, liquidity_window=20)
    a = run_backtest(tiny(), market, cfg)
    b = run_backtest(tiny(), market, cfg)
    assert a.weights.shape[0] > 3
    np.testing.assert_allclose(a.weights.sum(axis=1), 1.0)
    pd.testing.assert_series_equal(a.returns, b.returns)
