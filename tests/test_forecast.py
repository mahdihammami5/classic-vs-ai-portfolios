import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pfopt.analysis import information_coefficients, summarize_ic  # noqa: E402
from pfopt.backtest.engine import run_backtest  # noqa: E402
from pfopt.config import BacktestConfig  # noqa: E402
from pfopt.forecasters import Chronos2, HistoricalMean, Momentum, growth_series  # noqa: E402
from pfopt.reporting import save_run  # noqa: E402
from pfopt.strategies.classic import MeanVariance  # noqa: E402
from pfopt.strategies.forecast import ForecastMeanVariance  # noqa: E402

from .conftest import make_market  # noqa: E402


@pytest.fixture
def cfg():
    return BacktestConfig(start="2016-01-01", lookback=252, universe_size=8, liquidity_window=20)


def test_historical_mean_forecaster_reproduces_markowitz(cfg):
    market = make_market(n_assets=8, n_days=600)
    mv = run_backtest(MeanVariance("mv"), market, cfg)
    fmv = run_backtest(
        ForecastMeanVariance("fmv", {"type": "historical_mean", "context": 252}), market, cfg)
    pd.testing.assert_series_equal(mv.returns, fmv.returns, check_names=False, atol=1e-8)


def test_momentum_ranks_past_winners_first():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.DataFrame({"winner": 0.002, "flat": 0.0, "loser": -0.002}, index=idx)
    f = Momentum(context=252).predict(r)
    assert f["winner"] > f["flat"] > f["loser"]
    assert f["flat"] == pytest.approx(0.0)


def test_growth_series_drops_leading_nans():
    r = pd.DataFrame({"a": [np.nan, np.nan, 0.1, 0.1], "b": [0.0, 0.1, 0.0, -0.5]})
    a, b = growth_series(r)
    np.testing.assert_allclose(a, [1.1, 1.21], rtol=1e-6)
    assert len(b) == 4 and b[-1] == pytest.approx(0.55)


class FakePipeline:
    """Stands in for Chronos2Pipeline: forecasts +10% for every series, records fits."""

    def __init__(self):
        self.fits = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, cross_learning):
        mean = [torch.full((1, prediction_length), float(x[0, -1]) * 1.1) for x in inputs]
        return None, mean

    def fit(self, series, **kwargs):
        self.fits.append((len(series), kwargs["prediction_length"]))
        return self


def test_chronos_wrapper_converts_level_forecasts_to_returns(tmp_path):
    fake = FakePipeline()
    model = Chronos2(context=60, pipeline=fake, cache_dir=str(tmp_path))
    r = make_market(n_assets=4, n_days=100).returns
    f = model.predict(r)
    np.testing.assert_allclose(f.to_numpy(), 0.1, rtol=1e-5)


def test_chronos_finetunes_once_per_year(cfg, tmp_path):
    fake = FakePipeline()
    strat = ForecastMeanVariance("fm", {"type": "momentum"})
    strat.forecaster = Chronos2(context=60, finetune="lora", pipeline=fake,
                                cache_dir=str(tmp_path))
    market = make_market(n_assets=8, n_days=600)
    res = run_backtest(strat, market, cfg)
    years = {d.year for d in res.weights.index}
    assert len(fake.fits) == len(years)


def test_information_coefficient_extremes():
    market = make_market(n_assets=10, n_days=300)
    R = market.returns
    dates = R.index[[50, 100, 150]]
    perfect = pd.DataFrame({d: (1 + R.loc[R.index > d].iloc[:21]).prod() - 1 for d in dates}).T
    ic = information_coefficients(perfect, R, 21)
    np.testing.assert_allclose(ic.to_numpy(), 1.0)
    ic_rev = information_coefficients(-perfect, R, 21)
    np.testing.assert_allclose(ic_rev.to_numpy(), -1.0)
    assert summarize_ic(ic)["hit_rate"] == 1.0


def test_save_run_writes_forecast_skill(cfg, tmp_path):
    market = make_market(n_assets=8, n_days=600)
    res = run_backtest(ForecastMeanVariance("mom", {"type": "momentum"}), market, cfg)
    periods = {"full": ("2016-01-01", None)}
    save_run(tmp_path, [res], market, periods, {})
    assert (tmp_path / "forecast_ic.csv").exists()
    assert "Forecast skill" in (tmp_path / "metrics.md").read_text(encoding="utf-8")
    json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))  # serializable


def test_historical_mean_matches_sample_mean():
    r = make_market(n_assets=3, n_days=300).returns
    f = HistoricalMean(horizon=21, context=252).predict(r)
    np.testing.assert_allclose(f.to_numpy(), r.iloc[-252:].mean().to_numpy() * 21)
