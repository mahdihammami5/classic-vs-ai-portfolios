"""Predict-then-optimize: a forecaster supplies expected returns, mean-variance does the rest.

All forecast strategies share the optimizer, covariance estimate, constraints and costs of
the phase-1 Markowitz baseline, so differences in results come only from the forecasts.
Each forecast is stored so its skill can be measured directly (rank IC, see
``pfopt.analysis``), independently of how it is turned into a portfolio.
"""

from __future__ import annotations

import pandas as pd

from pfopt.forecasters import build_forecaster
from pfopt.strategies.base import TRADING_DAYS, RebalanceContext, Strategy, shrunk_covariance
from pfopt.strategies.classic import solve_mean_variance


class ForecastMeanVariance(Strategy):
    def __init__(
        self,
        name: str,
        forecaster: dict,
        risk_aversion: float = 5.0,
        mean_shrinkage: float = 0.5,
        max_weight: float = 0.10,
    ):
        super().__init__(name)
        self.forecaster = build_forecaster(forecaster)
        self.risk_aversion = risk_aversion
        self.mean_shrinkage = mean_shrinkage
        self.max_weight = max_weight
        self.predictions: dict[pd.Timestamp, pd.Series] = {}
        self._returns: pd.DataFrame | None = None
        self._date: pd.Timestamp | None = None
        self._fitted_year: int | None = None

    def on_rebalance(self, ctx: RebalanceContext) -> None:
        self._returns, self._date = ctx.returns, ctx.date
        if self.forecaster.refit_yearly and ctx.date.year != self._fitted_year:
            self.forecaster.fit(ctx)
            self._fitted_year = ctx.date.year

    def allocate(self, hist, prev, rng):
        source = hist if self._returns is None else self._returns  # standalone use: hist only
        context = source[hist.columns].iloc[-self.forecaster.context :]
        forecast = self.forecaster.predict(context).reindex(hist.columns).fillna(0.0)
        self.predictions[self._date] = forecast
        mu = forecast.to_numpy() * TRADING_DAYS / self.forecaster.horizon  # annualized
        w = solve_mean_variance(mu, shrunk_covariance(hist), self.risk_aversion,
                                self.mean_shrinkage, self.max_weight, self.name)
        return pd.Series(w, index=hist.columns)

    def report(self) -> dict:
        return {
            "horizon": self.forecaster.horizon,
            "forecaster_log": self.forecaster.log,
            "predictions": pd.DataFrame(self.predictions).T,
        }
