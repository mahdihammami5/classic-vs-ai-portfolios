"""Return forecasters for predict-then-optimize strategies (phase 3).

Every forecaster maps a context of past daily returns (days x stocks) to an expected
return over the next ``horizon`` trading days for each stock. Learning forecasters may
also be refit on a ``RebalanceContext`` (which only contains past data).
"""

from __future__ import annotations

import logging
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from pfopt.strategies.base import RebalanceContext

log = logging.getLogger(__name__)


class Forecaster(ABC):
    refit_yearly = False

    def __init__(self, horizon: int = 21, context: int = 252):
        self.horizon = horizon
        self.context = context
        self.log: list[dict] = []

    def fit(self, ctx: RebalanceContext) -> None:  # noqa: B027 - optional hook
        """Retrain on data available at ``ctx.date``. Default: nothing to learn."""

    @abstractmethod
    def predict(self, returns: pd.DataFrame) -> pd.Series:
        """``returns``: (>= context days x stocks), may contain leading NaNs.
        Returns the expected simple return over the next ``horizon`` days per stock."""


class HistoricalMean(Forecaster):
    """Sample mean of daily returns x horizon. With a 252-day context and the same
    optimizer this reproduces the phase-1 Markowitz strategy exactly."""

    def predict(self, returns):
        return returns.iloc[-self.context :].mean() * self.horizon


class Momentum(Forecaster):
    """12-1 month momentum (Jegadeesh & Titman, 1993), expressed per ``horizon`` days:
    the past 11-month return, skipping the latest month, is assumed to continue."""

    def __init__(self, horizon: int = 21, context: int = 252, skip: int = 21):
        super().__init__(horizon, context)
        self.skip = skip

    def predict(self, returns):
        window = returns.iloc[-self.context : -self.skip].fillna(0.0)
        past = np.expm1(np.log1p(window.clip(lower=-0.99)).sum())
        months = len(window) / self.horizon
        return (1.0 + past) ** (1.0 / months) - 1.0


def growth_series(returns: pd.DataFrame) -> list[np.ndarray]:
    """Turn each column of returns into a price-like index (starting at 1), dropping the
    leading NaNs of stocks with shorter history. Chronos models forecast levels."""
    out = []
    for col in returns.columns:
        r = returns[col]
        first = r.first_valid_index()
        r = r.loc[first:].fillna(0.0).to_numpy() if first is not None else np.zeros(1)
        out.append(np.cumprod(1.0 + r).astype(np.float32))
    return out


class Chronos2(Forecaster):
    """Amazon Chronos-2 time-series foundation model (Ansari et al., 2025).

    * zero-shot (``finetune=None``): the pretrained model forecasts each stock's price index;
    * fine-tuned (``finetune="lora"``): every January a LoRA adapter is trained from the base
      model on the current universe's price history up to that date (no future data).

    The expected return is the mean forecast ``horizon`` days ahead over the last price.
    Caveat: the pretraining corpus of a 2025 model may include financial series that overlap
    the backtest period, a look-ahead risk shared by all foundation models.
    """

    def __init__(
        self,
        horizon: int = 21,
        context: int = 512,
        model_id: str = "amazon/chronos-2",
        cross_learning: bool = False,
        finetune: str | None = None,
        finetune_steps: int = 300,
        finetune_lr: float = 1e-4,
        finetune_batch_size: int = 32,
        cache_dir: str = ".cache/chronos_finetune",
        device: str = "auto",
        seed: int = 0,
        pipeline=None,  # injectable for tests
    ):
        super().__init__(horizon, context)
        self.model_id, self.cross_learning = model_id, cross_learning
        self.finetune, self.finetune_steps = finetune, finetune_steps
        self.finetune_lr, self.finetune_batch_size = finetune_lr, finetune_batch_size
        self.cache_dir, self.seed = Path(cache_dir), seed
        self.refit_yearly = finetune is not None
        self.device = device
        self._base = pipeline
        self._pipe = pipeline

    def _load(self):
        if self._base is None:
            import torch
            from chronos import Chronos2Pipeline

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._base = Chronos2Pipeline.from_pretrained(self.model_id, device_map=device,
                                                          dtype=torch.float32)
            self._pipe = self._base
        return self._pipe

    def fit(self, ctx: RebalanceContext) -> None:
        if self.finetune is None:
            return
        self._load()
        t0 = time.perf_counter()
        history = ctx.returns[ctx.universe]
        series = [s for s in growth_series(history) if len(s) > self.context // 2]
        out_dir = self.cache_dir / f"{ctx.date.date()}"
        self._pipe = self._base.fit(
            series, prediction_length=self.horizon, context_length=self.context,
            finetune_mode=self.finetune, learning_rate=self.finetune_lr,
            num_steps=self.finetune_steps, batch_size=self.finetune_batch_size,
            output_dir=str(out_dir), remove_printer_callback=True, report_to=[],
            seed=self.seed, save_strategy="no",
        )
        shutil.rmtree(out_dir, ignore_errors=True)  # the adapter stays in memory
        self.log.append({"trained_on": str(ctx.date.date()), "series": len(series),
                         "steps": self.finetune_steps,
                         "seconds": round(time.perf_counter() - t0, 1)})
        log.info("Chronos-2 %s fine-tune up to %s: %d series, %.0fs", self.finetune,
                 ctx.date.date(), len(series), time.perf_counter() - t0)

    def predict(self, returns):
        pipe = self._load()
        series = growth_series(returns.iloc[-self.context :])
        _, mean = pipe.predict_quantiles(
            [s[None, :] for s in series], prediction_length=self.horizon,
            quantile_levels=[0.5], cross_learning=self.cross_learning,
        )
        last = np.array([s[-1] for s in series])
        end = np.array([float(m[0, -1]) for m in mean])
        return pd.Series(end / last - 1.0, index=returns.columns)


FORECASTERS = {"historical_mean": HistoricalMean, "momentum": Momentum, "chronos2": Chronos2}


def build_forecaster(spec: dict) -> Forecaster:
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in FORECASTERS:
        raise KeyError(f"Unknown forecaster {kind!r}. Available: {sorted(FORECASTERS)}")
    return FORECASTERS[kind](**spec)
