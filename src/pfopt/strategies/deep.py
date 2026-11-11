"""End-to-end deep learning portfolio (phase 2).

Instead of forecasting returns and then optimizing, the network outputs portfolio weights
directly and is trained on the quantity investors care about: the risk-adjusted return of
the portfolio it builds (Zhang et al., 2020; Fernandes & Desell, 2026).

Architecture (universe-agnostic, so it works although the stock list changes every year):

    per-stock history (60 days x 3 features)
        -> shared LSTM + temporal attention pooling   (one embedding per stock)
        -> Transformer encoder layer across stocks     (stocks attend to each other)
        -> linear score per stock -> softmax           (long-only weights summing to 1)

Loss on the next ``horizon`` days of realized returns:

    -Sharpe(portfolio)  +  cvar_weight * CVaR_5%  +  cap_penalty * sum(max(w - cap, 0)^2)

Walk-forward training: the model is retrained every January using only samples whose
target window ends before the rebalance date, with a chronological validation split and
an embargo gap for early stopping. Several models with different seeds are averaged
(a deep ensemble, the neural counterpart of the Wisdom of Crowds).
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from pfopt.strategies.base import TRADING_DAYS, RebalanceContext, Strategy, cap_weights

log = logging.getLogger(__name__)

VOL_WINDOW = 20
N_FEATURES = 3


def make_features(returns: np.ndarray, seq_len: int) -> np.ndarray:
    """Turn a (seq_len + VOL_WINDOW - 1, N) block of daily returns into (N, seq_len, 3) inputs.

    Features per stock and day: own return, equal-weighted universe return (market proxy),
    and trailing 20-day volatility. Scaled to roughly unit size and clipped.
    """
    need = seq_len + VOL_WINDOW - 1
    if returns.shape[0] < need:
        raise ValueError(f"Need {need} days of returns, got {returns.shape[0]}.")
    R = np.nan_to_num(returns[-need:], nan=0.0)
    market = R.mean(axis=1, keepdims=True)
    csum = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R**2, axis=0)])
    vol = np.sqrt((csum[VOL_WINDOW:] - csum[:-VOL_WINDOW]) / VOL_WINDOW)
    r = R[VOL_WINDOW - 1 :]
    feats = np.stack(
        [r * 20.0, np.broadcast_to(market[VOL_WINDOW - 1 :] * 20.0, r.shape),
         vol * np.sqrt(TRADING_DAYS) * 2.0],
        axis=-1,
    )
    return np.clip(feats, -5.0, 5.0).transpose(1, 0, 2).astype(np.float32)


class PortfolioNet(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 32, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.temporal_attention = nn.Linear(hidden, 1)
        self.cross_asset = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden, dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, assets, days, features); mask: (batch, assets), True = investable."""
        B, N, L, F = x.shape
        h, _ = self.encoder(x.reshape(B * N, L, F))
        attn = torch.softmax(self.temporal_attention(h).squeeze(-1), dim=1)
        z = (attn.unsqueeze(-1) * h).sum(dim=1).reshape(B, N, -1)
        pad = None if mask is None else ~mask
        z = self.cross_asset(z, src_key_padding_mask=pad)
        scores = self.head(z).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1)


def portfolio_loss(
    weights: torch.Tensor,
    future: torch.Tensor,
    cvar_weight: float,
    cap: float,
    cap_penalty: float,
    alpha: float = 0.05,
) -> torch.Tensor:
    """weights: (B, N); future: (B, H, N) realized returns after each decision date."""
    port = torch.einsum("bhn,bn->bh", future, weights).reshape(-1)
    sharpe = port.mean() / (port.std() + 1e-8) * math.sqrt(TRADING_DAYS)
    k = max(1, int(alpha * port.numel()))
    cvar = -torch.topk(-port, k).values.mean()  # mean of the worst alpha share of days
    over_cap = torch.relu(weights - cap).pow(2).sum(dim=-1).mean()
    return -sharpe - cvar_weight * cvar * 100.0 + cap_penalty * over_cap * 100.0


class DeepPortfolio(Strategy):
    def __init__(
        self,
        name: str,
        seq_len: int = 60,
        horizon: int = 21,
        sample_step: int = 5,
        train_start: str = "2007-01-01",
        hidden: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 40,
        batch_size: int = 16,
        patience: int = 6,
        val_fraction: float = 0.2,
        cvar_weight: float = 0.1,
        cap_penalty: float = 10.0,
        n_models: int = 5,
        max_weight: float = 0.10,
        device: str = "auto",
        seed: int = 0,
    ):
        super().__init__(name)
        self.seq_len, self.horizon, self.sample_step = seq_len, horizon, sample_step
        self.train_start = pd.Timestamp(train_start)
        self.net_kwargs = {"hidden": hidden, "heads": heads, "dropout": dropout}
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.val_fraction = val_fraction
        self.cvar_weight, self.cap_penalty = cvar_weight, cap_penalty
        self.n_models, self.max_weight, self.seed = n_models, max_weight, seed
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.models: list[PortfolioNet] = []
        self.trained_year: int | None = None
        self.training_log: list[dict] = []

    # ------------------------------------------------------------------ data
    def build_dataset(self, ctx: RebalanceContext):
        """Training samples available on ``ctx.date``: inputs end at day p, targets cover
        p+1 .. p+horizon, and every target ends on or before the rebalance date."""
        R = ctx.returns
        arr = R.to_numpy()
        idx = R.index
        need = self.seq_len + VOL_WINDOW - 1
        first = max(int(idx.searchsorted(self.train_start)), need - 1)
        last = len(idx) - 1 - self.horizon
        universes: dict[int, np.ndarray] = {}
        X, M, Y, dates = [], [], [], []
        for p in range(first, last + 1, self.sample_step):
            # Universe re-selected once per calendar year, on the first sample date with enough
            # history (point-in-time, mirroring the backtest's yearly refresh).
            year = idx[p].year
            if year not in universes:
                tickers = ctx.universe_at(idx[p])
                if not tickers:
                    continue
                universes[year] = R.columns.get_indexer(tickers)
            cols = universes[year]
            window = arr[p - need + 1 : p + 1][:, cols]
            X.append(make_features(window, self.seq_len))
            M.append(~np.isnan(window).any(axis=0))
            Y.append(np.nan_to_num(arr[p + 1 : p + 1 + self.horizon][:, cols]))
            dates.append(idx[p])
        if not X:
            return None
        n_max = max(x.shape[0] for x in X)
        pad = lambda a, n: np.concatenate(  # noqa: E731
            [a, np.zeros((n - a.shape[0], *a.shape[1:]), a.dtype)]) if a.shape[0] < n else a
        Xa = np.stack([pad(x, n_max) for x in X])
        Ma = np.stack([np.pad(m, (0, n_max - len(m))) for m in M])
        Ya = np.stack([np.pad(y, ((0, 0), (0, n_max - y.shape[1]))) for y in Y]).astype(np.float32)
        return Xa, Ma, Ya, dates

    # -------------------------------------------------------------- training
    def _loss(self, w, y):
        return portfolio_loss(w, y, self.cvar_weight, self.max_weight, self.cap_penalty)

    def _evaluate(self, model, X, M, Y) -> float:
        model.eval()
        with torch.no_grad():
            W = torch.cat([model(X[i : i + 64], M[i : i + 64]) for i in range(0, len(X), 64)])
            return float(self._loss(W, Y))

    def fit(self, ctx: RebalanceContext) -> None:
        data = self.build_dataset(ctx)
        if data is None:
            raise ValueError(f"{self.name}: no training samples before {ctx.date.date()}.")
        Xa, Ma, Ya, dates = data
        n = len(Xa)
        n_val = max(1, int(n * self.val_fraction))
        embargo = math.ceil(self.horizon / self.sample_step)  # no target overlap with val
        n_train = n - n_val - embargo
        if n_train < self.batch_size:
            raise ValueError(f"{self.name}: only {n_train} training samples.")
        dev = self.device
        X = torch.from_numpy(Xa).to(dev)
        M = torch.from_numpy(Ma).to(dev)
        Y = torch.from_numpy(Ya).to(dev)
        tr, va = slice(0, n_train), slice(n - n_val, n)

        self.models = []
        for m in range(self.n_models):
            t0 = time.perf_counter()
            torch.manual_seed(self.seed * 100_003 + ctx.date.year * 101 + m)
            rng = np.random.default_rng([self.seed, ctx.date.year, m])
            model = PortfolioNet(**self.net_kwargs).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
            best, best_state, bad, epoch = float("inf"), None, 0, 0
            for epoch in range(1, self.epochs + 1):  # noqa: B007
                model.train()
                perm = rng.permutation(n_train)
                for b in range(0, n_train - self.batch_size + 1, self.batch_size):
                    ib = torch.from_numpy(perm[b : b + self.batch_size]).to(dev)
                    loss = self._loss(model(X[ib], M[ib]), Y[ib])
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                val = self._evaluate(model, X[va], M[va], Y[va])
                if val < best - 1e-4:
                    best, bad = val, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if best_state is not None:  # None only if validation loss was never finite
                model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            self.training_log.append({
                "trained_on": str(ctx.date.date()), "model": m, "epochs": epoch,
                "train_samples": n_train, "val_samples": n_val,
                "first_sample": str(dates[0].date()), "last_val_sample": str(dates[-1].date()),
                "best_val_loss": round(best, 4), "train_loss": round(
                    self._evaluate(model, X[tr], M[tr], Y[tr]), 4),
                "seconds": round(time.perf_counter() - t0, 1),
            })
        log.info("%s: trained %d models on data up to %s (%d samples)", self.name,
                 self.n_models, ctx.date.date(), n)

    # ------------------------------------------------------------- interface
    def on_rebalance(self, ctx: RebalanceContext) -> None:
        if not self.models or ctx.date.year != self.trained_year:
            self.fit(ctx)
            self.trained_year = ctx.date.year

    def allocate(self, hist, prev, rng):
        x = torch.from_numpy(make_features(hist.to_numpy(), self.seq_len)[None]).to(self.device)
        with torch.no_grad():
            w = np.mean([m(x).cpu().numpy()[0] for m in self.models], axis=0)
        return pd.Series(cap_weights(w, self.max_weight), index=hist.columns)

    def report(self) -> dict:
        return {"device": str(self.device), "training_log": self.training_log}
