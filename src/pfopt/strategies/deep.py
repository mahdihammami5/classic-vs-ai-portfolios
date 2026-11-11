"""End-to-end deep learning portfolio (phases 2 and 2b).

Instead of forecasting returns and then optimizing, the network outputs portfolio weights
directly and is trained on the quantity investors care about: the risk-adjusted return of
the portfolio it builds (Zhang et al., 2020; Fernandes & Desell, 2026).

Architecture (universe-agnostic, so it works although the stock list changes every year):

    per-stock history (60 days x 3-4 features)
        -> shared LSTM + temporal attention pooling   (one embedding per stock)
        (+ projected per-stock profile: momentum, reversal, volatility, beta)   [phase 2b]
        -> Transformer encoder layer across stocks     (stocks attend to each other)
        -> linear score per stock -> softmax           (long-only weights summing to 1)

Loss on the next ``horizon`` days of realized returns:

    -Sharpe  +  cvar_weight * CVaR_5%  +  cap_penalty * sum(max(w - cap, 0)^2)
             +  turnover_weight * |w_t - w_{t-1 month}|_1                         [phase 2b]

Walk-forward training: the model is retrained every January using only samples whose
target window ends before the rebalance date, with a chronological validation split and
an embargo gap for early stopping. Several models with different seeds are averaged
(a deep ensemble, the neural counterpart of the Wisdom of Crowds).

Every phase-2b option defaults to off, so the phase-2 configuration is reproduced exactly.
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
STATIC_WINDOW = 252
N_STATIC = 4  # 12-1 month momentum, 1-month reversal, 1-year volatility, 1-year beta


def make_features(returns: np.ndarray, seq_len: int, vix: np.ndarray | None = None) -> np.ndarray:
    """Turn a (>= seq_len + VOL_WINDOW - 1, N) block of daily returns into (N, seq_len, F).

    Features per stock and day: own return, equal-weighted universe return (market proxy),
    trailing 20-day volatility and, if given, log(VIX / 20). Scaled to roughly unit size.
    """
    need = seq_len + VOL_WINDOW - 1
    if returns.shape[0] < need:
        raise ValueError(f"Need {need} days of returns, got {returns.shape[0]}.")
    R = np.nan_to_num(returns[-need:], nan=0.0)
    market = R.mean(axis=1, keepdims=True)
    csum = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R**2, axis=0)])
    vol = np.sqrt((csum[VOL_WINDOW:] - csum[:-VOL_WINDOW]) / VOL_WINDOW)
    r = R[VOL_WINDOW - 1 :]
    feats = [r * 20.0, np.broadcast_to(market[VOL_WINDOW - 1 :] * 20.0, r.shape),
             vol * np.sqrt(TRADING_DAYS) * 2.0]
    if vix is not None:
        v = np.log(np.asarray(vix[-seq_len:], dtype=float) / 20.0)[:, None]
        feats.append(np.broadcast_to(v * 2.0, r.shape))
    out = np.clip(np.stack(feats, axis=-1), -5.0, 5.0)
    return out.transpose(1, 0, 2).astype(np.float32)


def make_static(returns: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Per-stock profile from the last year of returns, z-scored across stocks: (N, 4).

    Classic cross-sectional signals: 12-1 month momentum, 1-month reversal,
    1-year volatility and beta to the universe average.
    """
    if returns.shape[0] < STATIC_WINDOW:
        raise ValueError(f"Need {STATIC_WINDOW} days of returns, got {returns.shape[0]}.")
    R = np.clip(np.nan_to_num(returns[-STATIC_WINDOW:], nan=0.0), -0.99, None)
    logs = np.log1p(R)
    momentum = np.expm1(logs[:-21].sum(axis=0))
    reversal = np.expm1(logs[-21:].sum(axis=0))
    vol = R.std(axis=0) * np.sqrt(TRADING_DAYS)
    m = R.mean(axis=1)
    beta = ((R - R.mean(axis=0)) * (m - m.mean())[:, None]).mean(axis=0) / max(m.var(), 1e-12)
    S = np.stack([momentum, reversal, vol, beta], axis=-1)
    ok = np.ones(len(S), bool) if valid is None else valid
    mu, sd = S[ok].mean(axis=0), S[ok].std(axis=0) + 1e-8
    Z = np.where(ok[:, None], (S - mu) / sd, 0.0)
    return np.clip(Z, -3.0, 3.0).astype(np.float32)


class PortfolioNet(nn.Module):
    def __init__(self, n_features: int = 3, hidden: int = 32, heads: int = 4,
                 dropout: float = 0.1, n_static: int = 0):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.temporal_attention = nn.Linear(hidden, 1)
        self.static = nn.Linear(n_static, hidden) if n_static else None
        self.cross_asset = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden, dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                static: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, assets, days, features); mask: (batch, assets), True = investable;
        static: (batch, assets, n_static)."""
        B, N, L, F = x.shape
        h, _ = self.encoder(x.reshape(B * N, L, F))
        attn = torch.softmax(self.temporal_attention(h).squeeze(-1), dim=1)
        z = (attn.unsqueeze(-1) * h).sum(dim=1).reshape(B, N, -1)
        if self.static is not None:
            z = z + self.static(static)
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
    prev_weights: torch.Tensor | None = None,
    turnover_weight: float = 0.0,
) -> torch.Tensor:
    """weights: (B, N); future: (B, H, N) realized returns after each decision date;
    prev_weights: (B, N) weights one rebalance earlier (rows of NaN = no previous decision)."""
    port = torch.einsum("bhn,bn->bh", future, weights).reshape(-1)
    sharpe = port.mean() / (port.std() + 1e-8) * math.sqrt(TRADING_DAYS)
    k = max(1, int(alpha * port.numel()))
    cvar = -torch.topk(-port, k).values.mean()  # mean of the worst alpha share of days
    over_cap = torch.relu(weights - cap).pow(2).sum(dim=-1).mean()
    loss = -sharpe - cvar_weight * cvar * 100.0 + cap_penalty * over_cap * 100.0
    if turnover_weight and prev_weights is not None:
        has_prev = ~torch.isnan(prev_weights).any(dim=-1)
        if has_prev.any():
            traded = (weights[has_prev] - prev_weights[has_prev]).abs().sum(dim=-1)
            loss = loss + turnover_weight * traded.mean()
    return loss


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
        use_vix: bool = False,
        static_features: bool = False,
        turnover_weight: float = 0.0,
        device: str = "auto",
        seed: int = 0,
    ):
        super().__init__(name)
        self.seq_len, self.horizon, self.sample_step = seq_len, horizon, sample_step
        self.train_start = pd.Timestamp(train_start)
        self.use_vix, self.static_features = use_vix, static_features
        self.net_kwargs = {"hidden": hidden, "heads": heads, "dropout": dropout,
                           "n_features": 3 + int(use_vix),
                           "n_static": N_STATIC if static_features else 0}
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.val_fraction = val_fraction
        self.cvar_weight, self.cap_penalty = cvar_weight, cap_penalty
        self.turnover_weight = turnover_weight
        self.n_models, self.max_weight, self.seed = n_models, max_weight, seed
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.models: list[PortfolioNet] = []
        self.trained_year: int | None = None
        self.training_log: list[dict] = []
        self._vix: pd.Series | None = None

    @property
    def window(self) -> int:
        """Days of history one decision needs."""
        need = self.seq_len + VOL_WINDOW - 1
        return max(need, STATIC_WINDOW) if self.static_features else need

    def _inputs(self, block: np.ndarray, vix: np.ndarray | None, valid: np.ndarray | None):
        x = make_features(block, self.seq_len, vix if self.use_vix else None)
        s = make_static(block, valid) if self.static_features else None
        return x, s

    # ------------------------------------------------------------------ data
    def build_dataset(self, ctx: RebalanceContext) -> dict | None:
        """Training samples available on ``ctx.date``: inputs end at day p, targets cover
        p+1 .. p+horizon, and every target ends on or before the rebalance date."""
        if self.use_vix and ctx.vix is None:
            raise ValueError(f"{self.name}: use_vix=True but no VIX data was provided.")
        R = ctx.returns
        arr = R.to_numpy()
        idx = R.index
        vix = None if ctx.vix is None else ctx.vix.reindex(idx).ffill().bfill().to_numpy()
        win = self.window
        first = max(int(idx.searchsorted(self.train_start)), win - 1)
        last = len(idx) - 1 - self.horizon
        lag = max(1, round(self.horizon / self.sample_step))  # samples per rebalance period
        universes: dict[int, np.ndarray] = {}
        X, S, M, Y, P, years, dates = [], [], [], [], [], [], []
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
            block = arr[p - win + 1 : p + 1][:, cols]
            valid = ~np.isnan(block[-(self.seq_len + VOL_WINDOW - 1):]).any(axis=0)
            x, s = self._inputs(block, None if vix is None else vix[p - win + 1 : p + 1], valid)
            k = len(X)
            P.append(k - lag if k >= lag and years[k - lag] == year else -1)
            X.append(x)
            S.append(s)
            M.append(valid)
            Y.append(np.nan_to_num(arr[p + 1 : p + 1 + self.horizon][:, cols]))
            years.append(year)
            dates.append(idx[p])
        if not X:
            return None

        n_max = max(x.shape[0] for x in X)

        def pad(a: np.ndarray, axis: int = 0) -> np.ndarray:
            width = [(0, 0)] * a.ndim
            width[axis] = (0, n_max - a.shape[axis])
            return np.pad(a, width)

        return {
            "X": np.stack([pad(x) for x in X]),
            "S": np.stack([pad(s) for s in S]) if self.static_features else None,
            "M": np.stack([pad(m) for m in M]),
            "Y": np.stack([pad(y, axis=1) for y in Y]).astype(np.float32),
            "P": np.array(P, dtype=np.int64),
            "dates": dates,
        }

    # -------------------------------------------------------------- training
    def _forward(self, model, T: dict, ib: torch.Tensor) -> torch.Tensor:
        static = None if T["S"] is None else T["S"][ib]
        return model(T["X"][ib], T["M"][ib], static)

    def _batch_loss(self, model, T: dict, ib: torch.Tensor) -> torch.Tensor:
        w = self._forward(model, T, ib)
        prev = None
        if self.turnover_weight:
            pidx = T["P"][ib]
            prev = torch.full_like(w, float("nan"))
            has = pidx >= 0
            if has.any():
                with torch.no_grad():
                    prev[has] = self._forward(model, T, pidx[has])
        return portfolio_loss(w, T["Y"][ib], self.cvar_weight, self.max_weight,
                              self.cap_penalty, prev_weights=prev,
                              turnover_weight=self.turnover_weight)

    def _weights(self, model, T: dict, rows: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        return torch.cat([self._forward(model, T, rows[i : i + chunk])
                          for i in range(0, len(rows), chunk)])

    def _evaluate(self, model, T: dict, rows: torch.Tensor) -> float:
        """Loss over many samples at once, computed in chunks to fit in GPU memory."""
        model.eval()
        with torch.no_grad():
            w = self._weights(model, T, rows)
            prev = None
            if self.turnover_weight:
                pidx = T["P"][rows]
                prev = torch.full_like(w, float("nan"))
                has = pidx >= 0
                if has.any():
                    prev[has] = self._weights(model, T, pidx[has])
            return float(portfolio_loss(w, T["Y"][rows], self.cvar_weight, self.max_weight,
                                        self.cap_penalty, prev_weights=prev,
                                        turnover_weight=self.turnover_weight))

    def fit(self, ctx: RebalanceContext) -> None:
        data = self.build_dataset(ctx)
        if data is None:
            raise ValueError(f"{self.name}: no training samples before {ctx.date.date()}.")
        n = len(data["X"])
        n_val = max(1, int(n * self.val_fraction))
        embargo = math.ceil(self.horizon / self.sample_step)  # no target overlap with val
        n_train = n - n_val - embargo
        if n_train < self.batch_size:
            raise ValueError(f"{self.name}: only {n_train} training samples.")
        dev = self.device
        T = {k: None if v is None else torch.from_numpy(v).to(dev)
             for k, v in data.items() if k != "dates"}
        train_rows = torch.arange(n_train, device=dev)
        val_rows = torch.arange(n - n_val, n, device=dev)

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
                    loss = self._batch_loss(model, T, ib)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                val = self._evaluate(model, T, val_rows)
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
                "first_sample": str(data["dates"][0].date()),
                "last_val_sample": str(data["dates"][-1].date()),
                "best_val_loss": round(best, 4),
                "train_loss": round(self._evaluate(model, T, train_rows), 4),
                "seconds": round(time.perf_counter() - t0, 1),
            })
        log.info("%s: trained %d models on data up to %s (%d samples)", self.name,
                 self.n_models, ctx.date.date(), n)

    # ------------------------------------------------------------- interface
    def on_rebalance(self, ctx: RebalanceContext) -> None:
        self._vix = ctx.vix
        if not self.models or ctx.date.year != self.trained_year:
            self.fit(ctx)
            self.trained_year = ctx.date.year

    def allocate(self, hist, prev, rng):
        if len(hist) < self.window:
            raise ValueError(f"{self.name}: needs {self.window} days of history, "
                             f"got {len(hist)}; increase the backtest lookback.")
        block = hist.to_numpy()[-self.window :]
        vix = None
        if self.use_vix:
            vix = self._vix.reindex(hist.index).ffill().bfill().to_numpy()[-self.window :]
        x, s = self._inputs(block, vix, None)
        xt = torch.from_numpy(x[None]).to(self.device)
        st = None if s is None else torch.from_numpy(s[None]).to(self.device)
        with torch.no_grad():
            w = np.mean([m(xt, None, st).cpu().numpy()[0] for m in self.models], axis=0)
        return pd.Series(cap_weights(w, self.max_weight), index=hist.columns)

    def report(self) -> dict:
        return {"device": str(self.device), "training_log": self.training_log}
