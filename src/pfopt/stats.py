"""Is a Sharpe ratio real or luck? Statistical tests used in phase 4.

* Stationary block bootstrap (Politis & Romano, 1994): resamples blocks of consecutive days
  (random length, mean ``block`` days) so volatility clustering is kept. Gives confidence
  intervals for the Sharpe ratio and for the Sharpe difference against a benchmark (the same
  resampled days are used for both series, so the comparison is paired).
* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012): probability that the true
  Sharpe exceeds a threshold, accounting for track-record length, skewness and fat tails.
* Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the PSR against the Sharpe that the
  best of ``n_trials`` strategies with no skill would be expected to reach by chance.
  This corrects for having tried many strategies and reporting the best one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def stationary_bootstrap_indices(
    n: int, n_boot: int, block: float, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n)`` resampled day indices; blocks have geometric length with mean ``block``."""
    t = np.arange(n)
    new = rng.random((n_boot, n)) < 1.0 / block
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    last_start = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    first = np.take_along_axis(starts, last_start, axis=1)
    return (first + t - last_start) % n


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=-1, ddof=1)
    return np.where(sd > 0, x.mean(axis=-1) / np.where(sd > 0, sd, 1.0), np.nan) * np.sqrt(
        TRADING_DAYS)


def bootstrap_sharpe(
    excess: pd.DataFrame,
    benchmark: str | None = None,
    n_boot: int = 5000,
    block: float = 21.0,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Annualized Sharpe, its bootstrap CI and (optionally) the paired difference vs a column."""
    x = excess.dropna().to_numpy().T  # (strategies, days)
    idx = stationary_bootstrap_indices(x.shape[1], n_boot, block, np.random.default_rng(seed))
    boot = np.stack([_sharpe_rows(row[idx]) for row in x])  # (strategies, n_boot)
    point = _sharpe_rows(x)
    q = [alpha / 2, 1 - alpha / 2]
    out = pd.DataFrame({
        "sharpe": point,
        "ci_low": np.nanquantile(boot, q[0], axis=1),
        "ci_high": np.nanquantile(boot, q[1], axis=1),
    }, index=excess.columns)
    if benchmark is not None:
        b = list(excess.columns).index(benchmark)
        diff = boot - boot[b]
        out["diff_vs_bench"] = point - point[b]
        out["diff_ci_low"] = np.nanquantile(diff, q[0], axis=1)
        out["diff_ci_high"] = np.nanquantile(diff, q[1], axis=1)
        # Two-sided p-value: how often the bootstrap difference lands on the other side of 0.
        share_pos = (diff > 0).mean(axis=1)
        out["p_value"] = np.clip(2 * np.minimum(share_pos, 1 - share_pos), 0, 1)
        out.loc[benchmark, ["diff_vs_bench", "diff_ci_low", "diff_ci_high", "p_value"]] = np.nan
    return out


def probabilistic_sharpe(
    sr: float, n_obs: int, skew: float, kurt: float, sr_benchmark: float = 0.0
) -> float:
    """PSR with all Sharpe ratios in per-period (daily) units; ``kurt`` is Pearson kurtosis."""
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2))
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(n_obs - 1) / denom))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Expected best (daily) Sharpe among ``n_trials`` skill-less strategies."""
    if n_trials < 2:
        return 0.0
    z1 = st.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = st.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trials) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_table(
    excess: pd.DataFrame, n_trials: int, var_trials: float | None = None
) -> pd.DataFrame:
    """PSR (vs 0) and DSR for each column of daily excess returns.

    ``var_trials`` is the variance of the daily Sharpe ratios of all the trials; by default it
    is estimated from the columns given.
    """
    rows = {}
    srs = excess.mean() / excess.std(ddof=1)
    var = float(srs.var(ddof=1)) if var_trials is None else var_trials
    sr0 = expected_max_sharpe(n_trials, var)
    for name, r in excess.items():
        r = r.dropna()
        sr = float(r.mean() / r.std(ddof=1))
        sk, ku = float(st.skew(r)), float(st.kurtosis(r, fisher=False))
        rows[name] = {
            "psr": probabilistic_sharpe(sr, len(r), sk, ku, 0.0),
            "dsr": probabilistic_sharpe(sr, len(r), sk, ku, sr0),
            "skew": sk,
            "kurtosis": ku,
        }
    out = pd.DataFrame(rows).T
    out.attrs["sr0_annual"] = sr0 * np.sqrt(TRADING_DAYS)
    out.attrs["n_trials"] = n_trials
    return out


def drawdown_episodes(r: pd.Series, threshold: float = -0.15) -> pd.DataFrame:
    """Peak-to-trough episodes of a return series whose depth is worse than ``threshold``."""
    wealth = (1 + r.fillna(0.0)).cumprod()
    peak_val, peak_date = 1.0, r.index[0]
    trough_val, trough_date = 1.0, r.index[0]
    rows = []
    for date, w in wealth.items():
        if w >= peak_val:
            if trough_val / peak_val - 1 <= threshold:
                rows.append({"peak": peak_date, "trough": trough_date, "recovery": date,
                             "depth": trough_val / peak_val - 1})
            peak_val, peak_date, trough_val, trough_date = w, date, w, date
        elif w < trough_val:
            trough_val, trough_date = w, date
    if trough_val / peak_val - 1 <= threshold:
        rows.append({"peak": peak_date, "trough": trough_date, "recovery": pd.NaT,
                     "depth": trough_val / peak_val - 1})
    return pd.DataFrame(rows, columns=["peak", "trough", "recovery", "depth"])
