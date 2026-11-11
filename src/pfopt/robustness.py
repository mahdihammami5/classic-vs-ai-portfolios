"""Phase 4 cross-experiment analysis: significance of every strategy and crisis behavior.

Reads ``returns.parquet`` from every experiment folder under ``results/`` so the tests cover
*all* strategies tried in the project, including the tuning trials (which count towards the
number of trials in the Deflated Sharpe Ratio even though they never saw the test period).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfopt.data.market import MarketData  # noqa: E402
from pfopt.stats import bootstrap_sharpe, deflated_sharpe_table, drawdown_episodes  # noqa: E402

log = logging.getLogger(__name__)
BENCH = "SPY"


def collect_returns(results_dir: Path, exclude: tuple[str, ...] = ("analysis",)) -> pd.DataFrame:
    """All strategy return series, one column per unique strategy name.

    A name seen in several experiments (e.g. EqualWeight) is kept once, from the first
    experiment in alphabetical order, and so is a series identical to one already kept under
    another name (``attrs['aliases']``). ``attrs['experiment']`` maps names to experiments.
    """
    cols, origin, aliases = {}, {}, {}
    for path in sorted(results_dir.glob("*/returns.parquet")):
        exp = path.parent.name
        if exp in exclude:
            continue
        for name, s in pd.read_parquet(path).items():
            if name in cols:
                continue
            same = [k for k, v in cols.items() if v.index.equals(s.index)
                    and np.allclose(v.to_numpy(), s.to_numpy(), atol=1e-12, equal_nan=True)]
            if same:
                aliases[name] = same[0]
                continue
            cols[name], origin[name] = s, exp
    out = pd.DataFrame(cols)
    out.attrs["experiment"] = origin
    out.attrs["aliases"] = aliases
    return out


def significance(
    returns: pd.DataFrame, data: MarketData, start: str, end: str | None = None,
    n_boot: int = 5000, seed: int = 0,
) -> pd.DataFrame:
    n_trials = returns.shape[1]
    span = returns.loc[start:end].index
    live = returns.loc[span].dropna(axis=1, how="any")  # strategies with a full test record
    live[BENCH] = data.benchmark.reindex(span).fillna(0.0)
    excess = live.sub(data.rf.reindex(span).fillna(0.0), axis=0)
    boot = bootstrap_sharpe(excess, benchmark=BENCH, n_boot=n_boot, seed=seed)
    dsr = deflated_sharpe_table(excess.drop(columns=BENCH), n_trials=n_trials)
    table = boot.join(dsr[["psr", "dsr"]])
    table["experiment"] = [returns.attrs["experiment"].get(c, "benchmark") for c in table.index]
    table = table.sort_values("sharpe", ascending=False)
    table.attrs.update(dsr.attrs)
    table.attrs["days"] = len(span)
    return table


def significance_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Strategy | Experiment | Sharpe | 95% CI | Sharpe − SPY | 95% CI | p-value | PSR | DSR |",
        "|---|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for name, r in table.iterrows():
        diff = ("–", "–", "–") if np.isnan(r.diff_vs_bench) else (
            f"{r.diff_vs_bench:+.2f}", f"[{r.diff_ci_low:+.2f}, {r.diff_ci_high:+.2f}]",
            f"{r.p_value:.2f}")
        psr = "–" if np.isnan(r.get("psr", np.nan)) else f"{r.psr:.0%}"
        dsr = "–" if np.isnan(r.get("dsr", np.nan)) else f"{r.dsr:.0%}"
        lines.append(f"| {name} | {r.experiment} | {r.sharpe:.2f} | "
                     f"[{r.ci_low:.2f}, {r.ci_high:.2f}] | {diff[0]} | {diff[1]} | {diff[2]} | "
                     f"{psr} | {dsr} |")
    return "\n".join(lines)


def crisis_table(returns: pd.DataFrame, data: MarketData, start: str,
                 threshold: float = -0.15) -> pd.DataFrame:
    """Return of every strategy from SPY's peak to its trough, for every SPY drawdown
    deeper than ``threshold`` since ``start``. Windows come from SPY only (no cherry-picking)."""
    bench = data.benchmark.loc[start:]
    episodes = drawdown_episodes(bench, threshold)
    rows = {}
    for _, ep in episodes.iterrows():
        label = f"{ep.peak:%Y-%m-%d} → {ep.trough:%Y-%m-%d}"
        window = (returns.index > ep.peak) & (returns.index <= ep.trough)
        sub = returns.loc[window].dropna(axis=1, how="any")
        col = (1 + sub).prod() - 1
        col[BENCH] = (1 + data.benchmark.loc[sub.index]).prod() - 1
        rows[label] = col
    table = pd.DataFrame(rows)
    table.attrs["episodes"] = episodes
    return table


def crisis_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| Strategy | " + " | ".join(cols) + " | Average |",
             "|---|" + "---:|" * (len(cols) + 1)]
    order = table.mean(axis=1, skipna=False).sort_values(ascending=False).index
    for name in order:
        r = table.loc[name]
        if r.isna().any():
            continue
        cells = " | ".join(f"{v:.1%}" for v in r)
        lines.append(f"| {name} | {cells} | {r.mean():.1%} |")
    return "\n".join(lines)


def plot_sharpe_ci(table: pd.DataFrame, path: Path, title: str) -> None:
    t = table.sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.2))
    y = np.arange(len(t))
    colors = ["black" if n == BENCH else "#2a6fdb" for n in t.index]
    ax.hlines(y, t.ci_low, t.ci_high, color=colors, lw=2, alpha=0.6)
    ax.scatter(t.sharpe, y, color=colors, zorder=3, s=18)
    bench = table.loc[BENCH, "sharpe"]
    ax.axvline(bench, color="black", ls="--", lw=1)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(y, t.index, fontsize=8)
    ax.set_xlabel("Annualized Sharpe ratio (95% stationary-bootstrap CI)")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_analysis(results_dir: Path, data: MarketData, out_dir: Path,
                 test_start: str = "2020-01-01", full_start: str = "2010-01-01",
                 n_boot: int = 5000) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    returns = collect_returns(results_dir)
    # Small market file so the dashboard runs from results/ alone (no data download needed).
    market = pd.DataFrame({BENCH: data.benchmark, "rf": data.rf})
    if data.vix is not None:
        market["VIX"] = data.vix
    market.loc[full_start:].to_parquet(out_dir / "market.parquet")
    sig = significance(returns, data, test_start, n_boot=n_boot)
    sig.to_csv(out_dir / "significance_test.csv")
    plot_sharpe_ci(sig, out_dir / "sharpe_ci_test.png",
                   f"Test period ({test_start[:4]}–): Sharpe with 95% confidence intervals")
    crisis = crisis_table(returns, data, full_start)
    crisis.to_csv(out_dir / "crisis.csv")
    episodes = crisis.attrs["episodes"]

    n_trials, sr0 = sig.attrs["n_trials"], sig.attrs["sr0_annual"]
    md = [
        "# Phase 4: is any of this better than luck?\n",
        f"Test period: {test_start} onward ({sig.attrs['days']} trading days). Sharpe ratios are "
        "on excess returns over the risk-free rate, annualized. Confidence intervals come from "
        "a stationary block bootstrap (5,000 resamples, mean block 21 days); the difference "
        "with SPY uses the same resampled days for both (paired test).\n",
        f"**Deflated Sharpe Ratio.** {n_trials} strategy configurations were tried in this "
        "project (including tuning trials that never saw the test period). With that many "
        "tries, the best skill-less strategy would be expected to reach an annualized Sharpe "
        f"of about **{sr0:.2f}** by luck alone. DSR is the probability that a strategy's true "
        "Sharpe beats that bar; PSR is the probability that it beats 0. A DSR above 95% "
        "would be convincing evidence of skill.\n",
        *([("Identical return series counted once: "
            + ", ".join(f"{a} = {b}" for a, b in returns.attrs["aliases"].items()) + ".\n")]
          if returns.attrs["aliases"] else []),
        significance_markdown(sig),
        "\n![Sharpe confidence intervals](sharpe_ci_test.png)\n",
        "## Crisis behavior\n",
        f"Every SPY drawdown deeper than 15% since {full_start} ({len(episodes)} episodes; "
        "windows chosen by SPY alone, peak to trough). Cells are each strategy's total return "
        "over the same days.\n",
        crisis_markdown(crisis),
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Analysis of %d strategies written to %s", n_trials, out_dir)
    return {"significance": sig, "crisis": crisis}
