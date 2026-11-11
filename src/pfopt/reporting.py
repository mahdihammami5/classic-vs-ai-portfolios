"""Turn backtest results into metric tables, figures and files on disk."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pfopt.analysis import information_coefficients, summarize_ic  # noqa: E402
from pfopt.backtest.engine import BacktestResult  # noqa: E402
from pfopt.backtest.metrics import drawdown_series, summarize  # noqa: E402
from pfopt.data.market import MarketData  # noqa: E402

PCT_COLS = ["total_return", "cagr", "volatility", "max_drawdown", "cvar_95", "turnover"]
COLUMN_LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "volatility": "Volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max drawdown",
    "calmar": "Calmar",
    "cvar_95": "CVaR 95% (daily)",
    "turnover": "Turnover / yr",
}


def _slice(s: pd.Series, start, end) -> pd.Series:
    return s.loc[pd.Timestamp(start) if start else None : pd.Timestamp(end) if end else None]


def metrics_table(
    results: list[BacktestResult],
    data: MarketData,
    periods: dict[str, tuple[str | None, str | None]],
    benchmark_name: str = "SPY (benchmark)",
) -> pd.DataFrame:
    rows = []
    for period, (start, end) in periods.items():
        for res in results:
            r = _slice(res.returns, start, end)
            if r.empty:
                continue
            rows.append({"period": period, "strategy": res.name,
                         **summarize(r, data.rf, res.turnover)})
        span = _slice(results[0].returns, start, end).index
        if len(span):
            bench = data.benchmark.reindex(span).fillna(0.0)
            rows.append({"period": period, "strategy": benchmark_name,
                         **summarize(bench, data.rf)})
    return pd.DataFrame(rows).set_index(["period", "strategy"])


def to_markdown(table: pd.DataFrame, period: str) -> str:
    df = table.loc[period].copy()
    lines = ["| Strategy | " + " | ".join(COLUMN_LABELS[c] for c in df.columns) + " |",
             "|---|" + "---:|" * len(df.columns)]
    for name, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                cells.append("–")
            elif c in PCT_COLS:
                cells.append(f"{v:.1%}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_equity(
    results: list[BacktestResult],
    data: MarketData,
    path: Path,
    test_period: tuple[str | None, str | None] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for res in results:
        ax.plot((1 + res.returns).cumprod(), label=res.name, linewidth=1.4)
    bench = data.benchmark.reindex(results[0].returns.index).fillna(0.0)
    ax.plot((1 + bench).cumprod(), label="SPY (benchmark)", color="black", linestyle="--",
            linewidth=1.4)
    if test_period:
        start = pd.Timestamp(test_period[0])
        end = pd.Timestamp(test_period[1]) if test_period[1] else bench.index[-1]
        ax.axvspan(start, end, color="grey", alpha=0.12, label="Out-of-sample test")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (net of transaction costs, log scale)")
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdowns(
    results: list[BacktestResult],
    data: MarketData,
    path: Path,
    period: tuple[str | None, str | None],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for res in results:
        ax.plot(drawdown_series(_slice(res.returns, *period)), label=res.name, linewidth=1.2)
    span = _slice(results[0].returns, *period).index
    bench = data.benchmark.reindex(span).fillna(0.0)
    ax.plot(drawdown_series(bench), label="SPY (benchmark)", color="black", linestyle="--")
    ax.set_title("Drawdown from previous peak (test period)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def forecast_skill(
    results: list[BacktestResult],
    data: MarketData,
    periods: dict[str, tuple[str | None, str | None]],
    out_dir: Path,
    min_dates: int = 12,
) -> pd.DataFrame | None:
    """Rank IC per forecasting strategy and period; saves the raw forecasts too."""
    rows = []
    for res in results:
        preds = res.meta.pop("predictions", None)
        if preds is None or preds.empty:
            continue
        preds.to_parquet(out_dir / f"forecasts_{res.name}.parquet")
        ic = information_coefficients(preds, data.returns, res.meta["horizon"])
        for period, (start, end) in periods.items():
            stats = summarize_ic(_slice(ic, start, end))
            if stats["n_dates"] >= min_dates:
                rows.append({"period": period, "strategy": res.name, **stats})
    return pd.DataFrame(rows).set_index(["period", "strategy"]) if rows else None


def ic_markdown(table: pd.DataFrame) -> str:
    lines = ["| Period | Forecaster | Mean IC | IC t-stat | IC > 0 | Months |",
             "|---|---|---:|---:|---:|---:|"]
    for (period, name), row in table.iterrows():
        lines.append(f"| {period} | {name} | {row.mean_ic:.3f} | {row.t_stat:.2f} | "
                     f"{row.hit_rate:.0%} | {int(row.n_dates)} |")
    return "\n".join(lines)


def save_run(
    out_dir: Path,
    results: list[BacktestResult],
    data: MarketData,
    periods: dict[str, tuple[str | None, str | None]],
    run_info: dict,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = metrics_table(results, data, periods)
    table.to_csv(out_dir / "metrics.csv")
    md = [f"## {p}\n\n{to_markdown(table, p)}\n" for p in table.index.get_level_values(0).unique()]
    ic_table = forecast_skill(results, data, periods, out_dir)
    if ic_table is not None:
        ic_table.to_csv(out_dir / "forecast_ic.csv")
        md.append("## Forecast skill (rank IC vs realized next-period returns)\n\n"
                  + ic_markdown(ic_table) + "\n")
    (out_dir / "metrics.md").write_text("\n".join(md), encoding="utf-8")

    pd.concat([r.returns for r in results], axis=1).to_parquet(out_dir / "returns.parquet")
    for r in results:
        r.weights.to_parquet(out_dir / f"weights_{r.name}.parquet")

    plot_equity(results, data, out_dir / "equity.png", periods.get("test"))
    if "test" in periods:
        plot_drawdowns(results, data, out_dir / "drawdowns_test.png", periods["test"])

    run_info["runtime_s"] = {r.name: round(r.runtime_s, 2) for r in results}
    run_info["strategy_meta"] = {r.name: r.meta for r in results if r.meta}
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str),
                                           encoding="utf-8")
    return table
