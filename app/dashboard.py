"""Interactive results explorer.

    pip install -e ".[dashboard]"
    pfopt analyze            # once, writes results/analysis/
    streamlit run app/dashboard.py

Reads only the files under ``results/``; no market data download is needed. On Streamlit
Community Cloud the light ``app/requirements.txt`` is installed and ``src/`` is put on the path,
so the heavy optimizer dependencies are never needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pfopt.backtest.metrics import drawdown_series, summarize
from pfopt.robustness import collect_returns

RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
BENCH = "SPY"
DEFAULTS = ["EqualWeight", "MeanVariance", "GA", "GA-Combined", "DeepPortfolio",
            "MV-Chronos2-LoRA"]
PERIODS = {
    "Test (2020 →)": ("2020-01-01", None),
    "Development (2010–2019)": ("2010-01-01", "2019-12-31"),
    "Full (2010 →)": ("2010-01-01", None),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 bear market": ("2022-01-03", "2022-10-12"),
}
PERIOD_KEY = dict(zip(PERIODS, ["test", "development", "full", "covid_crash", "bear_2022"],
                      strict=True))
FAMILY = {
    "baselines": "Classic / GA", "deep": "Deep learning", "deep_v2": "Deep learning",
    "foundation": "Foundation model", "rebalancing": "Smarter rebalancing",
}

st.set_page_config(page_title="Classic vs AI portfolios", page_icon="📈", layout="wide")


@st.cache_data
def load():
    returns = collect_returns(RESULTS, exclude=("analysis", "deep_tuning", "rebalancing_tuning"))
    origin = dict(returns.attrs["experiment"])
    market = pd.read_parquet(ANALYSIS / "market.parquet")
    returns = returns.loc[returns.index >= market.index[0]]
    returns[BENCH] = market[BENCH].reindex(returns.index)
    origin[BENCH] = "benchmark"
    turnover = {}
    for exp in set(origin.values()):
        f = RESULTS / exp / "turnover.parquet"
        if f.exists():
            t = pd.read_parquet(f)
            for c in t.columns:
                if origin.get(c) == exp:
                    turnover[c] = t[c].dropna()
    # Saved metrics, used for turnover when an older run has no turnover.parquet.
    saved = pd.concat([pd.read_csv(f).assign(experiment=f.parent.name)
                       for f in RESULTS.glob("*/metrics.csv")])
    keep = [origin.get(n) == e for n, e in zip(saved.strategy, saved.experiment, strict=True)]
    saved = saved[keep]
    saved = saved.set_index(["period", "strategy"])["turnover"]
    return returns, market, origin, turnover, saved


def weights_for(name: str, exp: str) -> pd.DataFrame | None:
    f = RESULTS / exp / f"weights_{name}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def optional(path: Path, **kw) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, **kw)


returns, market, origin, turnover, saved_turnover = load()
names = [c for c in returns.columns if c != BENCH]

st.title("Classic vs AI portfolios")
st.caption("Walk-forward backtests on a point-in-time S&P 500 universe (100 most liquid "
           "stocks), monthly decisions, 10 bps costs, long-only, 10% cap. The 2020→ test period "
           "was never used for tuning.")

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("Period", list(PERIODS))
    chosen = st.multiselect("Strategies", names, default=[n for n in DEFAULTS if n in names])
    log_scale = st.toggle("Log scale", value=True)
    st.markdown("[Source code](https://github.com/mahdihammami5/classic-vs-ai-portfolios)")

start, end = PERIODS[period]
cols = [*chosen, BENCH]
span = returns.loc[start:end, cols].dropna(axis=1, how="all")
rf = market["rf"].reindex(span.index).fillna(0.0)

tab_perf, tab_sig, tab_crisis, tab_port, tab_reb = st.tabs(
    ["Performance", "Is it luck?", "Crises", "Portfolio explorer", "When to trade"])

with tab_perf:
    rows = []
    for c in span.columns:
        r = span[c].dropna()
        m = summarize(r, rf, turnover.get(c))
        if "turnover" not in m and (PERIOD_KEY[period], c) in saved_turnover.index:
            m["turnover"] = saved_turnover[(PERIOD_KEY[period], c)]
        rows.append({"Strategy": c, "Family": FAMILY.get(origin.get(c, ""), "Benchmark"),
                     "CAGR": m["cagr"], "Volatility": m["volatility"], "Sharpe": m["sharpe"],
                     "Max drawdown": m["max_drawdown"], "Turnover / yr": m.get("turnover")})
    table = pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False)
    st.dataframe(
        table.style.format({"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
                            "Max drawdown": "{:.1%}", "Turnover / yr": "{:.0%}"}, na_rep="–"),
        width="stretch")

    wealth = (1 + span.fillna(0.0)).cumprod()
    fig = px.line(wealth, labels={"value": "Growth of $1", "index": "", "variable": ""},
                  log_y=log_scale)
    fig.update_traces(selector={"name": BENCH}, line={"dash": "dash", "color": "black"})
    fig.update_layout(height=430, legend={"orientation": "h", "y": -0.15})
    st.plotly_chart(fig, width="stretch")

    dd = span.fillna(0.0).apply(drawdown_series)
    fig = px.line(dd, labels={"value": "Drawdown", "index": "", "variable": ""})
    fig.update_traces(selector={"name": BENCH}, line={"dash": "dash", "color": "black"})
    fig.update_layout(height=300, yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, width="stretch")

with tab_sig:
    sig = optional(ANALYSIS / "significance_test.csv", index_col=0)
    if sig is None:
        st.info("Run `pfopt analyze` to create the significance tests.")
    else:
        st.markdown(
            "Each dot is a strategy's Sharpe ratio over the **test period**; the bar is its 95% "
            "confidence interval (stationary block bootstrap). Bars that overlap the dashed SPY "
            "line mean the difference with the market could be luck. **DSR** (Deflated Sharpe "
            "Ratio) is the probability of real skill after accounting for every strategy tried "
            "in this project.")
        s = sig.sort_values("sharpe")
        fig = go.Figure(go.Scatter(
            x=s.sharpe, y=s.index, mode="markers",
            error_x={"type": "data", "symmetric": False, "array": s.ci_high - s.sharpe,
                     "arrayminus": s.sharpe - s.ci_low},
            marker={"color": ["black" if n == BENCH else "#2a6fdb" for n in s.index]}))
        fig.add_vline(x=float(sig.loc[BENCH, "sharpe"]), line_dash="dash")
        fig.update_layout(height=28 * len(s) + 120, xaxis_title="Sharpe ratio (test)")
        st.plotly_chart(fig, width="stretch")
        show = sig[["experiment", "sharpe", "ci_low", "ci_high", "diff_vs_bench", "p_value",
                    "psr", "dsr"]]
        st.dataframe(show.style.format(
            {"sharpe": "{:.2f}", "ci_low": "{:.2f}", "ci_high": "{:.2f}",
             "diff_vs_bench": "{:+.2f}", "p_value": "{:.2f}", "psr": "{:.0%}", "dsr": "{:.0%}"},
            na_rep="–"), width="stretch")

with tab_crisis:
    crisis = optional(ANALYSIS / "crisis.csv", index_col=0)
    if crisis is None:
        st.info("Run `pfopt analyze` to create the crisis table.")
    else:
        st.markdown("Total return from SPY's peak to its trough, for every SPY drawdown "
                    "deeper than 15% since 2010. The windows are picked by SPY alone.")
        c = crisis.dropna().loc[lambda d: d.index.isin(cols) | (len(chosen) == 0)]
        c = c.loc[c.mean(axis=1).sort_values(ascending=False).index]
        fig = px.imshow(c * 100, text_auto=".1f", color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0, aspect="auto",
                        labels={"color": "Return (%)"})
        fig.update_layout(height=40 * len(c) + 150)
        st.plotly_chart(fig, width="stretch")

with tab_port:
    with_w = [n for n in names if weights_for(n, origin[n]) is not None]
    default = with_w.index("GA") if "GA" in with_w else 0
    name = st.selectbox("Strategy", with_w, index=default)
    w = weights_for(name, origin[name])
    w = w.loc[:, (w > 0).any()]
    when = st.select_slider("Rebalance date", options=list(w.index.date),
                            value=w.index[-1].date())
    row = w.loc[pd.Timestamp(when)]
    row = row[row > 1e-4].sort_values(ascending=False)
    left, right = st.columns([1, 1.5])
    with left:
        top = row.head(20)
        other = row.iloc[20:].sum()
        bar = pd.concat([top, pd.Series({f"Other ({len(row) - len(top)})": other})])
        fig = px.bar(bar[::-1] * 100, orientation="h",
                     labels={"value": "Weight (%)", "index": ""})
        fig.update_layout(height=560, showlegend=False,
                          title=f"{name} on {when}: {len(row)} holdings")
        st.plotly_chart(fig, width="stretch")
    with right:
        sub = w.loc[start:end] if len(w.loc[start:end]) else w
        k = 10
        topn = sub.mean().sort_values(ascending=False).head(k).index
        area = sub[topn].copy()
        area["Other"] = sub.drop(columns=topn).sum(axis=1)
        fig = px.area(area * 100, labels={"value": "Weight (%)", "index": "", "variable": ""})
        fig.update_traces(selector={"name": "Other"}, line={"color": "#d0d3d8"},
                          fillcolor="rgba(208,211,216,0.6)")
        fig.update_layout(height=560, title=f"Top {k} holdings over time ({period})")
        st.plotly_chart(fig, width="stretch")

    contrib = optional(RESULTS / origin[name] / f"contributions_{name}.parquet")
    if contrib is not None:
        y0 = int(start[:4])
        y1 = int(end[:4]) if end else int(contrib.index.max())
        c = contrib.loc[(contrib.index >= y0) & (contrib.index <= y1)].sum()
        c = c[c.abs() > 0].sort_values()
        best, worst = c.tail(12)[::-1], c.head(8)
        st.subheader("Where did the return come from?")
        st.caption("Sum of each stock's daily contribution (weight × return) before costs, "
                   f"over {period.lower()}.")
        fig = px.bar(pd.concat([best, worst]) * 100, orientation="h",
                     labels={"value": "Contribution (% points)", "index": ""})
        fig.update_layout(height=520, showlegend=False, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, width="stretch")
        share = best.head(5).sum() / c.sum() if c.sum() > 0 else np.nan
        if np.isfinite(share):
            st.write(f"The top 5 stocks produced **{share:.0%}** of the gross return.")

with tab_reb:
    exp_dir = RESULTS / "rebalancing"
    files = sorted(exp_dir.glob("rebalances_*.csv")) if exp_dir.exists() else []
    if not files:
        st.info("Run `pfopt backtest --config configs/rebalancing.yaml` first.")
    else:
        st.markdown(
            "Each mark is a date where the strategy **reconsidered** its portfolio: on the "
            "calendar schedule, when holdings drifted too far, or when the VIX crossed 30. "
            "Hollow marks are proposals that were **not traded** because the change was too "
            "small to be worth the costs (no-trade band).")
        frames = []
        for f in files:
            e = pd.read_csv(f, parse_dates=["date"])
            e["strategy"] = f.stem.removeprefix("rebalances_")
            frames.append(e)
        ev = pd.concat(frames)
        ev = ev[(ev.date >= pd.Timestamp(start)) & (ev.date <= pd.Timestamp(end or "2100"))]
        ev["symbol"] = np.where(ev.traded, "circle", "circle-open")
        fig = px.scatter(ev, x="date", y="strategy", color="reason", symbol="symbol",
                         symbol_map="identity", hover_data=["drift", "proposal_distance"])
        fig.update_layout(height=120 + 45 * ev.strategy.nunique())
        st.plotly_chart(fig, width="stretch")
        vix = market["VIX"].loc[start:end] if "VIX" in market else None
        if vix is not None:
            fig = px.line(vix, labels={"value": "VIX", "index": ""})
            fig.add_hline(y=30, line_dash="dot")
            fig.update_layout(height=220, showlegend=False)
            st.plotly_chart(fig, width="stretch")
        summary = ev.groupby(["strategy", "reason"]).traded.agg(["count", "sum"]).rename(
            columns={"count": "decisions", "sum": "trades"})
        st.dataframe(summary, width="stretch")
