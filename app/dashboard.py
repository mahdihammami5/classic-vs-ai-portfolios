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
BENCH_COLOR = "black"
BENCH_LINE = {"dash": "dash", "color": BENCH_COLOR, "width": 2.5}
PERIODS = {
    "Test (2020 →)": ("2020-01-01", None),
    "Development (2010–2019)": ("2010-01-01", "2019-12-31"),
    "Full (2010 →)": ("2010-01-01", None),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 bear market": ("2022-01-03", "2022-10-12"),
}
PERIOD_KEY = dict(zip(PERIODS, ["test", "development", "full", "covid_crash", "bear_2022"],
                      strict=True))
REPO = "https://github.com/mahdihammami5/classic-vs-ai-portfolios"

# The 7 methods shown by default; the other variants stay one toggle away.
MAIN = ["EqualWeight", "MinVariance", "MeanVariance", "GA", "GA-Band", "DeepPortfolio-v2",
        "MV-Chronos2-LoRA"]
FAMILY_COLOR = {"Classic": "blue", "Genetic algorithm": "orange", "Smarter rebalancing": "orange",
                "Deep learning": "violet", "Foundation model": "green"}

# Detailed, plain-English cards for the main methods (Methods tab).
DETAILS = {
    "EqualWeight": {
        "icon": "⚖️", "family": "Classic",
        "idea": "Split the money equally across the 100 stocks.",
        "steps": [
            "Take the 100 most liquid S&P 500 stocks of the year.",
            "Put 1% of the money in each one.",
            "At the end of every month, sell a little of what went up and buy what went down "
            "to get back to 1% each.",
        ],
        "facts": {"Looks at": "nothing: no data, no forecast", "Learns": "nothing",
                  "Trades": "monthly, back to equal weights"},
        "strength": "No estimate means no estimation error. DeMiguel et al. (2009) showed it "
                    "beats most optimizers out of sample, which makes it the honest baseline.",
        "weakness": "Ignores risk: a very jumpy stock gets as much money as a calm one.",
        "result": "Test Sharpe 0.63, almost exactly the market (0.64). A good sanity check "
                  "that the backtester does not favor anyone.",
        "math": r"w_i = \frac{1}{N}",
    },
    "MinVariance": {
        "icon": "🛡️", "family": "Classic",
        "idea": "Build the calmest possible portfolio.",
        "steps": [
            "Measure, over the last year, how much each stock moves and how stocks move "
            "together (the covariance matrix).",
            "Clean that estimate with Ledoit-Wolf shrinkage, which pulls extreme values towards "
            "an average to reduce noise.",
            "Solve for the weights with the lowest portfolio volatility (at most 10% per stock).",
            "Repeat every month.",
        ],
        "facts": {"Looks at": "last 252 days of returns", "Solver": "convex optimizer (cvxpy)",
                  "Ignores": "expected returns completely"},
        "strength": "Best crisis protection of the project: it lost less than SPY in all 6 "
                    "market drops of more than 15% since 2010 (average -13% vs -22%).",
        "weakness": "Calm stocks also rise less, so it has the lowest growth in the test period.",
        "result": "Test Sharpe 0.51 vs 0.64 for SPY, with 15% volatility instead of 20%.",
        "math": r"\min_w \; w^\top \Sigma w \quad \text{s.t.} \; \textstyle\sum_i w_i = 1,"
                r"\; 0 \le w_i \le 0.1",
    },
    "MeanVariance": {
        "icon": "📐", "family": "Classic",
        "idea": "Markowitz's method: trade off expected return against risk.",
        "steps": [
            "Estimate each stock's expected return from its average over the past year.",
            "Shrink these estimates halfway towards their average, because past averages are "
            "very noisy.",
            "Estimate risk with the shrunk covariance matrix.",
            "Pick the weights that maximize return minus a risk penalty (at most 10% per stock).",
        ],
        "facts": {"Looks at": "last 252 days of returns", "Solver": "quadratic program (cvxpy)",
                  "Settings": "risk aversion γ = 5, 50% mean shrinkage"},
        "strength": "The foundation of modern portfolio theory (Nobel Prize 1990), with an "
                    "explicit, explainable trade-off.",
        "weakness": "Very sensitive to the return estimates: it chases last year's winners and "
                    "trades about 3.5x the portfolio a year.",
        "result": "Test Sharpe 0.69, slightly above SPY, with 18.6% growth a year.",
        "math": r"\max_w \; \hat\mu^\top w - \frac{\gamma}{2} w^\top \Sigma w, \qquad "
                r"\hat\mu = \tfrac12 \mu + \tfrac12 \bar\mu",
    },
    "GA": {
        "icon": "🧬", "family": "Genetic algorithm",
        "idea": "Evolve portfolios like a population: keep the best, mix them, mutate, repeat.",
        "steps": [
            "Create 100 random portfolios of about 20 stocks each, plus the one already held.",
            "Score each one by its Sharpe ratio over the past year (its fitness).",
            "Pick parents by tournament (the best of 3 random ones), blend them (crossover) and "
            "randomly nudge a few weights (mutation). The 2 best survive unchanged (elitism).",
            "Repeat for 150 generations, then buy the best portfolio. Every month.",
        ],
        "facts": {"Looks at": "last 252 days of returns",
                  "Search": "100 portfolios × 150 generations",
                  "Origin": "the course project this repository grew out of"},
        "strength": "Best Sharpe of the project on the test period (0.82 vs 0.64). Any "
                    "objective can be plugged in, even ones an equation solver cannot handle.",
        "weakness": "Concentrated (NVDA and LLY gave 20% of its return), it trades about 2.7x "
                    "the portfolio a year, and it is not statistically better than SPY "
                    "(p = 0.39).",
        "result": "Test Sharpe 0.82 with 18.9% growth a year, second only to its own "
                  "GA-Band variant.",
        "math": r"\text{fitness}(w) = \frac{\mu^\top w}{\sqrt{w^\top \Sigma w}}",
    },
    "GA-Band": {
        "icon": "🎚️", "family": "Smarter rebalancing",
        "idea": "Same GA, but only trade when the change is big enough to be worth the cost.",
        "steps": [
            "Run the GA every month as usual.",
            "Measure how much of the portfolio would have to be traded to reach the new weights.",
            "If it is less than 20%, do nothing and keep the current holdings.",
            "Otherwise, trade.",
        ],
        "facts": {"Band": "20% of the portfolio",
                  "Chosen how": "10%, 20% and 30% tried on 2010–2019 only",
                  "Trades": "only when the change is large"},
        "strength": "Highest Sharpe of all 20 strategies, with less trading than the plain GA "
                    "(2.5x vs 2.7x a year) and a smaller loss in 2022 (-11.6% vs -12.3%).",
        "weakness": "The gain over the plain GA is small, and it keeps the same concentration.",
        "result": "Test Sharpe 0.83, the best of the project, but still within the margin "
                  "of luck (p = 0.34).",
        "math": r"\text{trade only if } \tfrac12 \textstyle\sum_i |w_i^{new} - w_i^{held}| "
                r"\ge 0.20",
    },
    "DeepPortfolio-v2": {
        "icon": "🧠", "family": "Deep learning",
        "idea": "A neural network reads every stock's recent history and outputs the weights "
                "directly.",
        "steps": [
            "For each stock, read 60 days of its return, the market return, its 20-day "
            "volatility and the VIX, plus a profile (momentum, reversal, 1-year volatility, beta).",
            "An LSTM with attention summarizes each stock's history into a vector.",
            "A Transformer layer lets the 100 stocks compare with each other.",
            "A softmax turns the scores into weights that sum to 100%.",
            "Trained to maximize the next 21 days' Sharpe, with a penalty on the worst days. "
            "5 networks are averaged, and they are retrained every January on past data only.",
        ],
        "facts": {"Framework": "PyTorch", "Size": "hidden 32, 4 attention heads, 5-model ensemble",
                  "Chosen how": "best of 6 variants on 2010–2019 only"},
        "strength": "Learns the portfolio end to end and stays defensive: 16% volatility "
                    "instead of 20% for the market.",
        "weakness": "With price data alone there is little signal to learn, and its gain in "
                    "development did not survive on new data.",
        "result": "Test Sharpe 0.50: safer than the market, but too slow in rallies.",
        "math": r"\mathcal{L} = -\text{Sharpe}_{21d}(w) + 0.1 \cdot \text{CVaR}_{5\%}(w) + "
                r"\text{cap penalty}",
    },
    "MV-Chronos2-LoRA": {
        "icon": "🤖", "family": "Foundation model",
        "idea": "Ask a pre-trained AI forecaster for each stock's next month, then let Markowitz "
                "pick the weights.",
        "steps": [
            "Every January, fine-tune Chronos-2 (Amazon, 120M parameters) on the past prices of "
            "the stocks with LoRA: a small add-on trained in about 2.5 minutes on a GTX 1650.",
            "Every month, give it each stock's last 512 days of prices; it forecasts the price "
            "21 days ahead.",
            "Turn the forecast into an expected return.",
            "Use the same Markowitz optimizer as MeanVariance with this forecast, so only the "
            "forecast differs.",
        ],
        "facts": {"Model": "Chronos-2, 120M parameters", "Fine-tuning": "LoRA, 300 steps, yearly",
                  "Skill measure": "rank information coefficient (IC)"},
        "strength": "Real forecasting skill: the best and only significant ranking skill in "
                    "development (IC t-stat 2.4).",
        "weakness": "Its forecasts change a lot from month to month, so it trades about 8.7x "
                    "the portfolio a year and costs eat the edge. The pre-training data may "
                    "overlap the backtest period.",
        "result": "Test Sharpe 0.55: better forecasts did not turn into a better portfolio.",
        "math": r"\text{IC}_t = \text{rank corr}\big(\hat r_{i,t+21},\; r_{i,t+21}\big)",
    },
}

# Short descriptions of the variants that are hidden by default.
VARIANTS = [
    ("Classic", {
        "InverseVol": "More money in calm stocks: each weight is proportional to 1 / volatility.",
        "RiskParity": "Every stock contributes the same share of the portfolio's total risk.",
    }),
    ("Genetic algorithm", {
        "GA+WoC": "\"Wisdom of Crowds\": averages the 10 best portfolios of the last generation "
                  "instead of the single best one (the original course-project method).",
    }),
    ("Smarter rebalancing", {
        "GA-CostAware": "The GA's score becomes Sharpe - 4 × the share of the portfolio it "
                        "would have to trade.",
        "GA-EventVIX": "Rebalances quarterly, plus when the VIX crosses 30 or holdings drift "
                       "10% from target (checked weekly).",
        "GA-Combined": "Cost-aware score + the event-driven schedule.",
        "GA+WoC-Combined": "The same combination with Wisdom of Crowds.",
    }),
    ("Deep learning", {
        "DeepPortfolio": "The first version: 3 inputs per stock, no stock profile, no VIX.",
        "DeepPortfolio-SharpeOnly": "The first version trained on Sharpe alone, without the "
                                    "worst-days (CVaR) penalty.",
    }),
    ("Foundation model", {
        "MV-HistoricalMean": "Forecast = last year's average (identical to MeanVariance; it "
                             "checks that the forecasting pipeline is correct).",
        "MV-Momentum": "Forecast = recent winners keep winning (12-1 month momentum).",
        "MV-Chronos2-ZeroShot": "Chronos-2 with no fine-tuning at all.",
        "MV-Chronos2-ZeroShot-Cross": "Zero-shot, with the stocks forecast together "
                                      "(cross-learning). The only significant result of the "
                                      "project, and it is negative (p = 0.04).",
    }),
]
RULES = [
    ("🗓️", "Only the past", "Each decision sees data up to that day only."),
    ("🏢", "100 stocks", "The most liquid S&P 500 members at that time."),
    ("💸", "0.1% per trade", "Every buy and sell pays a cost."),
    ("🔒", "Max 10% per stock", "Long-only, always fully invested."),
    ("🧪", "Locked test period", "2020 onward was never used to tune anything."),
]
TERMS = {
    "CAGR": "average yearly growth.",
    "Volatility": "how much returns swing over a year. Lower is calmer.",
    "Sharpe ratio": "return above the risk-free rate divided by volatility: return per unit "
                    "of risk. The main score here.",
    "Max drawdown": "the worst fall from a peak to a bottom.",
    "Turnover / yr": "share of the portfolio bought and sold per year (100% = all of it once).",
    "p-value": "chance of seeing a difference this big if the strategy were no better than "
               "SPY. Below 0.05 is usually called significant.",
}
FAMILY = {
    "baselines": "Classic / GA", "deep": "Deep learning", "deep_v2": "Deep learning",
    "foundation": "Foundation model", "rebalancing": "Smarter rebalancing",
}

st.set_page_config(page_title="Classic vs AI portfolios", page_icon="📈", layout="wide")


@st.cache_data
def load():
    returns = collect_returns(RESULTS, exclude=("analysis", "deep_tuning", "rebalancing_tuning"))
    origin = dict(returns.attrs["experiment"])
    aliases = dict(returns.attrs["aliases"])
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
    return returns, market, origin, turnover, saved, aliases


def weights_for(name: str, exp: str) -> pd.DataFrame | None:
    f = RESULTS / exp / f"weights_{name}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def optional(path: Path, **kw) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, **kw)


returns, market, origin, turnover, saved_turnover, aliases = load()
all_names = [c for c in returns.columns if c != BENCH]
main_names = [n for n in MAIN if n in all_names]
N_TESTED = len(all_names) + len(aliases)  # identical series are shown once

st.title("Classic vs AI portfolios")
st.caption("Walk-forward backtests on a point-in-time S&P 500 universe (100 most liquid "
           "stocks), monthly decisions, 10 bps costs, long-only, 10% cap. The 2020→ test period "
           "was never used for tuning. New here? Start with the **Methods** tab. "
           f"[Source code]({REPO})")

# Controls stay at the top of the page (the sidebar is hidden by default on phones).
with st.container(border=True):
    c_period, c_names, c_opts = st.columns([1.3, 5, 1.2], vertical_alignment="bottom")
    with c_opts:
        show_all = st.toggle(f"All {N_TESTED} strategies", value=False,
                             help="By default only the 7 main methods are listed. Turn this on "
                                  "to see every variant that was tested.")
        log_scale = st.toggle("Log scale", value=True)
    names = all_names if show_all else main_names
    period = c_period.selectbox("Period", list(PERIODS))
    chosen = c_names.multiselect("Strategies (SPY is always shown)", names, default=main_names,
                                 key=f"chosen_{show_all}")
focus = chosen or names  # what the other tabs show

start, end = PERIODS[period]
cols = [*chosen, BENCH]
span = returns.loc[start:end, cols].dropna(axis=1, how="all")
rf = market["rf"].reindex(span.index).fillna(0.0)

tab_perf, tab_methods, tab_sig, tab_crisis, tab_port, tab_reb = st.tabs(
    ["Performance", "Methods", "Is it luck?", "Crises", "Portfolio explorer", "When to trade"])

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
    fig.update_traces(selector={"name": BENCH}, line=BENCH_LINE)
    fig.update_layout(height=430, legend={"orientation": "h", "y": -0.15})
    st.plotly_chart(fig, width="stretch")

    dd = span.fillna(0.0).apply(drawdown_series)
    fig = px.line(dd, labels={"value": "Drawdown", "index": "", "variable": ""})
    fig.update_traces(selector={"name": BENCH}, line=BENCH_LINE)
    fig.update_layout(height=300, yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, width="stretch")

with tab_methods:
    st.markdown(f"**{N_TESTED} strategies in 5 families** were tested with the same "
                "backtester and the same rules. The page focuses on the **7 main methods**; "
                "the other variants are listed at the bottom.")
    st.markdown("##### The rules every method follows")
    for col, (icon, title, text) in zip(st.columns(len(RULES)), RULES, strict=True):
        with col.container(border=True):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)

    # Test-period scorecard, computed from the saved returns.
    test = returns.loc["2020-01-01":]
    test_rf = market["rf"].reindex(test.index).fillna(0.0)
    score = {}
    for n in [*main_names, BENCH]:
        m = summarize(test[n].dropna(), test_rf, turnover.get(n))
        if "turnover" not in m and ("test", n) in saved_turnover.index:
            m["turnover"] = saved_turnover[("test", n)]
        score[n] = m
    spy = score[BENCH]

    st.markdown("##### The 7 main methods at a glance (test period, 2020 →)")
    glance = pd.DataFrame([{
        "Method": f"{DETAILS[n]['icon']} {n}", "Family": DETAILS[n]["family"],
        "Sharpe": score[n]["sharpe"],
        "vs SPY": score[n]["sharpe"] - spy["sharpe"], "Growth / yr": score[n]["cagr"],
        "Max drawdown": score[n]["max_drawdown"], "Turnover / yr": score[n].get("turnover"),
    } for n in main_names]).sort_values("Sharpe", ascending=False)
    st.dataframe(
        glance.style.format({"Sharpe": "{:.2f}", "vs SPY": "{:+.2f}", "Growth / yr": "{:.1%}",
                             "Max drawdown": "{:.1%}", "Turnover / yr": "{:.0%}"}, na_rep="–")
        .map(lambda v: f"color: {'#1a7f37' if v > 0 else '#cf222e'}", subset=["vs SPY"]),
        hide_index=True, width="stretch")
    st.caption(f"SPY over the same period: Sharpe {spy['sharpe']:.2f}, growth "
               f"{spy['cagr']:.1%} a year, max drawdown {spy['max_drawdown']:.1%}.")

    st.markdown("##### Explore a method")
    pick = st.pills("Method", main_names, default="GA" if "GA" in main_names else None,
                    format_func=lambda n: f"{DETAILS[n]['icon']} {n}",
                    label_visibility="collapsed")
    pick = pick or main_names[0]
    d, m = DETAILS[pick], score[pick]
    with st.container(border=True):
        color = FAMILY_COLOR.get(d["family"], "gray")
        st.markdown(f"### {d['icon']} {pick} &nbsp; :{color}-badge[{d['family']}]")
        st.markdown(f"*{d['idea']}*")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sharpe (test)", f"{m['sharpe']:.2f}",
                  f"{m['sharpe'] - spy['sharpe']:+.2f} vs SPY")
        k2.metric("Growth / yr", f"{m['cagr']:.1%}", f"{m['cagr'] - spy['cagr']:+.1%} vs SPY")
        k3.metric("Max drawdown", f"{m['max_drawdown']:.1%}",
                  f"{m['max_drawdown'] - spy['max_drawdown']:+.1%} vs SPY")
        to = m.get("turnover")
        k4.metric("Turnover / yr", "–" if to is None or np.isnan(to) else f"{to:.0%}")

        left, right = st.columns([1.25, 1], gap="large")
        with left:
            st.markdown("**How it works**")
            st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(d["steps"], 1)))
            st.markdown("**Key facts**")
            st.markdown("\n".join(f"- **{k}:** {v}" for k, v in d["facts"].items()))
            with st.expander("The math"):
                st.latex(d["math"])
        with right:
            st.success(f"**Strength.** {d['strength']}", icon="✅")
            st.warning(f"**Weakness.** {d['weakness']}", icon="⚠️")
            st.info(f"**Result.** {d['result']}", icon="📊")

        wealth = (1 + test[[pick, BENCH]].fillna(0.0)).cumprod()
        fig = px.line(wealth, labels={"value": "Growth of $1", "index": "", "variable": ""},
                      title=f"Growth of $1 since 2020: {pick} vs SPY")
        fig.update_traces(selector={"name": BENCH}, line=BENCH_LINE)
        fig.update_layout(height=320, legend={"orientation": "h", "y": -0.2}, xaxis_title=None,
                          margin={"t": 50, "b": 10})
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Other variants tested")
    with st.expander(f"{N_TESTED - len(main_names)} more variants (turn on "
                     f"\"All {N_TESTED} strategies\" at the top to compare them)"):
        for family, items in VARIANTS:
            items = {n: t for n, t in items.items() if n in all_names or n in aliases}
            if items:
                st.markdown(f"**{family}**")
                st.markdown("\n".join(f"- **{n}**: {t}" for n, t in items.items()))
    listed = set(main_names) | {n for _, it in VARIANTS for n in it}
    if missing := [n for n in all_names if n not in listed]:
        st.caption("Not described: " + ", ".join(missing))

    st.markdown("##### Key terms")
    term_cols = st.columns(3)
    for i, (k, v) in enumerate(TERMS.items()):
        term_cols[i % 3].markdown(f"**{k}**: {v}")
    st.markdown(f"Full details: [methodology]({REPO}/blob/main/docs/methodology.md).")

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
        sig = sig.loc[sig.index.isin([*focus, BENCH])]
        s = sig.sort_values("sharpe")
        fig = go.Figure(go.Scatter(
            x=s.sharpe, y=s.index, mode="markers",
            error_x={"type": "data", "symmetric": False, "array": s.ci_high - s.sharpe,
                     "arrayminus": s.sharpe - s.ci_low},
            marker={"color": [BENCH_COLOR if n == BENCH else "#2a6fdb" for n in s.index]}))
        fig.add_vline(x=float(sig.loc[BENCH, "sharpe"]), line_dash="dash", line_color=BENCH_COLOR)
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
        c = crisis.dropna().loc[lambda d: d.index.isin([*focus, BENCH])]
        c = c.loc[c.mean(axis=1).sort_values(ascending=False).index]
        fig = px.imshow(c * 100, text_auto=".1f", color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0, aspect="auto",
                        labels={"color": "Return (%)"})
        fig.update_layout(height=40 * len(c) + 150)
        st.plotly_chart(fig, width="stretch")

with tab_port:
    with_w = [n for n in names if weights_for(n, origin[n]) is not None]
    pick = [n for n in chosen if n in with_w] + ["GA"]
    default = with_w.index(pick[0]) if pick[0] in with_w else 0
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
            strategy = f.stem.removeprefix("rebalances_")
            if strategy in names:  # the 7 main methods, or every variant when toggled on
                frames.append(pd.read_csv(f, parse_dates=["date"]).assign(strategy=strategy))
        if not show_all:
            st.caption(f"Turn on \"All {N_TESTED} strategies\" at the top to compare "
                       "every rebalancing variant (cost-aware, event-driven, VIX trigger).")
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
