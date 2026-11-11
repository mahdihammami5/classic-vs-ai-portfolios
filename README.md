# Classic vs AI Portfolios on the S&P 500

**Do modern AI portfolio models actually beat classic methods once you add realistic
transaction costs and test them fairly?**

This project is a reproducible, walk-forward benchmark that puts classic portfolio
construction, evolutionary optimization and (coming next) deep learning and time-series
foundation models through the **same honest backtester** on 16 years of S&P 500 data,
including the 2020 COVID crash and the 2022 bear market.

![Growth of $1](results/baselines/equity.png)

## Results so far (phase 1: classic and evolutionary baselines)

Out-of-sample test period, **Jan 2020 → Nov 2026**, monthly rebalancing, 10 bps costs,
100 most liquid point-in-time S&P 500 members, nothing tuned on this period:

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Turnover / yr |
|---|---:|---:|---:|---:|---:|
| SPY (benchmark) | 14.7% | 20.0% | 0.64 | -33.7% | – |
| Equal weight | 15.1% | 21.1% | 0.63 | -34.6% | 51% |
| Inverse volatility | 13.7% | 19.2% | 0.61 | -33.7% | 49% |
| Minimum variance | 9.9% | 14.9% | 0.51 | -27.9% | 183% |
| Mean-variance (Markowitz) | 18.6% | 25.1% | 0.69 | -29.3% | 354% |
| Risk parity | 13.2% | 18.3% | 0.61 | -33.0% | 57% |
| Genetic algorithm | 18.9% | 20.1% | **0.82** | -30.5% | 274% |
| Genetic algorithm + Wisdom of Crowds | 18.1% | 20.0% | 0.79 | -30.5% | 284% |

What the numbers say so far:

* **Passive is hard to beat.** Equal weight, inverse volatility and risk parity end up
  close to SPY once costs are included.
* **Minimum variance did its job in crises**: it lost only 12.6% in the 2022 bear market
  vs 24.1% for SPY, but it lagged badly in the 2023–2026 rally.
* **The genetic algorithm has the best Sharpe ratio**, but it trades ~2.7x its portfolio per year,
  and averaging the crowd (WoC) did not improve on the single best individual.
* These gaps are **not yet tested for statistical significance**. That is phase 4, and it matters:
  one historical path can easily make a lucky strategy look skilled.

Full tables for every period (development 2010–2019, COVID crash, 2022 bear market) are in
[`results/baselines/metrics.md`](results/baselines/metrics.md).

## Phase 2: end-to-end deep learning portfolio

A neural network outputs portfolio weights directly and is trained on the portfolio's
risk-adjusted return instead of on price forecasts. The universe changes every year, so the
model is **universe-agnostic and permutation-equivariant**:

```
60 days x 3 features per stock -> shared LSTM + temporal attention -> one embedding per stock
-> Transformer layer across stocks -> score per stock -> softmax -> long-only weights
loss = -Sharpe(next 21 days) + λ·CVaR(5%) + penalty above the 10% cap
```

It is retrained **every January on past data only** (chronological validation split with an
embargo gap, early stopping), and 5 seeds are averaged as a deep ensemble. One full walk-forward
run (2 variants × 18 yearly retrainings × 5 models) takes ~10 minutes on a GTX 1650.

| Test 2020 → Nov 2026 | CAGR | Volatility | Sharpe | Max drawdown | 2022 bear market |
|---|---:|---:|---:|---:|---:|
| SPY (benchmark) | 14.7% | 20.0% | 0.64 | -33.7% | -24.1% |
| Genetic algorithm | 18.9% | 20.1% | 0.82 | -30.5% | -12.3% |
| DeepPortfolio (Sharpe + CVaR) | 10.2% | 15.9% | 0.51 | -29.3% | -13.9% |
| DeepPortfolio (Sharpe only) | 11.7% | 17.0% | 0.57 | -30.4% | -15.8% |

**Honest result: the network did not beat the market.** It learned a diversified, defensive
portfolio (~55 stocks, max ~4% each, 0.88 correlation with equal weight): 20% less volatile than
SPY and much better in 2022, but it gave up too much upside in rallies. The CVaR term does reduce
tail risk, at a cost in return. This matches recent literature (Fernandes & Desell, 2026, report
a Sharpe of 0.29 on 2022–23): with price-only inputs there is little signal to learn, and early
stopping triggers after ~6 epochs. Full tables: [`results/deep/metrics.md`](results/deep/metrics.md).

## How the evaluation stays honest

* **Point-in-time universe**: stocks are chosen from the S&P 500 members *on that date*
  (historical membership), not from today's index. Renamed companies (e.g. FB → META) are mapped.
* **Walk-forward, no look-ahead**: each decision only sees the trailing year of data; trades
  execute at the close and earn returns from the next day. Unit tests enforce this.
* **Realistic frictions**: weights drift between rebalances, and every trade pays
  transaction costs, including the initial purchase.
* **Locked test set**: 2020 onward is never used for tuning any model.
* **Correct risk-free rate**: Sharpe and Sortino use excess returns over 1-month T-bills.

Details and known limitations: [`docs/methodology.md`](docs/methodology.md).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
# optional, for the deep learning models (GPU build: see pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu126

pfopt download --config configs/baselines.yaml   # ~3 min, free public data
pfopt backtest --config configs/baselines.yaml   # ~3 min on a laptop CPU
pytest                                           # 33 tests, synthetic data, no network
```

Results (tables, figures, weights) are written to `results/<experiment>/`. A new strategy is
a class with one method, `allocate(history) -> weights`, plus one line in a YAML config.

## Project structure

```
configs/            experiment definitions (YAML)
src/pfopt/
  data/             downloaders, point-in-time universe, MarketData loader
  backtest/         walk-forward engine and performance metrics
  strategies/       classic optimizers, genetic algorithm, deep learning (+ registry)
  reporting.py      metric tables and figures
  cli.py            `pfopt download` / `pfopt backtest`
tests/              unit tests (engine timing, costs, metrics, constraints)
docs/               methodology and limitations
legacy/             the original course notebook this project grew from
```

## Roadmap

- [x] **Phase 1**: data pipeline, backtester, classic baselines, genetic algorithm + WoC
- [x] **Phase 2**: end-to-end deep learning portfolio (attention-LSTM + cross-asset Transformer
      trained directly on a Sharpe/CVaR objective, following Fernandes & Desell, 2026)
- [ ] **Phase 2b**: richer inputs (momentum, VIX, sectors) and a turnover-aware loss,
      tuned on 2010–2019 only, with every tried configuration logged
- [ ] **Phase 3**: time-series foundation model (Chronos) forecasts → optimizer, zero-shot vs fine-tuned
- [ ] **Phase 4**: statistical significance (bootstrap Sharpe CIs, Deflated Sharpe Ratio),
      crisis analysis, interactive Streamlit dashboard

## Data sources

* Prices: Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance)
* Historical S&P 500 membership: [fja05680/sp500](https://github.com/fja05680/sp500) (MIT)
* Risk-free rate: [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

*Research and education project, not investment advice.*
