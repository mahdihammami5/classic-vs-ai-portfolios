# Methodology

This page explains exactly how every strategy is evaluated, so results can be trusted and
reproduced. The same rules apply to classic methods and to the AI models added later.

## Data

| What | Source | Notes |
|---|---|---|
| Daily prices and volumes | Yahoo Finance via `yfinance` | Split- and dividend-adjusted closes, 2005 onward |
| Historical S&P 500 membership | [fja05680/sp500](https://github.com/fja05680/sp500) (MIT) | Constituent list for every change date since 1996 |
| Risk-free rate | Kenneth R. French data library | Daily 1-month T-bill return, used for Sharpe/Sortino |
| Benchmark | SPY ETF (total return) | What a passive S&P 500 investor actually earned |

Run `pfopt download --config configs/baselines.yaml` to rebuild everything in `data/`.
`data/processed/coverage.csv` reports, per year, how many index members have Yahoo prices.

## Investable universe (point in time)

At the first rebalance of each year the universe is re-selected using only information
available on that date:

1. take the stocks that were **S&P 500 members on that date**;
2. keep those with a complete return history over the lookback window;
3. keep the **100 most liquid** by average daily dollar volume over the last 63 days.

Using historical membership prevents the classic mistake of backtesting today's winners
(e.g. holding Tesla or Nvidia in 2010 because they are in the index *today*).

## Backtest rules

* **Walk-forward**: on each rebalance date a strategy sees only the trailing 252 days of returns.
* **Execution**: trades happen at the close of the rebalance date; new weights earn returns from
  the next day. Tests in `tests/test_engine.py` check this.
* **Rebalancing**: monthly (last trading day of the month).
* **Drift**: between rebalances, weights drift with prices like a real buy-and-hold portfolio.
* **Transaction costs**: 10 bps of traded value, charged at each rebalance on the trades needed
  to go from the drifted weights to the new targets (including the initial purchase).
* **Constraints**: long-only, fully invested, at most 10% in any single stock for the optimizers.

## Periods

| Period | Dates | Role |
|---|---|---|
| Development | 2010–2019 | Where AI models will be trained and tuned |
| Test | 2020 → latest | Out-of-sample: never used for any tuning decision |
| COVID crash | 19 Feb – 23 Mar 2020 | Stress window |
| 2022 bear market | 3 Jan – 12 Oct 2022 | Stress window |

## Metrics

CAGR, annualized volatility, Sharpe and Sortino (on returns in excess of the T-bill rate),
maximum drawdown, Calmar (CAGR / |max drawdown|), daily CVaR at 95% (mean of the worst 5% of
days) and annual one-way turnover.

## Strategies (phase 1)

| Strategy | Idea |
|---|---|
| Equal weight | 1/N in every stock. Famously hard to beat |
| Inverse volatility | Weight ∝ 1 / volatility |
| Minimum variance | Lowest-risk long-only portfolio (Ledoit-Wolf shrunk covariance) |
| Mean-variance | Markowitz utility with shrunk expected returns |
| Risk parity | Every stock contributes the same share of risk |
| GA / GA+WoC | Genetic algorithm maximizing Sharpe; WoC averages the 10 best individuals |

## Known limitations

* **Residual survivorship bias**: Yahoo has no prices for many delisted or acquired companies.
  Those members are skipped, which slightly flatters every strategy (including the benchmark
  comparison). `coverage.csv` quantifies it. CRSP data would remove it.
* **Ticker reuse**: a few tickers were reused by different companies; Yahoo returns the current
  company's history.
* **Costs**: a flat 10 bps ignores market impact and bid-ask differences between stocks.
* **Single path**: one historical path, so results should be read with the confidence intervals
  added in phase 4.
