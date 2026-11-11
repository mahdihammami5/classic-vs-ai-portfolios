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

## When to trade (phase 4)

Every strategy can carry a rebalancing policy (`rebalance:` in its YAML entry):

| Setting | Meaning |
|---|---|
| `schedule` | calendar rebalancing: `D`, `W`, `M`, `Q` or `Y` (last trading day of the period) |
| `check` | extra dates (`D` or `W`) on which the triggers below are evaluated |
| `drift` | rebalance when holdings have drifted more than this one-way share from the last target |
| `vix_level` | rebalance when the VIX closes on the other side of this level than at the previous check |
| `band` | no-trade band: re-optimize, but skip the trade if the new target is closer than this (one-way) to the holdings |
| `min_gap` | minimum trading days between two trigger-driven rebalances |

Triggers only look at data up to the check date, and trades still execute at that day's close.
Every decision (date, reason, traded or skipped, drift) is saved in `rebalances_<strategy>.csv`.
The GA can also use a **cost-aware fitness**: Sharpe − λ × one-way turnover from current holdings.

## Statistical tests (phase 4)

`pfopt analyze` reads the returns of every experiment and writes `results/analysis/`:

* **Bootstrap confidence intervals.** Stationary block bootstrap (Politis & Romano, 1994; mean
  block 21 days, 5,000 resamples) of the annualized excess-return Sharpe ratio. The Sharpe
  difference with SPY resamples the same days for both series (paired), giving a two-sided
  p-value.
* **Probabilistic and Deflated Sharpe Ratio** (Bailey & López de Prado, 2012, 2014). PSR is the
  probability that the true Sharpe is above 0 given the track-record length, skewness and
  kurtosis. DSR raises the bar to the Sharpe the best of N skill-less strategies would reach by
  chance, where N counts every configuration tried in the project, tuning trials included.
* **Crisis table.** Every SPY drawdown deeper than 15% since 2010, peak to trough. The windows
  are chosen from SPY alone so no strategy can benefit from a hand-picked window.
* **Return attribution.** The backtester records each stock's daily contribution (weight ×
  return, before costs), summed per year (`contributions_<strategy>.parquet`).

## Known limitations

* **Residual survivorship bias**: Yahoo has no prices for many delisted or acquired companies.
  Those members are skipped, which slightly flatters every strategy (including the benchmark
  comparison). `coverage.csv` quantifies it. CRSP data would remove it.
* **Ticker reuse**: a few tickers were reused by different companies; Yahoo returns the current
  company's history.
* **Costs**: a flat 10 bps ignores market impact and bid-ask differences between stocks.
* **Single path**: one historical path, so results should be read with the confidence intervals
  of phase 4. With ~7 years of test data, Sharpe ratio confidence intervals are about ±0.7 wide.
