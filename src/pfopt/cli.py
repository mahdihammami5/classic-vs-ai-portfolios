"""Command-line entry point.

    pfopt download --config configs/baselines.yaml
    pfopt backtest --config configs/baselines.yaml [--only EqualWeight,MinVariance]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from pfopt import __version__
from pfopt.backtest.engine import run_backtest
from pfopt.config import load_config
from pfopt.data.market import MarketData, build_dataset
from pfopt.reporting import save_run, to_markdown
from pfopt.strategies import build_strategy

log = logging.getLogger("pfopt")


def cmd_download(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    build_dataset(cfg.data)


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data = MarketData.load(cfg.data.dir)
    specs = cfg.strategies
    if args.only:
        wanted = set(args.only.split(","))
        specs = [s for s in specs if s.name in wanted]
    results = []
    for spec in specs:
        log.info("Running %s", spec.name)
        results.append(run_backtest(build_strategy(spec), data, cfg.backtest, seed=cfg.seed))

    out_dir = Path(cfg.output_dir) / cfg.experiment
    info = {
        "pfopt_version": __version__,
        "config": asdict(cfg),
        "data_first_date": data.returns.index[0],
        "data_last_date": data.returns.index[-1],
    }
    table = save_run(out_dir, results, data, cfg.periods, info)
    for period in table.index.get_level_values(0).unique():
        print(f"\n=== {period} ===\n{to_markdown(table, period)}")
    print(f"\nSaved results to {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pfopt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download prices, membership and risk-free rate")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("backtest", help="run the strategies in a config and save results")
    p.add_argument("--config", required=True)
    p.add_argument("--only", help="comma-separated strategy names to run")
    p.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    args.func(args)


if __name__ == "__main__":
    main()
