# The engine is imported lazily so that light modules (e.g. ``pfopt.backtest.metrics``, used by
# the dashboard) do not pull in the optimizers and their heavy dependencies.
__all__ = ["BacktestResult", "rebalance_dates", "run_backtest"]


def __getattr__(name: str):
    if name in __all__:
        from pfopt.backtest import engine

        return getattr(engine, name)
    raise AttributeError(f"module 'pfopt.backtest' has no attribute {name!r}")
