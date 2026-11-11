import numpy as np
import pandas as pd

from pfopt.stats import (
    bootstrap_sharpe,
    deflated_sharpe_table,
    drawdown_episodes,
    expected_max_sharpe,
    probabilistic_sharpe,
    stationary_bootstrap_indices,
)


def test_bootstrap_indices_are_valid_and_mostly_consecutive():
    idx = stationary_bootstrap_indices(500, 50, block=20, rng=np.random.default_rng(0))
    assert idx.shape == (50, 500) and idx.min() >= 0 and idx.max() < 500
    consecutive = (np.diff(idx, axis=1) % 500 == 1).mean()
    assert 0.9 < consecutive < 0.99  # about 1 - 1/20 of steps continue a block


def test_bootstrap_ci_contains_point_and_detects_a_real_edge():
    rng = np.random.default_rng(1)
    n = 2500
    noise = rng.normal(0, 0.01, n)
    df = pd.DataFrame({
        "bench": noise,
        "skill": noise + 0.002 + rng.normal(0, 0.002, n),  # big, obvious edge
        "same": noise + rng.normal(0, 0.0005, n),
    })
    out = bootstrap_sharpe(df, benchmark="bench", n_boot=500, seed=0)
    assert ((out.ci_low <= out.sharpe) & (out.sharpe <= out.ci_high)).all()
    assert out.loc["skill", "p_value"] < 0.01
    assert out.loc["same", "p_value"] > 0.05
    assert np.isnan(out.loc["bench", "p_value"])


def test_psr_and_dsr_behave():
    assert probabilistic_sharpe(0.05, 2000, 0, 3) > probabilistic_sharpe(0.05, 200, 0, 3)
    assert probabilistic_sharpe(0.0, 1000, 0, 3) == 0.5
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(0.0005, 0.01, (1500, 8)))
    out = deflated_sharpe_table(df, n_trials=50)
    assert (out.dsr <= out.psr).all()


def test_drawdown_episodes_finds_the_crash():
    r = pd.Series(0.0, index=pd.bdate_range("2020-01-01", periods=100))
    r.iloc[10] = -0.30  # crash
    r.iloc[50] = 0.50  # full recovery
    r.iloc[70] = -0.05  # small dip, below threshold
    ep = drawdown_episodes(r, threshold=-0.15)
    assert len(ep) == 1
    assert ep.trough.iloc[0] == r.index[10] and ep.recovery.iloc[0] == r.index[50]
    np.testing.assert_allclose(ep.depth.iloc[0], -0.30)


def test_collect_returns_dedupes_names_and_identical_series(tmp_path):
    from pfopt.robustness import collect_returns

    idx = pd.bdate_range("2020-01-01", periods=5)
    a = pd.Series([0.01, 0.0, -0.01, 0.02, 0.0], index=idx)
    for exp, cols in {"exp1": {"EW": a, "X": a * 2}, "exp2": {"EW": a * 3, "Y": a * 2}}.items():
        (tmp_path / exp).mkdir()
        pd.DataFrame(cols).to_parquet(tmp_path / exp / "returns.parquet")
    out = collect_returns(tmp_path)
    assert list(out.columns) == ["EW", "X"]  # EW kept from exp1; Y identical to X
    assert out.attrs["aliases"] == {"Y": "X"}
    assert out.attrs["experiment"] == {"EW": "exp1", "X": "exp1"}
