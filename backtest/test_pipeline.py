import numpy as np

from backtest import run_seed, timing_metrics
from data import make_synthetic
from engines import CallableRidgeEngine, DualRidgeEngine, compare_engines, primal_ridge
from features import make_rff


def test_dual_equals_primal():
    rng = np.random.default_rng(1)
    X, y, x = rng.standard_normal((12, 60)), rng.standard_normal(12), rng.standard_normal(60)
    P_grid, z_grid = [4, 10, 12, 14, 30, 60], 10.0 ** np.arange(-3, 4)
    dp, dn = compare_engines(DualRidgeEngine(), CallableRidgeEngine(primal_ridge, "paper"),
                             X, y, x, P_grid, z_grid)
    assert dp < 1e-6 and dn < 1e-6, (dp, dn)


def test_ridgeless_is_min_norm():
    rng = np.random.default_rng(2)
    X, y, x = rng.standard_normal((12, 100)), rng.standard_normal(12), rng.standard_normal(100)
    beta = np.linalg.pinv(X) @ y
    pred, norm2 = DualRidgeEngine().window(X, y, x, [100], [1e-10])
    assert abs(pred[0, 0] - x @ beta) < 1e-5 and abs(norm2[0, 0] - beta @ beta) < 1e-4


def test_sklearn_convention():
    try:
        from adapters import sklearn_fit_predict
        import sklearn
    except ImportError:
        print("  sklearn нет, тест пропущен")
        return
    rng = np.random.default_rng(3)
    X, y, x = rng.standard_normal((12, 40)), rng.standard_normal(12), rng.standard_normal(40)
    dp, _ = compare_engines(DualRidgeEngine(), CallableRidgeEngine(sklearn_fit_predict, "sklearn"),
                            X, y, x, [8, 40], [0.01, 1, 100])
    assert dp < 1e-6, dp


def test_metrics_scale_invariance():
    R = np.array([1.0, -1.0, 2.0, 0.5, -0.3])
    a, b = timing_metrics(R / 3, R), timing_metrics(R, R)
    assert abs(a["sharpe"] - b["sharpe"]) < 1e-12 and a["r2"] != b["r2"]


def test_no_lookahead():
    data = make_synthetic(n=80, k=5)
    fn, eng = make_rff(2.0), DualRidgeEngine()
    p1, *_ = run_seed(0, data, fn, eng, 12, [20], [1.0])
    data.R[50:] = 999.0
    data.G[51:] = 999.0
    p2, *_ = run_seed(0, data, fn, eng, 12, [20], [1.0])
    assert np.allclose(p1[:51 - 12], p2[:51 - 12])


def test_core_matches():
    from adapters import CoreEngine, core_features
    rng = np.random.default_rng(4)
    G = rng.standard_normal((40, 15))
    assert np.array_equal(core_features(G, 200, 3), make_rff(2.0)(G, 200, 3))
    X, y, x = rng.standard_normal((12, 500)), rng.standard_normal(12), rng.standard_normal(500)
    dp, dn = compare_engines(DualRidgeEngine(), CoreEngine(), X, y, x,
                             [2, 12, 14, 100, 500], 10.0 ** np.arange(-3, 4))
    assert dp < 1e-8 and dn < 1e-8, (dp, dn)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"OK  {name}")
