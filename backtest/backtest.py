import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

EPS = 1e-8

PAPER_TABLE_I = {
    12: dict(r2=0.006, sharpe=0.47, mean_t=4.5, ir=0.31, alpha_t=2.9, max_loss=1.2, skew=2.5),
    60: dict(r2=0.005, sharpe=0.42, mean_t=3.9, ir=0.25, alpha_t=2.3, max_loss=0.5, skew=1.7),
    120: dict(r2=0.003, sharpe=0.41, mean_t=3.7, ir=0.24, alpha_t=2.2, max_loss=0.3, skew=0.9),
}
PAPER_KITCHEN_SINK = {
    12: dict(r2=-0.038, sharpe=0.46, mean_t=4.4, ir=0.33, alpha_t=3.1, max_loss=2.4, skew=-0.1),
    60: dict(r2=-0.005, sharpe=0.44, mean_t=4.1, ir=0.10, alpha_t=0.9, max_loss=1.4, skew=-0.3),
    120: dict(r2=0.001, sharpe=0.49, mean_t=4.4, ir=0.13, alpha_t=1.2, max_loss=0.8, skew=-0.9),
}


def default_P_grid(T, P_max, left_c_max, right_c_min, n_mid=14, n_right=3):
    step = max(2, 2 * round(T / 12))
    near = np.r_[np.arange(2, 3 * T + 1, step), T - 2, T, T + 2]
    mid = np.geomspace(3 * T, left_c_max * T, n_mid)
    right = np.linspace(right_c_min * T, P_max, n_right)
    grid = 2 * np.round(np.r_[near, mid, right] / 2).astype(int)
    return np.unique(grid[(grid >= 2) & (grid <= P_max)])


def run_seed(seed, data, feature_fn, engine, T, P_grid, z_grid, standardize_window=True):
    P_grid = np.asarray(P_grid, int)
    F = feature_fn(data.G, int(P_grid.max()), seed)
    n = len(data.R)
    if F.shape != (n, P_grid.max()):
        raise ValueError(f"генератор вернул {F.shape}, ожидалось {(n, P_grid.max())}")

    idx = np.arange(T, n)
    pred = np.empty((len(idx), len(P_grid), len(z_grid)))
    norm2 = np.empty_like(pred)
    for s, i in enumerate(idx):
        X, y, x = F[i - T:i], data.R[i - T:i], F[i]
        if standardize_window:
            sd = X.std(axis=0)
            sd[sd < EPS] = 1.0
            X, x = X / sd, x / sd
        pred[s], norm2[s] = engine.window(X, y, x, P_grid, z_grid)
    return pred, norm2, data.R[idx], data.dates[idx]


def timing_metrics(pred, R):
    r = pred * R
    n = len(r)
    m, sd = r.mean(), r.std(ddof=1)
    sse = np.sum((R - pred) ** 2)

    Xr = np.column_stack([np.ones(n), R])
    XtX_inv = np.linalg.inv(Xr.T @ Xr)
    a, b = XtX_inv @ (Xr.T @ r)
    resid = r - a - b * R
    s = np.sqrt(resid @ resid / (n - 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(
            r2=1 - sse / np.sum((R - R.mean()) ** 2),
            r2_zero=1 - sse / np.sum(R ** 2),
            mean=m,
            vol=sd,
            sharpe=m / sd * np.sqrt(12),
            mean_t=m / sd * np.sqrt(n),
            alpha=a,
            alpha_t=a / (s * np.sqrt(XtX_inv[0, 0])),
            ir=a / s * np.sqrt(12),
            beta_mkt=b,
            max_loss=-r.min(),
            skew=np.mean((r - m) ** 3) / r.std() ** 3,
        )


def seed_metrics(seed, data, feature_fn, engine, T, P_grid, log10_z,
                 standardize_window=True):
    t0 = time.time()
    z_grid = 10.0 ** np.asarray(log10_z, float)
    pred, norm2, R, _ = run_seed(seed, data, feature_fn, engine, T, P_grid, z_grid,
                                 standardize_window)
    rows = []
    for j, P in enumerate(P_grid):
        for l, lz in enumerate(log10_z):
            met = timing_metrics(pred[:, j, l], R)
            rows.append(dict(seed=seed, T=T, P=int(P), c=P / T, log10z=lz,
                             beta_norm2=norm2[:, j, l].mean(), **met))
    print(f"  seed {seed}: {time.time() - t0:.1f} c", flush=True)
    return pd.DataFrame(rows)


def run_experiment(data, feature_fn, engine, T, P_grid, log10_z, seeds, *,
                   n_jobs=1, standardize_window=True):
    job = partial(seed_metrics, data=data, feature_fn=feature_fn, engine=engine, T=T,
                  P_grid=P_grid, log10_z=log10_z, standardize_window=standardize_window)
    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            parts = list(ex.map(job, seeds))
    else:
        parts = [job(s) for s in seeds]
    return pd.concat(parts, ignore_index=True)


def aggregate(per_seed):
    keys = ["T", "P", "c", "log10z"]
    metrics = [c for c in per_seed.columns if c not in keys + ["seed"]]
    g = per_seed.groupby(keys)[metrics]
    agg = g.mean().join(g.std().add_suffix("_std")).reset_index()
    agg["n_seeds"] = per_seed["seed"].nunique()
    return agg


def compare_with_paper(agg, T, paper=None, P=None, log10z=3):
    paper = paper if paper is not None else PAPER_TABLE_I.get(T)
    P = P if P is not None else agg["P"].max()
    row = agg[(agg["P"] == P) & (agg["log10z"] == log10z)]
    if row.empty or paper is None:
        return None
    row = row.iloc[0]
    return pd.DataFrame({"ours": {k: row[k] for k in paper}, "paper": paper}).round(3)
