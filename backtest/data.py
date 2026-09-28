from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    dates: np.ndarray
    G: np.ndarray
    R: np.ndarray
    names: list

    def __post_init__(self):
        if not (len(self.dates) == self.G.shape[0] == len(self.R)):
            raise ValueError("dates, G и R разной длины")


def load_dataset(path, date_col, ret_col, predictors, *,
                 target_is_next_month=False, add_lag_mkt=True,
                 standardize_returns=False, standardize_predictors=False,
                 predictor_warmup=36, ret_vol_window=12,
                 start_date=None, end_date=None) -> Dataset:
    df = pd.read_csv(path)
    missing = [c for c in [date_col, ret_col, *predictors] if c not in df.columns]
    if missing:
        raise KeyError(f"В файле нет колонок {missing}. Доступны: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if target_is_next_month:
        r_next = df[ret_col].astype(float)
        ret_t = r_next.shift(1)
    else:
        ret_t = df[ret_col].astype(float)
        r_next = ret_t.shift(-1)

    G = df[predictors].astype(float).copy()
    if add_lag_mkt:
        G["lag_mkt"] = ret_t

    if standardize_predictors:
        G = G / G.expanding(min_periods=predictor_warmup).std()

    if standardize_returns:
        vol_t = np.sqrt((ret_t ** 2).rolling(ret_vol_window).mean())
        r_next = r_next / vol_t

    full = pd.concat([df[date_col], G, r_next.rename("R_next")], axis=1).dropna()
    if start_date is not None:
        full = full[full[date_col] >= pd.Timestamp(start_date)]
    if end_date is not None:
        full = full[full[date_col] <= pd.Timestamp(end_date)]
    return Dataset(full[date_col].to_numpy(), full[list(G.columns)].to_numpy(),
                   full["R_next"].to_numpy(), list(G.columns))


def make_synthetic(n=1092, k=15, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.9, 0.99, k)
    G = np.zeros((n, k))
    for t in range(1, n):
        G[t] = rho * G[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(k)
    signal = 0.5 * np.tanh(2 * G[:, 0]) + 0.4 * np.sin(2 * G[:, 1]) * G[:, 2] + 0.3 * np.cos(2 * G[:, 3])
    R = signal + rng.standard_normal(n)
    dates = pd.period_range("1930-01", periods=n, freq="M").to_timestamp().to_numpy()
    return Dataset(dates, G, R / R.std(), [f"g{i}" for i in range(k)])
