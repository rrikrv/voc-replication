import sys, warnings
import numpy as np
from voc_data import load_voc_data, PREDICTORS
warnings.filterwarnings("ignore")

df = load_voc_data(sys.argv[1] if len(sys.argv) > 1 else None)
G_REAL = df[PREDICTORS].values
R = df["R_next"].values
N, T = len(R), 12
YEARS = np.array([p.year for p in df.index])


def rff_forecast(G, P=12000, z=1e3, seed=0, gamma=2.0):
    rng = np.random.default_rng(seed)
    X = gamma * G @ rng.standard_normal((G.shape[1], P // 2))
    S = np.hstack([np.sin(X), np.cos(X)])
    pred = np.full(N, np.nan)
    for t in range(T, N):
        Str = S[t - T:t]; sd = Str.std(0) + 1e-12; Str = Str / sd
        pred[t] = (S[t] / sd) @ (Str.T @ np.linalg.solve(z * T * np.eye(T) + Str @ Str.T, R[t - T:t]))
    return pred


def avg_forecast(G, seeds=5, **kw):
    return np.nanmean([rff_forecast(G, seed=s, **kw) for s in range(seeds)], axis=0)


def sharpe(pred, mask=None):
    m = ~np.isnan(pred) if mask is None else (~np.isnan(pred) & mask)
    x = pred[m] * R[m]
    return x.mean() / x.std() * np.sqrt(12)


def ar1_noise(seed, rho=0.98):
    rng = np.random.default_rng(seed); e = rng.standard_normal(G_REAL.shape); Z = np.zeros_like(e)
    for t in range(1, N):
        Z[t] = rho * Z[t - 1] + e[t]
    return Z / Z.std(0)


if __name__ == "__main__":
    pv = avg_forecast(G_REAL)
    mom = np.full(N, np.nan)
    for t in range(T, N):
        mom[t] = R[t - T:t].mean()
    m = ~np.isnan(pv)
    print(f"VoC (реальные предикторы) SR = {sharpe(pv):.3f}")
    print(f"12-мес. momentum          SR = {sharpe(mom):.3f},  corr(прогнозы) = {np.corrcoef(pv[m], mom[m])[0,1]:.3f}")

    a, b = pv[m] * R[m], mom[m] * R[m]
    a, b = a / a.std(), b / b.std()
    X = np.c_[np.ones(m.sum()), b]
    coef = np.linalg.lstsq(X, a, rcond=None)[0]; e = a - X @ coef
    print(f"Альфа VoC над momentum: IR = {coef[0]/e.std()*np.sqrt(12):.3f}, t = {coef[0]/np.sqrt(e.var()/m.sum()):.2f}")

    print(f"Подвыборки: 1930–1974 SR = {sharpe(pv, YEARS < 1975):.3f}, 1975–2020 SR = {sharpe(pv, YEARS >= 1975):.3f}")

    for g in [0.5, 1, 2, 4]:
        print(f"gamma = {g}: SR = {sharpe(avg_forecast(G_REAL, seeds=3, gamma=g)):.3f}")

    rng = np.random.default_rng(1)
    print(f"Плацебо: предикторы перемешаны во времени SR = {sharpe(avg_forecast(G_REAL[rng.permutation(N)], seeds=3)):.3f}")
    srs = [sharpe(avg_forecast(ar1_noise(100 + k), seeds=2)) for k in range(6)]
    print(f"Плацебо: персистентный шум AR(1) rho=0.98 SR = {np.round(srs,3)} → среднее {np.mean(srs):.3f}")
    srs = [sharpe(avg_forecast(np.random.default_rng(200 + k).standard_normal(G_REAL.shape), seeds=2)) for k in range(4)]
    print(f"Плацебо: i.i.d. шум SR = {np.round(srs,3)} → среднее {np.mean(srs):.3f}")
