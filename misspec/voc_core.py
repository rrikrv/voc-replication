import numpy as np


def rff(G, P, gamma=2.0, seed=0):
    G = np.asarray(G, dtype=float)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((G.shape[1], P // 2))
    Z = gamma * G @ W
    S = np.empty((G.shape[0], 2 * (P // 2)))
    S[:, 0::2] = np.sin(Z)
    S[:, 1::2] = np.cos(Z)
    return S


def _eig_inv(K, z, T):
    lam, U = np.linalg.eigh(K)
    keep = lam > max(lam.max(), 0.0) * 1e-10
    inv = np.zeros((lam.size, z.size))
    inv[keep] = 1.0 / (z[None, :] * T + lam[keep, None])
    return lam, U, inv


def ridge_dual(S, y, z):
    S = np.asarray(S, dtype=float)
    T = S.shape[0]
    z = np.atleast_1d(np.asarray(z, dtype=float))
    lam, U, inv = _eig_inv(S @ S.T, z, T)
    return S.T @ (U @ ((U.T @ y)[:, None] * inv))


def mp_stieltjes(z, c):
    z = np.asarray(z, dtype=float)
    a = (1 - c) + z
    return (-a + np.sqrt(a * a + 4 * c * z)) / (2 * c * z)


def rolling_timing(S, R, T, z, standardize=True):
    S = np.asarray(S, dtype=float)
    R = np.asarray(R, dtype=float)
    z = np.atleast_1d(np.asarray(z, dtype=float))
    idx = np.arange(T, S.shape[0] - 1)
    fc = np.empty((idx.size, z.size))
    bnorm = np.empty((idx.size, z.size))
    for i, t in enumerate(idx):
        Sw, st = S[t - T:t], S[t]
        if standardize:
            sd = Sw.std(axis=0)
            sd[sd == 0] = 1.0
            Sw, st = Sw / sd, st / sd
        lam, U, inv = _eig_inv(Sw @ Sw.T, z, T)
        D = (U.T @ R[t - T + 1:t + 1])[:, None] * inv
        fc[i] = (U.T @ (Sw @ st)) @ D
        bnorm[i] = (lam[:, None] * D ** 2).sum(axis=0)
    y = R[idx + 1]
    ret = fc * y[:, None]
    r2 = 1 - (y[:, None] - fc).var(axis=0) / y.var()
    return {"t": idx, "forecast": fc, "ret": ret, "beta_norm2": bnorm.mean(axis=0),
            "r2": r2, "er": ret.mean(axis=0), "vol": ret.std(axis=0),
            "sr": ret.mean(axis=0) / ret.std(axis=0) * np.sqrt(12)}
