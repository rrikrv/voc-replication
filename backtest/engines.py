import numpy as np


class DualRidgeEngine:

    def window(self, X, y, x, P_grid, z_grid):
        T = X.shape[0]
        zT = np.asarray(z_grid, float)[:, None] * T
        K = np.zeros((T, T))
        k = np.zeros(T)
        pred = np.empty((len(P_grid), len(z_grid)))
        norm2 = np.empty_like(pred)
        prev = 0
        for j, P in enumerate(P_grid):
            B = X[:, prev:P]
            K += B @ B.T
            k += B @ x[prev:P]
            prev = P
            lam, U = np.linalg.eigh(K)
            lam = np.clip(lam, 0.0, None)
            uy, uk = U.T @ y, U.T @ k
            inv = 1.0 / (zT + lam)
            pred[j] = inv @ (uk * uy)
            norm2[j] = (inv ** 2) @ (lam * uy ** 2)
        return pred, norm2


class CallableRidgeEngine:

    def __init__(self, fit_predict, z_convention="paper"):
        if z_convention not in ("paper", "sklearn"):
            raise ValueError("z_convention: 'paper' или 'sklearn'")
        self.fit_predict = fit_predict
        self.z_convention = z_convention

    def window(self, X, y, x, P_grid, z_grid):
        T = X.shape[0]
        pred = np.empty((len(P_grid), len(z_grid)))
        norm2 = np.empty_like(pred)
        for j, P in enumerate(P_grid):
            for l, z in enumerate(z_grid):
                alpha = z if self.z_convention == "paper" else z * T
                p, beta = self.fit_predict(X[:, :P], y, x[:P], alpha)
                pred[j, l] = p
                norm2[j, l] = float(np.dot(beta, beta))
        return pred, norm2


def primal_ridge(X, y, x, z):
    T, P = X.shape
    beta = np.linalg.solve(z * np.eye(P) + X.T @ X / T, X.T @ y / T)
    return float(x @ beta), beta


def compare_engines(engine_a, engine_b, X, y, x, P_grid, z_grid):
    pa, na = engine_a.window(X, y, x, P_grid, z_grid)
    pb, nb = engine_b.window(X, y, x, P_grid, z_grid)
    return float(np.max(np.abs(pa - pb))), float(np.max(np.abs(na - nb) / (1 + np.abs(nb))))
