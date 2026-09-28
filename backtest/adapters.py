import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "misspec"))
from voc_core import rff, ridge_dual


def core_features(G, n_features, seed):
    return rff(G, n_features, gamma=2.0, seed=seed)


class CoreEngine:

    def window(self, X, y, x, P_grid, z_grid):
        pred = np.empty((len(P_grid), len(z_grid)))
        norm2 = np.empty_like(pred)
        for j, P in enumerate(P_grid):
            B = ridge_dual(X[:, :P], y, z_grid)
            pred[j] = x[:P] @ B
            norm2[j] = (B ** 2).sum(axis=0)
        return pred, norm2


def core_fit_predict(X, y, x, alpha):
    beta = ridge_dual(X, y, alpha)[:, 0]
    return float(x @ beta), beta


CORE_Z_CONVENTION = "paper"


def sklearn_fit_predict(X, y, x, alpha):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
    return float(x @ m.coef_), m.coef_
