from functools import partial

import numpy as np


def rff_features(G, n_features, seed, gamma=2.0):
    if n_features % 2:
        raise ValueError("n_features должно быть чётным (признаки идут парами sin/cos)")
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((G.shape[1], n_features // 2))
    A = gamma * (G @ W)
    F = np.empty((G.shape[0], n_features))
    F[:, 0::2] = np.sin(A)
    F[:, 1::2] = np.cos(A)
    return F


def make_rff(gamma):
    return partial(rff_features, gamma=gamma)


def identity_features(G, n_features, seed):
    return G[:, :n_features]
