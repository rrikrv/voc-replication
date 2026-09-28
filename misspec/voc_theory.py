import numpy as np


def _g(z, c):
    a = (1 - c) + z
    s = np.sqrt(a * a + 4 * c * z)
    ds = (a + 2 * c) / s
    pos = a >= 0
    w = np.where(pos, 2 * z / (a + s), (s - a) / (2 * c))
    dw = np.where(pos, 2 / (a + s) - 2 * z * (1 + ds) / (a + s) ** 2, (ds - 1) / (2 * c))
    den = 1 / c - 1 + w
    b2 = c - 1 + z
    g_neg = c * (1 - w) * (s + b2) / 2
    dg_neg = c / 2 * (-dw * (s + b2) + (1 - w) * (ds + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        g_pos = z * (1 - w) / den
        dg_pos = ((1 - w) - z * dw) / den - z * (1 - w) * dw / den ** 2
    return np.where(pos, g_pos, g_neg), np.where(pos, dg_pos, dg_neg)


def misspec_theory(q, c, z, b=0.2):
    q = np.asarray(q, dtype=float)
    cq = c * q
    g, dg = _g(z, cq)
    nu = 1 - g / cq
    dnu = -dg / cq
    nuh = nu + z * dnu
    E = b * q * nu
    L = q * (b * nuh - c * (1 + b * (1 - q)) * dnu)
    V = 2 * E ** 2 + (1 + b) * L
    return {"cq": cq, "E": E, "L": L, "V": V,
            "R2": (2 * E - L) / (1 + b), "SR": E / np.sqrt(V)}
