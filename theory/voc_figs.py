import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = 0.2
ZS = [0, 0.01, 0.1, 1, 10, 50]
COLORS = {0: "k", 0.01: "tab:red", 0.1: "tab:orange", 1: "purple", 10: "tab:green", 50: "tab:cyan"}

def ridge(S, R, z):
    T, P = S.shape
    if P <= T:
        return np.linalg.solve(S.T @ S / T + z * np.eye(P), S.T @ R / T)
    return S.T @ np.linalg.solve(S @ S.T + z * T * np.eye(T), R)

def mc_point(T, c, z, seeds, rng):
    P = max(1, int(round(c * T)))
    out = []
    for _ in range(seeds):
        b = rng.normal(0, np.sqrt(B / P), P)
        S = rng.standard_normal((T, P))
        R = S @ b + rng.standard_normal(T)
        bh = ridge(S, R, z)
        E, L = bh @ b, bh @ bh
        s2 = 1 + b @ b
        V = 2 * E**2 + s2 * L
        out.append([E, L, (2 * E - L) / s2, V, E / np.sqrt(V), np.sqrt(L)])
    return np.mean(out, axis=0)

def m_mp(z, c):
    a = (1 - c) + z
    s = np.sqrt(a * a + 4 * c * z)
    return 2 / (a + s) if a > 0 else (-a + s) / (2 * c * z)

def nu_xi(z, c):
    m = m_mp(z, c)
    xi = (1 - z * m) / (1 / c - 1 + z * m)
    return 1 - z * xi / c

def nu(z, c):
    return 1 - z * m_mp(z, c)

def analytic(c, z):
    z = max(z, 1e-6)
    h = z * 1e-3
    n = nu(z, c)
    n1 = (nu(z + h, c) - nu(z - h, c)) / (2 * h)
    nh = n + z * n1
    E = B * n
    L = B * nh - c * n1
    s2 = 1 + B
    V = 2 * E**2 + s2 * L
    return E, L, (2 * E - L) / s2, V, E / np.sqrt(V), np.sqrt(L)

def run(T=100, seeds=30, cs_mc=None, out="."):
    rng = np.random.default_rng(0)
    if cs_mc is None:
        cs_mc = sorted(set(list(np.round(np.arange(0.1, 1.0, 0.1), 2)) + [0.95, 1.05] +
                           list(np.round(np.arange(1.2, 3.0, 0.4), 2)) + [3.5, 4, 5, 6, 8, 10]))
    cs_an = np.concatenate([np.linspace(0.02, 0.985, 200), np.linspace(1.015, 10, 300)])
    MC = {z: [] for z in ZS}; AN = {z: np.array([analytic(c, z) for c in cs_an]) for z in ZS}
    for z in ZS:
        for c in cs_mc:
            if z == 0 and abs(c - 1) < 0.02:
                MC[z].append([np.nan] * 6); continue
            MC[z].append(mc_point(T, c, z, seeds, rng))
        MC[z] = np.array(MC[z])
    true = dict(E=B, L=B, R2=B / (1 + B), V=3 * B**2 + B, SR=1 / np.sqrt(3 + 1 / B))

    def panel(ax, k, title, ylim, tf=lambda x: x, true_val=None):
        for z in ZS:
            ax.plot(cs_an, tf(AN[z][:, k]), color=COLORS[z], lw=1.2, label=f"z={z}" if z else "ridgeless")
            ax.plot(cs_mc, tf(MC[z][:, k]), ".", color=COLORS[z], ms=5)
        if true_val is not None: ax.axhline(true_val, color="r", ls="--", lw=1, label="True")
        ax.axvline(1, color="gray", ls=":", lw=.8); ax.set_ylim(*ylim); ax.set_title(title); ax.set_xlabel("c")

    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    panel(ax[0], 2, "R²", (-0.3, 0.3), true_val=true["R2"]); ax[0].legend(fontsize=7)
    panel(ax[1], 1, "‖β̂‖² (L)", (0, 6)); f.tight_layout(); f.savefig(f"{out}/fig1.png", dpi=130)

    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    panel(ax[0], 0, "Expected return E", (0, 0.21), true_val=true["E"]); ax[0].legend(fontsize=7)
    panel(ax[1], 3, "Second moment V (для √V поменять k/tf)", (0, 6), true_val=true["V"])
    f.tight_layout(); f.savefig(f"{out}/fig2.png", dpi=130)

    f, ax = plt.subplots(figsize=(6, 4))
    panel(ax, 4, "Sharpe ratio", (0, 0.4), true_val=true["SR"]); ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(f"{out}/fig3.png", dpi=130)
    return cs_mc, MC, cs_an, AN, true

if __name__ == "__main__":
    import sys
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cs_mc, MC, cs_an, AN, true = run(seeds=seeds, out=".")
    i = cs_mc.index(0.5)
    print("c=0.5, z=0: MC E =", MC[0][i, 0].round(3), " (ожидаем b* = 0.2, несмещённость OLS)")
    print("c->0: аналитический SR(z=0, c=0.02) =", round(analytic(0.02, 0)[4], 3), " True =", round(true["SR"], 3))
    print("z=0.01, c=1: ||b||=", round(analytic(1, .01)[5], 2), " ||b||^2=", round(analytic(1, .01)[1], 2), " (в статье пик ~4.7)")
    print("nu через xi vs 1-zm (z=1,c=2):", round(nu_xi(1, 2), 6), round(nu(1, 2), 6))
    print("ridgeless c=2: R2 =", round(analytic(2, 0)[2], 3), "(ожидаем -0.75); c=10:", round(analytic(10, 0)[2], 3), "(-0.076)")
    print("R2(z*=50, c=10) =", round(analytic(10, 50)[2], 4), "(0.003)")
    print("True: V =", round(true["V"], 3), " sqrt(V) =", round(np.sqrt(true["V"]), 3))
