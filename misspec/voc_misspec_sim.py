import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from voc_core import ridge_dual
from voc_theory import misspec_theory

B, C, T, NSIM = 0.2, 10, 200, 20
P = C * T
ZS = [0.0, 0.1, 1.0, 10.0, 50.0]
COL = ["k", "orange", "purple", "green", "deepskyblue"]
LAB = ["ridgeless", "z = 0.1", "z = 1", "z = 10", "z = 50"]
QMC = np.array([0.02, 0.05, 0.08, 0.13, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    out = {k: np.zeros((NSIM, QMC.size, len(ZS))) for k in ["E", "L", "V", "R2", "SR"]}
    for s in range(NSIM):
        beta = rng.standard_normal(P) * np.sqrt(B / P)
        S = rng.standard_normal((T, P))
        R = S @ beta + rng.standard_normal(T)
        bb = beta @ beta
        for j, q in enumerate(QMC):
            P1 = int(round(q * P))
            bh = ridge_dual(S[:, :P1], R, np.array(ZS))
            E = bh.T @ beta[:P1]
            L = (bh ** 2).sum(axis=0)
            V = L * (1 + bb) + 2 * E ** 2
            out["E"][s, j], out["L"][s, j], out["V"][s, j] = E, L, V
            out["R2"][s, j], out["SR"][s, j] = (2 * E - L) / (1 + bb), E / np.sqrt(V)
    return {k: v.mean(axis=0) for k, v in out.items()}


def plot(mc, path):
    q = np.linspace(0.002, 1, 600)
    q = q[np.abs(C * q - 1) > 0.01]
    panels = [("R2", "$R^2$", (-0.3, 0.05)), ("L", r"leverage $\mathcal{L}=\|\hat\beta\|^2$", (0, 6)),
              ("E", "expected return", (0, 0.021)), ("V", r"second moment $\mathcal{V}$", (0, 6)),
              ("SR", "Sharpe ratio", (0, 0.06))]
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.6))
    for a, (k, t, yl) in zip(ax, panels):
        for j, z in enumerate(ZS):
            th = misspec_theory(q, C, max(z, 1e-9), B)
            a.plot(th["cq"], th[k], color=COL[j], lw=1.5, label=LAB[j])
            a.plot(C * QMC, mc[k][:, j], "o", color=COL[j], ms=4)
        a.axvline(1, ls="--", color="gray", lw=0.8)
        a.set_title(t)
        a.set_xlabel("cq")
        a.set_xlim(0, 10)
        a.set_ylim(*yl)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Misspecified model, $\\Psi=I$, $b_*={B}$, $c={C}$: theory (lines) vs Monte Carlo (dots, T={T}, {NSIM} sims)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.replace(".png", ".pdf"))
    fig2, ax2 = plt.subplots(2, 1, figsize=(5, 5.2))
    for a, (k, t, yl) in zip(ax2, [panels[2], panels[4]]):
        for j, z in enumerate(ZS):
            th = misspec_theory(q, C, max(z, 1e-9), B)
            a.plot(th["cq"], th[k], color=COL[j], lw=1.5, label=LAB[j])
            a.plot(C * QMC, mc[k][:, j], "o", color=COL[j], ms=3.5)
        a.axvline(1, ls="--", color="gray", lw=0.8)
        a.set_title(t)
        a.set_xlabel("cq")
        a.set_xlim(0, 10)
        a.set_ylim(*yl)
    ax2[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig2.tight_layout()
    fig2.savefig("fig_slide_E_SR.pdf")
    fig2.savefig("fig_slide_E_SR.png", dpi=150)


if __name__ == "__main__":
    mc = simulate()
    np.savez("misspec_mc.npz", q=QMC, **mc)
    plot(mc, "fig4_6_misspec.png")
