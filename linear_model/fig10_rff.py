import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "misspec"))
from voc_core import rff, rolling_timing
from common import load, NBER

P, Z, GAMMA = 12000, 1e3, 2.0

def compute(T, n_seeds):
    df, G, R_next, idx = load()
    R_same = df["R"].values
    R_ext = np.append(R_same, R_next[-1]); G_ext = np.vstack([G, G[-1]])
    fcs, srs = [], []
    for s in range(n_seeds):
        S = rff(G_ext, P, GAMMA, seed=s)
        out = rolling_timing(S, R_ext, T, [Z], standardize=True)
        fcs.append(out["forecast"][:, 0]); srs.append(out["sr"][0])
        print(f"T={T} seed {s}: SR {srs[-1]:.3f}", flush=True)
    fc = np.mean(fcs, axis=0)
    y = R_next[T:]
    assert len(fc) == len(y)
    np.savez(os.path.join(HERE, f"positions_rff_T{T}.npz"), fc=fc, y=y,
             dates=np.array([str(p) for p in idx[T:]]), srs=np.array(srs))

def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 4.5)); pos = {}
    for T in (12, 60, 120):
        f = os.path.join(HERE, f"positions_rff_T{T}.npz")
        if not os.path.exists(f): continue
        d = np.load(f); dts = pd.PeriodIndex(d["dates"], freq="M").to_timestamp()
        s = pd.Series(d["fc"], index=dts); pos[T] = s
        x = d["fc"] * d["y"]; sr_avg = x.mean() / x.std() * np.sqrt(12)
        print(f"T={T}: seeds {len(d['srs'])}, SR of averaged positions {sr_avg:.3f}, mean per-seed SR {d['srs'].mean():.3f}")
        plt.plot(s.rolling(6).mean().index, s.rolling(6).mean().values, lw=1, label=f"T={T}")
    for a, b in NBER: plt.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=.3)
    plt.axhline(0, color="k", lw=.5); plt.legend(); plt.ylabel(r"position $\hat\pi_t$")
    plt.title("RFF ridge model (P=12,000, z=$10^3$): positions vs. NBER recessions (analogue of Fig. 10)")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig10_positions_rff.png"), dpi=150)
    if len(pos) == 3:
        print("correlations of positions: 12-60 %.2f, 12-120 %.2f, 60-120 %.2f (paper: 0.90, 0.87, 0.97)" % tuple(
            np.corrcoef(pos[a].values[-len(pos[b]):] if False else pd.concat([pos[a], pos[b]], axis=1).dropna().values.T)[0, 1]
            for a, b in ((12, 60), (12, 120), (60, 120))))
    if 12 in pos:
        s = pos[12]; n_neg = (s < 0).mean(); print("share of months with negative position, T=12: %.1f%%" % (100 * n_neg),
              "| mean positive %.3f, mean negative %.3f" % (s[s > 0].mean(), s[s < 0].mean()))
        cnt = tot = 0
        for a, b in NBER:
            a = pd.Timestamp(a)
            pre = s[(s.index >= a - pd.DateOffset(months=6)) & (s.index < a)]; before = s[s.index < a - pd.DateOffset(months=6)]
            if len(pre) and len(before):
                tot += 1; cnt += pre.mean() < before.mean(); print(a.strftime("%Y-%m"), round(pre.mean(), 3), round(before.mean(), 3), "below" if pre.mean() < before.mean() else "above")
        print("T=12 RFF: position below its usual level before", cnt, "of", tot, "recessions")

if __name__ == "__main__":
    if sys.argv[1] == "plot": plot()
    else: compute(int(sys.argv[1]), int(sys.argv[2]))
