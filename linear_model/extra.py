import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

df, G, R, idx = load()
T, z = 12, 1000.0
full = metrics(*run_rolling(G, R, T, z))
rows = []
for j, name in enumerate(PREDICTORS):
    m = metrics(*run_rolling(G, R, T, z, cols=[k for k in range(15) if k != j]))
    rows.append(dict(predictor=name, VI_R2=full["R2"]-m["R2"], VI_SR=full["SR"]-m["SR"]))
vi = pd.DataFrame(rows).sort_values("VI_R2", ascending=False); vi.to_csv("variable_importance.csv", index=False)
print(vi.round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(vi.predictor, vi.VI_R2, color="#3b5fa0"); ax.axhline(0,color="k",lw=.6)
ax.set_ylabel("VI, out-of-sample R² (pp)", color="#3b5fa0"); ax.tick_params(axis="x", rotation=45)
ax2 = ax.twinx(); ax2.plot(vi.predictor, vi.VI_SR, "o-", color="#c0392b"); ax2.set_ylabel("VI, Sharpe ratio", color="#c0392b")
plt.title("Variable importance, linear model (T=12, z=$10^3$)"); plt.tight_layout(); plt.savefig("fig11_variable_importance.png", dpi=150); plt.close()

NBER = [("1929-08","1933-03"),("1937-05","1938-06"),("1945-02","1945-10"),("1948-11","1949-10"),
 ("1953-07","1954-05"),("1957-08","1958-04"),("1960-04","1961-02"),("1969-12","1970-11"),
 ("1973-11","1975-03"),("1980-01","1980-07"),("1981-07","1982-11"),("1990-07","1991-03"),
 ("2001-03","2001-11"),("2007-12","2009-06"),("2020-02","2020-04")]
plt.figure(figsize=(11,4.5))
for T in (12, 60, 120):
    pi = np.load(f"pos_T{T}_1000.npy")
    s = pd.Series(pi, index=idx[T:].to_timestamp()).rolling(6).mean()
    plt.plot(s.index, s.values, lw=1, label=f"T={T}")
for a,b in NBER: plt.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=.3)
plt.axhline(0,color="k",lw=.5); plt.legend(); plt.ylabel("position $\\pi_t$")
plt.title("Linear model (z=$10^3$): positions vs. NBER recessions (analogue of Fig. 10)")
plt.tight_layout(); plt.savefig("fig10_positions_recessions.png", dpi=150)

pi = pd.Series(np.load("pos_T12_1000.npy"), index=idx[12:].to_timestamp())
res = []
for a,b in NBER:
    a = pd.Timestamp(a); pre = pi[(pi.index>=a-pd.DateOffset(months=6))&(pi.index<a)]
    if len(pre): res.append((a.strftime("%Y-%m"), pre.mean(), pi[pi.index<a-pd.DateOffset(months=6)].mean()))
print("\nmean position in the 6 months before each recession vs. mean position before that:")
for r in res: print(r[0], round(r[1],3), round(r[2],3), "below" if r[1]<r[2] else "above")
print("below usual level in", sum(r[1]<r[2] for r in res), "of", len(res))
