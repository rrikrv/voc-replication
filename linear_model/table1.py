import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

G, R = load()[1:3]
rows = []
for T in (12, 60, 120):
    for z, lab in ((0.0, "z=0+"), (1000.0, "z=10^3")):
        pi, y = run_rolling(G, R, T, z)
        rows.append(dict(T=T, shrinkage=lab, **metrics(pi, y)))
        np.save(f"pos_T{T}_{'0' if z==0 else '1000'}.npy", pi)
ours = pd.DataFrame(rows); ours.to_csv("table1_linear.csv", index=False)

paper = pd.DataFrame([
 (12,"z=0+",-9764,-0.11,-1.0,-0.16,-1.6,98.5,-0.9),(12,"z=10^3",-3.8,0.46,4.4,0.33,3.1,2.4,-0.1),
 (60,"z=0+",-96.6,0.00,0.0,-0.07,-0.6,35.8,-11.1),(60,"z=10^3",-0.5,0.44,4.1,0.10,0.9,1.4,-0.3),
 (120,"z=0+",-26.6,0.20,1.8,0.14,1.2,15.4,-6.5),(120,"z=10^3",0.1,0.49,4.4,0.13,1.2,0.8,-0.9)],
 columns=["T","shrinkage","R2_p","SR_p","t_SR_p","IR_p","t_IR_p","maxloss_p","skew_p"])
m = ours.merge(paper, on=["T","shrinkage"])
out = pd.DataFrame({"T":m["T"],"shrinkage":m["shrinkage"],
  "R2 paper":m["R2_p"],"R2 ours":m["R2"].round(1),"SR paper":m["SR_p"],"SR ours":m["SR"].round(2),
  "t(SR) paper":m["t_SR_p"],"t(SR) ours":m["t_SR"].round(1),
  "IR paper":m["IR_p"],"IR ours":m["IR_mkt"].round(2),
  "MaxLoss paper":m["maxloss_p"],"MaxLoss ours":m["max_loss"].round(1),
  "Skew paper":m["skew_p"],"Skew ours":m["skew"].round(1)})
out.to_csv("table1_comparison.csv", index=False); print(out.to_string(index=False))
fig, ax = plt.subplots(figsize=(13,3)); ax.axis("off")
t = ax.table(cellText=out.astype(str).values, colLabels=out.columns, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1,1.5)
plt.title("Table I, 'Linear' rows: paper vs. our reproduction (1929-2020)")
plt.savefig("table1_comparison.png", dpi=170, bbox_inches="tight")
