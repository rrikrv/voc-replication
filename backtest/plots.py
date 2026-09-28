import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

Z_COLORS = {-3: "#0072BD", -2: "#D95319", -1: "#EDB120", 0: "#7E2F8E",
            1: "#77AC30", 2: "#4DBEEE", 3: "#A2142F"}

BROKEN_AXIS = {12: ((0, 50), (990, 1000)), 60: ((0, 10), (195, 200)), 120: ((0, 10), (95, 100))}

FIG7 = [("r2", "Panel A: $R^2$", (-3, 0.1)),
        ("beta_norm2", r"Panel B: mean $\|\hat\beta\|^2$", (0, 3)),
        ("mean", "Panel C: Expected Return", (0, 0.035)),
        ("vol", "Panel D: Volatility", (0, 5))]
FIG8 = [("sharpe", "Panel A: Sharpe Ratio", (0, 0.5)),
        ("alpha", "Panel B: Alpha", (0, 0.025)),
        ("ir", "Panel C: Information Ratio", (0, 0.32)),
        ("alpha_t", "Panel D: Alpha $t$-statistic", (0, 3))]


def axis_ranges(T, P_max):
    c_max = P_max / T
    if T in BROKEN_AXIS and abs(BROKEN_AXIS[T][1][1] - c_max) < 1e-9:
        return BROKEN_AXIS[T]
    return (0, min(50, c_max)), (0.99 * c_max, c_max)


def _panel(fig, spec, agg, metric, title, ylim, ranges, band, legend):
    left, right = ranges
    sub = spec.subgridspec(1, 2, width_ratios=[5, 1], wspace=0.15)
    axL = fig.add_subplot(sub[0])
    axR = fig.add_subplot(sub[1], sharey=axL)

    for lz, g in agg.groupby("log10z"):
        g = g.sort_values("c")
        color = Z_COLORS.get(round(lz))
        for ax in (axL, axR):
            ax.plot(g["c"], g[metric], color=color, lw=1.2, label=f"$\\log_{{10}}(z) = {lz:g}$")
            std = f"{metric}_std"
            if band and std in g and g[std].notna().any():
                ax.fill_between(g["c"], g[metric] - g[std], g[metric] + g[std],
                                color=color, alpha=0.08, lw=0)
    axL.axvline(1, ls="--", color="0.6", lw=0.8, label="$c = 1$")

    axL.set_xlim(left)
    axR.set_xlim(right)
    if ylim is not None:
        lo, hi = ylim
        top = agg.loc[agg["c"] >= 5, metric].max()
        if top > hi:
            hi = 1.05 * top
        axL.set_ylim(lo, hi)
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(left=False, labelleft=False)
    axR.set_xticks(right)

    d = 0.015
    kw = dict(color="k", clip_on=False, lw=0.8)
    for y0 in (0, 1):
        axL.plot((1 - d, 1 + d), (y0 - d, y0 + d), transform=axL.transAxes, **kw)
        axR.plot((-5 * d, 5 * d), (y0 - d, y0 + d), transform=axR.transAxes, **kw)

    axL.set_title(title, fontsize=10)
    axL.set_xlabel("$c$")
    if legend:
        axL.legend(fontsize=7, loc="best")


def plot_panels(agg, panels, T, path, suptitle, band=True, P_max=None, paper_ylim=True):
    ranges = axis_ranges(T, P_max or agg["P"].max())
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.18)
    for n, (metric, title, ylim) in enumerate(panels):
        _panel(fig, gs[n // 2, n % 2], agg, metric, title, ylim if paper_ylim else None,
               ranges, band, legend=(n == 0))
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fig7(agg, T, out_dir, band=True, paper_ylim=True, tag=""):
    n = int(agg["n_seeds"].iloc[0])
    return plot_panels(agg, FIG7, T, Path(out_dir) / f"fig7_T{T}{tag}.png",
                       f"Out-of-sample performance (T = {T}), reproduction, {n} seeds", band,
                       paper_ylim=paper_ylim)


def plot_fig8(agg, T, out_dir, band=True, paper_ylim=True, tag=""):
    n = int(agg["n_seeds"].iloc[0])
    return plot_panels(agg, FIG8, T, Path(out_dir) / f"fig8_T{T}{tag}.png",
                       f"Out-of-sample market timing (T = {T}), reproduction, {n} seeds", band,
                       paper_ylim=paper_ylim)


def plot_fig9(agg60, agg120, out_dir, band=True):
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.18)
    cells = [(agg60, 60, "ir", "Panel A: Information Ratio (T = 60)", (0, 0.25)),
             (agg60, 60, "alpha_t", "Panel B: Alpha $t$-statistic (T = 60)", (0, 2.3)),
             (agg120, 120, "ir", "Panel C: Information Ratio (T = 120)", (0, 0.25)),
             (agg120, 120, "alpha_t", "Panel D: Alpha $t$-statistic (T = 120)", (0, 2.3))]
    for n, (agg, T, metric, title, ylim) in enumerate(cells):
        _panel(fig, gs[n // 2, n % 2], agg, metric, title, ylim,
               axis_ranges(T, agg["P"].max()), band, legend=(n == 0))
    path = Path(out_dir) / "fig9.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="agg_T*.csv")
    ap.add_argument("--fig9", action="store_true")
    ap.add_argument("--no-band", action="store_true")
    ap.add_argument("--auto-ylim", action="store_true", help="не фиксировать оси y по авторам")
    a = ap.parse_args()
    if a.fig9:
        print(plot_fig9(pd.read_csv(a.csv[0]), pd.read_csv(a.csv[1]),
                        Path(a.csv[0]).parent, not a.no_band))
    else:
        for p in a.csv:
            agg = pd.read_csv(p)
            T = int(agg["T"].iloc[0])
            tag = Path(p).stem.replace(f"agg_T{T}", "")
            kw = dict(band=not a.no_band, paper_ylim=not a.auto_ylim, tag=tag)
            print(plot_fig7(agg, T, Path(p).parent, **kw))
            print(plot_fig8(agg, T, Path(p).parent, **kw))
