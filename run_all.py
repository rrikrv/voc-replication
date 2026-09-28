import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(title, args, cwd, env=None):
    print(f"\n=== {title}", flush=True)
    t0 = time.time()
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    ok = subprocess.run([PY] + args, cwd=cwd, env=full_env).returncode == 0
    print(f"--- {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f} s)", flush=True)
    return ok


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_theory():
    figs = load_module("voc_figs", ROOT / "theory" / "voc_figs.py")
    th = load_module("voc_theory", ROOT / "misspec" / "voc_theory.py")
    worst = 0.0
    for c in (0.3, 0.7, 1.5, 3.0, 10.0):
        for z in (0.01, 0.1, 1.0, 10.0, 50.0):
            a = np.array(figs.analytic(c, z)[:5])
            t = th.misspec_theory(1.0, c, z, 0.2)
            b = np.array([t["E"], t["L"], t["R2"], t["V"], t["SR"]])
            worst = max(worst, float(np.abs(a - b).max()))
    ok = worst < 1e-4
    print(f"theory (Figs. 1-3) vs. misspecified-model theory at q = 1: max abs difference {worst:.1e} -> {'ok' if ok else 'FAILED'}")
    return ok


def check_linear():
    lin = pd.read_csv(ROOT / "linear_model" / "table1_linear.csv")
    ok = True
    print("linear model vs. backtest kitchen sink (z = 10^3), independent implementations:")
    print(f"  {'T':>4} {'Sharpe lin':>11} {'Sharpe bt':>10} {'IR lin':>8} {'IR bt':>7}")
    for T in (12, 60, 120):
        k = pd.read_csv(ROOT / "backtest" / "results" / f"agg_kitchen_T{T}.csv")
        k = k[k.log10z == 3].iloc[0]
        row = lin[(lin["T"] == T) & (lin["shrinkage"] == "z=10^3")].iloc[0]
        print(f"  {T:>4} {row.SR:>11.3f} {k.sharpe:>10.3f} {row.IR_mkt:>8.3f} {k.ir:>7.3f}")
        ok = ok and abs(row.SR - k.sharpe) < 0.03 and abs(row.IR_mkt - k.ir) < 0.05
    print(f"  -> {'ok' if ok else 'FAILED'}")
    return ok


def check_backtest():
    a = pd.read_csv(ROOT / "backtest" / "results" / "agg_T12.csv")
    r = a[(a.P == a.P.max()) & (a.log10z == 3)].iloc[0]
    ok = abs(r.sharpe - 0.47) < 0.05 and abs(r.ir - 0.31) < 0.05
    print(f"RFF backtest T = 12, P = {int(r.P)}, z = 10^3: Sharpe {r.sharpe:.3f} (paper 0.47), IR {r.ir:.3f} (paper 0.31) -> {'ok' if ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--critique", default=None)
    args = ap.parse_args()

    if not (ROOT / "data" / "voc_data.csv").exists():
        sys.exit("data/voc_data.csv is missing")

    bt, lm = ROOT / "backtest", ROOT / "linear_model"
    results = []

    def run(title, script, cwd, *a, env=None):
        results.append((title, step(title, [script, *a], cwd, env)))

    run("backtest: correctness tests", "test_pipeline.py", bt)
    run("theory: Figs. 1-3", "voc_figs.py", ROOT / "theory", "30")
    run("misspec: Figs. 4-6", "voc_misspec_sim.py", ROOT / "misspec")
    run("linear model: Table I, linear rows", "table1.py", lm)
    run("linear model: variable importance and positions", "extra.py", lm)
    for T in (12, 60, 120):
        run(f"backtest: linear kitchen sink, T = {T}", "run.py", bt, "--kitchen-sink", "--T", str(T))

    if args.full:
        run("backtest: main run, T = 12", "run.py", bt)
        run("backtest: T = 60, 10 seeds", "run.py", bt, "--T", "60", "--seeds", "10")
        run("backtest: T = 120, 10 seeds", "run.py", bt, "--T", "120", "--seeds", "10")
        run("backtest: Fig. 9", "plots.py", bt, "--fig9", "results/agg_T60.csv", "results/agg_T120.csv")
        run("linear model: RFF positions, T = 12", "fig10_rff.py", lm, "12", "20")
        run("linear model: RFF positions, T = 60", "fig10_rff.py", lm, "60", "10")
        run("linear model: RFF positions, T = 120", "fig10_rff.py", lm, "120", "6")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            results.append(("backtest: quick smoke test (T = 12, 2 seeds, P <= 1200)",
                            step("backtest: quick smoke test (T = 12, 2 seeds, P <= 1200)",
                                 [str(bt / "run.py"), "--quick"], tmp)))
    run("linear model: Fig. 10 for the RFF model", "fig10_rff.py", lm, "plot")

    if args.critique:
        run("critique checks", "critique_checks.py", ROOT / "critique", args.critique,
            env={"PYTHONPATH": str(ROOT / "data")})

    print("\n=== consistency checks")
    checks = [("theory vs. misspec theory", check_theory), ("linear vs. backtest", check_linear)]
    if args.full:
        checks.append(("RFF backtest vs. paper", check_backtest))
    for title, fn in checks:
        results.append((title, fn()))

    print("\n=== summary")
    for title, ok in results:
        print(f"  {'ok    ' if ok else 'FAILED'}  {title}")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
