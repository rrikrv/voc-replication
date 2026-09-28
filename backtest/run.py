import argparse
import time

import numpy as np

import adapters
import config
import plots
from backtest import (PAPER_KITCHEN_SINK, aggregate, compare_with_paper,
                      default_P_grid, run_experiment)
from data import load_dataset, make_synthetic
from engines import DualRidgeEngine
from features import identity_features, make_rff


def build_features(kind):
    return {"rff": make_rff(config.GAMMA),
            "identity": identity_features,
            "core": adapters.core_features}[kind]


def build_engine(kind):
    if kind == "dual":
        return DualRidgeEngine()
    if kind == "core":
        return adapters.CoreEngine()
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=config.T)
    ap.add_argument("--seeds", type=int, default=config.N_SEEDS)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--kitchen-sink", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        data = make_synthetic()
    else:
        data = load_dataset(config.DATA_PATH, config.DATE_COL, config.RET_COL, config.PREDICTORS,
                            target_is_next_month=config.TARGET_IS_NEXT_MONTH,
                            add_lag_mkt=config.ADD_LAG_MKT,
                            standardize_returns=config.STANDARDIZE_RETURNS,
                            standardize_predictors=config.STANDARDIZE_PREDICTORS,
                            predictor_warmup=config.PREDICTOR_WARMUP,
                            start_date=config.START_DATE, end_date=config.END_DATE)
    T = args.T
    print(f"Данные: {len(data.R)} мес. ({str(data.dates[0])[:7]} – {str(data.dates[-1])[:7]}), "
          f"{data.G.shape[1]} предикторов: {data.names}")
    print(f"OOS-прогнозов: {len(data.R) - T}")

    engine = build_engine(config.ENGINE)
    if args.kitchen_sink:
        feature_fn, seeds = identity_features, [0]
        P_grid = np.array([data.G.shape[1]])
        log10_z = [-8, 3]
        tag = f"kitchen_T{T}"
    else:
        feature_fn = build_features(config.FEATURES)
        n_seeds, P_max = (2, 1200) if args.quick else (args.seeds, config.P_MAX)
        seeds = list(range(config.SEED0, config.SEED0 + n_seeds))
        left, right = plots.axis_ranges(T, P_max)
        P_grid = (np.array(config.P_GRID) if config.P_GRID is not None
                  else default_P_grid(T, P_max, left[1], right[0]))
        log10_z = config.LOG10_Z
        tag = f"T{T}" + ("_quick" if args.quick else "") + ("_synth" if args.synthetic else "")

    print(f"P: {len(P_grid)} значений от {P_grid.min()} до {P_grid.max()}; "
          f"log10 z: {log10_z}; seeds: {len(seeds)}")

    t0 = time.time()
    per_seed = run_experiment(data, feature_fn, engine, T, P_grid, log10_z, seeds,
                              n_jobs=args.jobs, standardize_window=config.STANDARDIZE_IN_WINDOW)
    print(f"Готово за {time.time() - t0:.0f} c")

    out = config.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    agg = aggregate(per_seed)
    per_seed.to_csv(out / f"per_seed_{tag}.csv", index=False)
    agg.to_csv(out / f"agg_{tag}.csv", index=False)

    if args.kitchen_sink:
        cols = ["log10z", "r2", "sharpe", "mean_t", "ir", "alpha_t", "max_loss", "skew"]
        print(agg[cols].round(3).to_string(index=False))
        print("Авторы (z = 10^3):", PAPER_KITCHEN_SINK.get(T))
        return

    table = compare_with_paper(agg, T)
    if table is not None:
        print(f"\nP = {P_grid.max()}, z = 10^3, мы vs Table I:\n{table}")
    extra = tag[len(f"T{T}"):]
    print(plots.plot_fig7(agg, T, out, paper_ylim=not args.synthetic, tag=extra))
    print(plots.plot_fig8(agg, T, out, paper_ylim=not args.synthetic, tag=extra))


if __name__ == "__main__":
    main()
