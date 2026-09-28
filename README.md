# The Virtue of Complexity in Return Prediction: a reproduction

This project reproduces the main results of Kelly, Malamud and Zhou (2024), "The Virtue of Complexity in Return Prediction", *Journal of Finance* 79(1), 459-503 (DOI: 10.1111/jofi.13298).

## The paper in short

Return-prediction models with far more parameters than observations (P >> T) are usually considered hopeless: ordinary least squares fits the training data perfectly and forecasts badly. The paper argues that, once ridge shrinkage is chosen well, market-timing strategies built on such models perform better, not worse, as complexity grows, and that the out-of-sample R² is a poor measure of their economic value. The evidence has three parts:

* **Theory.** With P/T -> c, ridge regression is analysed with random matrix theory. Expected return, leverage, R² and the Sharpe ratio of the timing strategy are given in closed form (Propositions 1-6, Theorem 1).
* **Simulation.** For a correctly specified model complexity hurts (Figs. 1-3); for a misspecified model, where a larger model approximates the true one better, expected return and Sharpe ratio rise with complexity (Figs. 4-6).
* **Empirics.** Random Fourier features turn the 15 Goyal-Welch predictors into up to 12,000 signals; ridge regressions on a rolling window of only 12 months give a market-timing Sharpe ratio of about 0.47 over 1930-2020 (Figs. 7-9, Table I).

## What this repository does

It reimplements the theory, the simulations and the empirical backtest, compares them with the numbers and figures of the paper, and adds a linear benchmark and a few robustness and critique checks. Everything runs from one dataset, `data/voc_data.csv`, and one command, `python run_all.py`.

| Folder | Contents | Paper |
|---|---|---|
| `data/` | `voc_data.py` builds the dataset, `voc_data.csv` is the ready-to-use table (1929-11 to 2020-12) | Sec. V.A |
| `theory/` | Monte Carlo and closed-form curves for the correctly specified model | Figs. 1-3 |
| `misspec/` | RFF generator, dual-form ridge, misspecified-model simulation (`voc_core.py` is the engine shared by the whole project) | Figs. 4-6, eq. 20 |
| `backtest/` | rolling-window out-of-sample backtest over a grid of P and z, with tests | Figs. 7-9, Table I |
| `linear_model/` | linear kitchen-sink benchmark, Table I comparison, variable importance, positions versus recessions | Table I, Figs. 10-11 |
| `critique/` | checks for the momentum critique of the paper | Sec. V.F |
| `slides/` | slide sources, PDFs and speaker notes | |
| `run_all.py` | runs everything and checks that the parts agree with each other | |

## Setup

```bash
git clone <this repository>
cd voc-replication
pip install -r requirements.txt      # Python 3.9+
```

## Running

```bash
python run_all.py                # quick mode: about half a minute
python run_all.py --full         # recomputes all backtests from scratch: roughly 30 minutes on one core
python run_all.py --critique path/to/PredictorData.xlsx     # also runs the critique checks (5-10 minutes)
```

Quick mode runs the tests, the theory and misspecification simulations, the linear benchmark, the linear kitchen-sink backtest for T = 12, 60, 120 and a small smoke test of the RFF backtest, and re-draws Fig. 10 from the stored positions. It then runs the consistency checks:

* the closed-form curves in `theory/` agree with the independent implementation in `misspec/` (difference below 1e-4);
* the linear benchmark in `linear_model/` and the kitchen-sink backtest in `backtest/`, two independent implementations, give the same Sharpe ratios and information ratios;
* with `--full`: the RFF backtest at T = 12, P = 12,000, z = 10^3 lies close to the paper's Table I.

Full mode uses 20 random-feature draws for T = 12, 10 for T = 60 and 10 for T = 120, and 20 / 10 / 6 draws for the position plot in Fig. 10. The stored results in the repository were produced this way, so they can be inspected without running anything (`backtest/results/`, `linear_model/`, `theory/`, `misspec/`).

Each folder can also be run on its own; see the READMEs in `theory/` and `backtest/`.

### Which script makes which result

| Paper | Script |
|---|---|
| Figs. 1-3 | `theory/voc_figs.py` |
| Figs. 4-6 | `misspec/voc_misspec_sim.py` |
| Figs. 7-9 | `backtest/run.py` (`--T 60`, `--T 120`, `plots.py --fig9`) |
| Table I, "Linear" rows | `linear_model/table1.py` and `backtest/run.py --kitchen-sink` |
| Fig. 10 (positions and recessions) | `linear_model/fig10_rff.py` (RFF model), `linear_model/extra.py` (linear model) |
| Fig. 11 (variable importance) | `linear_model/extra.py`, linear model only |

## Data

`data/voc_data.csv`: row `t` holds the 15 standardized Goyal-Welch predictors known at the end of month `t` and `R_next`, the standardized CRSP value-weighted excess return of month `t+1`. Standardization uses only past information: returns are divided by the trailing 12-month uncentered standard deviation, predictors by an expanding standard deviation with at least 36 observations. The estimation sample is 1930-01 to 2020-12 (1,092 months), as in the paper.

The CSV is committed, so nothing has to be downloaded. To rebuild it from the Goyal-Welch spreadsheet: `python data/voc_data.py path/to/PredictorData.xlsx`.

## Main results

**Linear kitchen sink (Table I, "Linear" rows).** With P = 15 predictors and a 12-month window the model sits just past the interpolation boundary (c = P/T = 1.25). Ridgeless regression is unstable; ridge shrinkage z = 10^3 rescues it, although R² stays around zero:

| T | shrinkage | R² paper / ours | Sharpe paper / ours | IR vs. market paper / ours | Max loss paper / ours |
|---|---|---|---|---|---|
| 12 | 0+ | -9764% / -11569% | -0.11 / -0.11 | -0.16 / -0.15 | 98.5 / 99.5 |
| 12 | 10^3 | -3.8% / -4.0% | 0.46 / 0.46 | 0.33 / 0.33 | 2.4 / 2.4 |
| 60 | 10^3 | -0.5% / -0.6% | 0.44 / 0.44 | 0.10 / 0.10 | 1.4 / 1.3 |
| 120 | 10^3 | +0.1% / -0.0% | 0.49 / 0.47 | 0.13 / 0.10 | 0.8 / 0.8 |

All rows, including the ridgeless ones for T = 60 and 120, are in `linear_model/table1_comparison.csv`.

**RFF backtest (`backtest/`).** P = 12,000 random Fourier features, z = 10^3:

| T | c = P/T | Sharpe ours (paper) | IR vs. market ours (paper) | Max loss ours (paper) | Skewness ours (paper) |
|---|---|---|---|---|---|
| 12 | 1000 | 0.495 (0.47) | 0.33 (0.31) | 1.23 (1.2) | 2.60 (2.5) |
| 60 | 200 | 0.470 | 0.30 | 0.57 | 1.33 |
| 120 | 100 | 0.433 | 0.27 | 0.39 | 0.55 |

The paper's rows for T = 60 and 120 use c = 1,000, which would need P = 60,000 and 120,000; here P is capped at 12,000, so only the T = 12 row is directly comparable.

**Positions versus recessions (Fig. 10).** The RFF model is long-only at heart; the position patterns for the three window lengths are highly correlated (0.90 / 0.85 / 0.99 versus 0.90 / 0.87 / 0.97 in the paper). For T = 12 the position before a recession is below its usual level in 12 of 14 recessions, compared with 6 of 14 for the linear model (our criterion: mean position in the 6 months before the recession versus the earlier mean).

**Simulations.** The closed-form curves and the Monte Carlo agree for the correctly specified model (Figs. 1-3) and for the misspecified one (Figs. 4-6): the ridgeless expected return is flat beyond cq = 1 and the ridgeless Sharpe ratio dips at cq = 1 ("double ascent") but not with shrinkage.

**Critique checks (`critique/`).** A 12-month time-series momentum strategy earns a Sharpe ratio of 0.43 against 0.50 for the RFF model, and forecasts built from persistent AR(1) noise, unrelated to returns, also earn a positive Sharpe ratio. The results depend on the random-feature bandwidth gamma and on the shrinkage grid, and the sample splits at 1975 give lower performance in the second half. See the slides for the discussion.

## Differences from the paper

* **Fewer random-feature draws.** The paper averages over 1,000 draws of the random weights; we use 20, 10 and 10 (T = 12, 60, 120), and a P grid of about 35 points.
* **Smaller P for longer windows,** as explained above.
* **R² of the RFF model** at T = 12 is -0.2% with the centered definition and +2.0% relative to the zero forecast; the paper reports +0.6%. Neither definition reproduces it, and we do not know the reason. The other statistics match closely.
* **Ridgeless rows** are very sensitive to implementation details because the matrix is nearly singular (for example, at T = 60 our ridgeless R² is -123% versus -96.6% in the paper).
* No Newey-West correction for the t-statistics.
* Variable importance is computed for the linear model only; the paper's Fig. 11 is for the nonlinear model.

## Conventions

* Row `t` of the data holds predictors at the end of month `t` and the return of month `t+1`. A window of length T trains on rows `t-T..t-1`, the position is taken at row `t` and earns `R_next[t]`.
* Ridge: `beta(z) = (zI + X'X/T)^-1 X'y/T`, or `X'(zT I + XX')^-1 y` in dual form; `z = 0` uses the pseudo-inverse.
* Predictors are divided by their standard deviation in the training window, without centering (paper's footnote 39).
* R² is centered: `1 - var(error)/var(return)`. Max loss is the worst monthly strategy return in units of the standardized market return. Sharpe ratios and information ratios are annualized.
* `misspec/voc_core.py: rolling_timing` expects the same-month standardized return (`R`), not `R_next`.

## Not included

The paper itself is not included; see the DOI above. `critique/critique_checks.py` needs the original Goyal-Welch spreadsheet (or internet access) because it rebuilds the data with `data/voc_data.py`.
