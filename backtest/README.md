# backtest: rolling-window out-of-sample experiment (Figs. 7-9, Table I)

Out-of-sample backtest of the random-feature ridge timing strategy and of the linear kitchen-sink benchmark.

## Files

| File | Contents |
|---|---|
| `config.py` | data path, list of predictors, window length T, grids, number of seeds, choice of engine |
| `adapters.py` | connects the shared RFF / dual-ridge code in `../misspec/voc_core.py` to the backtest |
| `data.py` | loading and alignment: row t = (G_t, R_{t+1}) |
| `features.py` | random Fourier features (eq. 20) and raw predictors |
| `engines.py` | fast ridge in dual form, and a wrapper for any external ridge implementation |
| `backtest.py` | loop over windows, metrics, averaging over seeds, comparison with Table I |
| `plots.py` | Figs. 7, 8 and 9 in the axes used by the paper |
| `test_pipeline.py` | correctness checks |
| `run.py` | command-line entry point |

## Running

```bash
pip install numpy pandas matplotlib          # scikit-learn is only needed for the example in adapters.py
python test_pipeline.py                      # 6 checks must pass
python run.py --synthetic --quick            # smoke test on artificial data, seconds
python run.py --kitchen-sink                 # linear model: comparison with Table I
python run.py                                # main run (T and seeds from config.py)
python run.py --T 60 --seeds 10              # Fig. 9, part 1
python run.py --T 120 --seeds 10             # Fig. 9, part 2
python plots.py --fig9 results/agg_T60.csv results/agg_T120.csv
```

Time per seed (P up to 12,000, one core): T = 12 about 3 s, T = 60 about 20 s, T = 120 about 70 s.
`N_JOBS` in `config.py` parallelizes over seeds.

Results are written to `results/`: `per_seed_*.csv` (every seed), `agg_*.csv` (mean and `_std` over seeds), `fig*.png`.
The committed results are from 20 seeds for T = 12 and 10 seeds for T = 60 and T = 120.

## Data

`../data/voc_data.csv`, 1929-11 to 2020-12. Predictors are already standardized (expanding std);
`R_next` is the return of month t+1 divided by the volatility of months t-11..t (no look-ahead, checked).
The estimation sample 1930-01 to 2020-12 (1,092 months, as in the paper) is set by `START_DATE` and `END_DATE`.

If the data format changes, edit `config.py`:

- `PREDICTORS`: list of predictor columns; add or remove names, nothing else has to change;
- `TARGET_IS_NEXT_MONTH`: whether row t already holds the return of month t+1;
- `ADD_LAG_MKT`: add the return of month t as a predictor (if it is not in `PREDICTORS`);
- `STANDARDIZE_*`: switch on if the data are **raw**.

## Engine

The main run uses the shared RFF / ridge code (`FEATURES = "core"`, `ENGINE = "core"` in `config.py`).
It was checked against an independent implementation (`features.py`, `engines.py`):

- the RFF features are bit-for-bit identical;
- forecasts and ||beta||^2 agree to 1e-11 over the whole backtest (all P and z);
- the engine's own `rolling_timing` gives the same forecasts on the same data.

Test: `test_core_matches` in `test_pipeline.py`.

Note that `rff(G, P)` gives **non-nested** features for different P (the weight matrix has a different shape and is filled with different random numbers).
The features are therefore generated once at P = 12,000 and the model with P features uses the first P columns, as in step (ii) of the paper.
`ENGINE = "dual"` gives the same result about twice as fast.

## Conventions

- Forecast in month t: train on the pairs (S_{t-T}, R_{t-T+1}), ..., (S_{t-1}, R_t), test on S_t.
- Features are divided by their standard deviation in the training window, without centering and without an intercept (footnotes 35 and 39).
- Ridge as parameterized in the paper: `beta(z) = (zI + X'X/T)^{-1} X'y/T`.
- `r2`: 1 - SSE / sum (R - mean R)^2, as in footnote 40 (for the linear model it matches Table I: -3.96% vs. -3.8%).
  `r2_zero`: 1 - SSE / sum R^2, relative to the zero forecast, as in the theory. The two differ by a roughly constant shift of about 2 pp.
  For the nonlinear model at T = 12 the paper reports +0.6%; we get -0.2% (`r2`) and +2.0% (`r2_zero`), so neither definition reproduces it.
- `max_loss`: worst monthly strategy return, -min(r), in units of the standardized market return (as in Table I).
- Sharpe ratio and IR are annualized (times sqrt(12)); mean return, volatility and alpha are monthly.
- Panel B of Fig. 7 is the mean ||beta||^2 over windows (the text of the paper says ||beta||^2, the figure is labelled ||beta||).
- Differences from the paper: the number of seeds instead of 1,000, and a P grid of about 35 points.
