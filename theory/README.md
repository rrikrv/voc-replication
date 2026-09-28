# theory: Figs. 1-3 (correctly specified model)

Psi = I, b* = 0.2, sigma^2 = 1. `voc_figs.py` produces Figs. 1-3:
out-of-sample R², ||beta||^2, expected return, second moment and Sharpe ratio as functions of c = P/T for several ridge parameters z.
Lines are the closed-form expressions of Propositions 3-4 (Marchenko-Pastur Stieltjes transform of footnote 24); dots are a Monte Carlo simulation (T = 100, 30 seeds).

```bash
python voc_figs.py 30     # about 3 seconds; writes fig1.png, fig2.png, fig3.png and prints sanity checks
```

Cross-check: the analytic curves agree with the independent implementation in `misspec/voc_theory.py` (at q = 1) to 2e-6.
