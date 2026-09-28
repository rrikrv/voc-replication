# Speaking notes (about 2 min): Table I and Fig. 10

**Slide 1: the linear kitchen sink at the interpolation boundary.**
With P = 15 predictors and a T = 12 month window, complexity is c = P/T = 1.25, so we are just past the interpolation boundary. Ridgeless regression inverts a nearly singular 15x15 matrix, beta-hat explodes, and out-of-sample R² is about -11,600% in our data (paper: -9,764%). The strategy's worst month is about 99 standard deviations (paper: 98.5).
Yet the Sharpe ratio is only about -0.11 rather than catastrophic: R² is dominated by the variance of the forecasts, not by whether their sign is right.

**Slide 2: shrinkage rescues it (`table1_comparison.png`).**
Same linear model with z = 10^3: R² about -4% (paper -3.8%), Sharpe 0.46 (paper 0.46), t = 4.4 (paper 4.4), IR vs. the market 0.33 (paper 0.33), worst month 2.4 std (paper 2.4). At T = 60 and 120 the Sharpe ratios are 0.44 and 0.47 (paper 0.44 and 0.49).
Point to make: R² is still negative, yet the timing strategy is profitable and significant. A second, independent implementation (`backtest/`) gives the same numbers.
Caveat: the ridgeless rows are sensitive to details (for example T = 60 R² is -123% vs. -96.6%).

**Slide 3: positions vs. recessions (`linear_model/fig10_positions_rff.png`, the paper's Fig. 10 for the nonlinear model).**
Positions of the RFF ridge model (P = 12,000, z = 10^3, gamma = 2), averaged over random-feature draws (20 for T = 12, 10 for T = 60, 6 for T = 120; the paper uses 1,000), with a 6-month moving average, as in the paper.
Three things match the paper: (i) the strategy is long-only at heart, negative positions are small (T = 12: 32% of months have a negative position, but the mean negative position is -0.03 vs. +0.10 for positive ones); (ii) the three window lengths give the same pattern, with position correlations 0.90 (T = 12 vs. 60), 0.85 (12 vs. 120) and 0.99 (60 vs. 120) vs. 0.90, 0.87, 0.97 in the paper; (iii) the position is near zero heading into most recessions: for T = 12 the mean position in the 6 months before a recession is below its earlier average in 12 of 14 recessions in our sample (the paper reports 14 of 15 with its own criterion, which it does not define precisely).
For contrast, the linear model (`fig10_positions_recessions.png`) does this in only 6 of 14 cases with the same criterion: divesting before recessions is a property of the complex nonlinear model, not of the linear one.
Say on the slide: the criterion is ours, and we use far fewer random-feature draws than the paper.

**Variable importance (`fig11_variable_importance.png`), for Q&A.**
In the linear model almost every predictor has negative importance: dropping it improves out-of-sample R², most of all for the slow valuation ratios dy, dp, de, lty, ep (0.15-0.35 pp). Only mkt_lag and ltr are non-negative, and negligibly so. Possible explanation (our hypothesis): persistent predictors are nearly constant within a 12-month window and just add noise. In the paper's nonlinear model the top variables are lag mkt, ltr and dfr, i.e. fast-moving ones.

**Role of gamma (eq. 21).** As gamma -> 0, sin(gamma * w'G) is approximately gamma * w'G, so the RFF features become random linear combinations of the 15 predictors. They span the same 15-dimensional space as our linear model, but the ridge penalty acts differently on them (after in-window standardization the scale is different), so the two models are related, not identical: in our check, RFF with gamma = 0.01 and z = 10^3 gives Sharpe 0.08, while our linear model gives 0.46. There is no single best gamma. At a fixed z = 10^3 the Sharpe ratio of the RFF model varies with gamma (0.27, 0.36, 0.49, 0.35 for gamma = 0.5, 1, 2, 4), but much of this is the shrinkage scale: with z = 10^4 (outside the paper's grid) gamma = 0.5 and 1 reach 0.47 and 0.51. In the paper the linear and nonlinear models contain complementary information (an equal-weighted blend has Sharpe 0.53).

**Q&A: "Why not the nonlinear rows of Table I?"** Those belong to the RFF backtest part of the project. We checked that the effect (shrinkage turns a useless-looking forecast into a profitable timing strategy) is already visible in the simple linear model.
