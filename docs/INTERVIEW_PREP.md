# Interview preparation: Monte Carlo option pricing engine

**Pitch.** "I built a Monte Carlo engine pricing European, Asian, barrier and American options under risk-neutral GBM, with antithetic and control variates, LSM for early exercise and finite-difference Greeks.
Every style is validated against an independent benchmark: Black-Scholes, closed-form geometric Asian, barrier formulas with the Broadie-Glasserman-Kou correction, and a binomial tree. The standard error is calibrated (z-scores mean -0.05, sd 1.00 over 300 seeds)."
**Why does the error go like 1/sqrt(N)?** Central limit theorem; fitted slope -0.4999.
**Variance reduction?** Antithetic: pair averaging, negative correlation, SE reduced 1.4x for the European call; control variate: 2.3x; both 4.6x; Asian arithmetic with geometric control: 62x. Pairs are the independent samples, so the SE must use pair averages (tested).
**Why do discrete barriers differ from the formula?** Fewer monitoring points than continuous: knock-outs are worth more; BGK shifts the barrier by exp(0.5826 sigma sqrt(dt)); max gap to MC $0.011 vs $0.20 for the continuous formula.
**Why 0.5-day steps for short expiries?** Few daily observations make monitoring coarse; at 29 days the up-and-out price differs by 6% between half-day and daily steps.
**How does LSM work, and what is its bias?** Backward induction, regression of continuation value on S (cubic) using ITM paths; exercise if intrinsic > estimate. Two-pass is a lower bound (feasible rule): about -0.08 (0.7%) on a $12 option in my tests; in-sample was ~ +0.02. Linear basis -0.27.
**Is the American call special?** Without dividends it equals the European call (LSM 10.419 +/- 0.046 vs 10.451).
**Greeks?** Central differences with common random numbers. Without CRN, delta RMSE 0.76 at a 0.05% bump; with CRN 0.0022. Theta bumps expiry by one day keeping the same number of steps, else CRN breaks (my first version gave a 16% theta error). Barrier Greeks are noisy near the barrier.
**Limitations?** Constant volatility (no smile), no dividends, Bermudan approximation of American, pseudo-random numbers, discrete monitoring only.
**What next?** Sobol + Brownian bridge; pathwise Greeks; Heston/local vol; calibration to an option chain.
**Bug story.** Theta was 16% off because bumping expiry changed the number of steps and thus the random stream; fix: keep n_steps fixed. Another: the single-seed LSM bias chart contradicted theory; averaging over seeds showed it was noise.
