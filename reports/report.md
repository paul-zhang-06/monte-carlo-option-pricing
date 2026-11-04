# Report: Monte Carlo option pricing engine

## 1. Objective
Build a flexible engine pricing European, Asian, barrier and American options under risk-neutral GBM, quantify simulation error, and study convergence, variance reduction, discretisation (time-step) effects and Greeks.

## 2. Engine
See README. Single entry point `price_option(...)` with the specified inputs, plus `variance_reduction`, `step_days` (study override) and LSM options. Results carry price, standard error, 95% CI, steps, runtime and optional Greeks.

## 3. Validation (every style against an independent benchmark)
European: Black-Scholes (30 cases, 93% inside 95% CI). Geometric Asian: exact closed form. Barrier: Reiner-Rubinstein with BGK discrete-monitoring correction (max gap $0.011) and in+out = vanilla identity. American: CRR tree with 2,000 steps (max gap $0.023) and the European/American-call equivalence without dividends. Greeks: Black-Scholes (max relative error 0.4%).

## 4. Findings
1. Error shrinks like N^{-1/2} (fitted slope -0.4999); to halve the error you need 4x the paths.
2. Variance reduction is very uneven: antithetic 1.2-1.7x lower SE across styles; control variates give 2.3x (European) and 62x (Asian arithmetic, where the control is almost perfectly correlated).
3. **Discrete monitoring matters**: ignoring it with the continuous barrier formula is off by up to $0.20 on a $1.6-4 option; the BGK shift removes most of the gap. A 4-day to 0.25-day monitoring step changes a 20-day up-and-out price from 1.283 to 1.056.
4. **The 0.5-day rule for short expiries is not cosmetic**: at 29 days the up-and-out price is 6% lower with half-day than with daily monitoring; Asian prices differ by 1.3%. European prices are unaffected (exact GBM).
5. Early exercise: American put premium grows with maturity and moneyness (up to $3.02 for 1y, S0=80, K=100); American call = European call without dividends.
6. LSM: two-pass is a lower bound; accuracy depends on basis degree (linear bad, cubic good).
7. Common random numbers make finite-difference Greeks usable: with independent seeds the delta from a small bump is garbage (RMSE 0.76), with CRN it is 0.002.

## 5. Limitations and extensions
See README.
