# Monte Carlo Option Pricing Engine

A reusable, vectorised Monte Carlo engine that prices **European, Asian, barrier (knock-in/out) and American** options under risk-neutral geometric Brownian motion (zero dividends), with standard errors, confidence intervals,
variance reduction and finite-difference Greeks. Every price is **validated against an independent closed form or tree**.

```python
from src.pricer import price_option
res = price_option(S0=100, K=100, expiry_days=90, r=0.05, sigma=0.2, option_type="call",
                   option_style="barrier", barrier=115, barrier_type="up-and-out",
                   n_simulations=200_000, variance_reduction="antithetic", compute_greeks=True, seed=42)
res.price, res.std_err, (res.ci_low, res.ci_high), res.greeks
```

## Simulation rules (from the specification)
252 trading days/year; **expiry < 30 days -> 0.5-day steps, otherwise 1-day steps**; barriers and Asian averages are monitored at every simulated step; fixed seed for reproducibility; zero dividends.

## Key results (all in `outputs/`)
| Check | Result |
|---|---|
| **European vs Black-Scholes** (30 cases: expiry 10-252d, K 90/100/110, call/put) | 93% inside the 95% CI, max abs error $0.049, max |z| = 2.2 |
| **Standard error calibration** (300 seeds) | z-scores mean -0.05, sd 1.00; 4.7% outside +/-1.96 (expected 5%) |
| Error vs N | log-log slope **-0.4999** (theory -1/2) |
| **Variance reduction** (100k paths, European, 90d) | antithetic 1.4x lower SE, control variate 2.3x, both **4.6x** (efficiency per second 6x / 13x / 64x) |
| Asian (arithmetic) with geometric-average control variate | SE falls **62x** (0.0144 -> 0.0002): efficiency 3,500x |
| Asian vs European (ATM, 90d, sigma 0.1-0.4) | Asian price is 56-57% of the European price |
| **Barrier**, 8 types, discrete monitoring | match the Broadie-Glasserman-Kou-corrected closed form to within $0.011; the *continuous* formula is off by up to $0.20 (up-and-in call: 4.02 MC vs 4.22 continuous) |
| Monitoring step (20d up-and-out call): 4-day to 0.25-day steps | MC price 1.283 -> 1.056 (continuous limit 0.962) |
| **Time-step rule** (29 days, 0.5-day vs 1-day steps) | up-and-out call 6.1% lower with half-day steps (0.860 vs 0.915); Asian 1.3% lower |
| **American put (LSM)** vs 2,000-step binomial tree | max gap $0.023, mean -$0.010 (LSM is a lower bound); early-exercise premium up to $3.02 (1y, S0=80) |
| American call, no dividends | 10.419 +/- 0.046 vs European 10.451: **no early exercise**, as theory says |
| **Greeks** (central FD, common random numbers) vs Black-Scholes | all within 0.4% (call/put delta, gamma, vega, theta, rho) |
| Common random numbers | delta RMSE 0.0022 vs 0.76 (tiny bump, independent seeds) |
| Runtime (100k paths, 90d) | European 0.25s, Asian 0.26s, barrier 0.28s, American (LSM, two passes) 1.8s |

## Method notes
* **Exact GBM step** `S_{t+dt} = S_t exp((r - sigma^2/2) dt + sigma sqrt(dt) Z)`: no discretisation error for the dynamics; error comes only from monitoring discrete dates.
* **Standard error**: sample std / sqrt(n); with antithetic variates the *pairs* are the independent samples; with a control variate the estimator is `X - beta (Y - E[Y])`, beta = cov/var. Verified by test (SE vs repeated-run std within 20%).
* **Control variates**: European uses `e^{-rT} S_T` (mean S0). Arithmetic Asian uses the **geometric Asian** payoff (closed form for discrete monitoring).
* **Barrier**: `knock-in + knock-out = vanilla` holds exactly on the same paths (tested); closed forms by Reiner-Rubinstein (Haug) with BGK shift `exp(+/- 0.5826 sigma sqrt(dt))` for discrete monitoring.
* **American (Longstaff-Schwartz)**: regress continuation value on a cubic in S/K using in-the-money paths; exercise when intrinsic > fitted continuation. Default **two-pass**: policy fitted on one path set, valued on an independent set, so the result is a feasible-policy lower bound.
  Measured bias vs the tree (12 seeds, 20k paths, 1y put): two-pass about -0.08 +/- 0.02 (0.7%), in-sample about +0.02; a linear basis is too poor (-0.27); degree >= 3 is enough.
* **Greeks**: central differences, bump 1% of S0, 1 vol point, 1% rate, theta = one trading day closer with the *same number of steps* so that common random numbers still match.
* **Memory**: paths are generated in blocks; LSM needs the full path matrix (100k x 253 doubles ~ 200 MB).

## Repository
`src/` (`pricer.py` engine, `paths.py` GBM and step rule, `lsm.py`, `bs.py` closed forms, `binomial.py`) - `tests/` (18 tests) - `scripts/run_pipeline.py` (all experiments, ~7 min) - `notebooks/` - `reports/report.md` - `docs/`.

```bash
python -m venv .venv && .venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest                                # ~20 s
python scripts/run_pipeline.py        # regenerates every table and figure (or: ... american  to run one section)
```
No external data is needed. Optional extension: estimate sigma from `yfinance` prices.

## Limitations
* GBM only (constant sigma and r, no jumps, no stochastic vol, no dividends): it cannot reproduce the volatility smile; barrier prices in particular are model-dependent.
* LSM exercises on the monitoring grid (Bermudan, daily), so it is slightly below a continuous-exercise American price; basis choice affects the result; regression uses a single state variable.
* Finite-difference Greeks of barrier options are noisy near the barrier (discontinuous payoff); pathwise or likelihood-ratio methods would be better.
* Discrete barrier monitoring: 1 observation per step, no intraperiod Brownian-bridge correction (BGK is used only as a benchmark).
* Pseudo-random numbers; quasi-Monte Carlo (Sobol) would converge faster for smooth payoffs.

## Possible extensions
Sobol sequences and Brownian-bridge construction; Brownian-bridge barrier correction; pathwise Greeks; stochastic volatility (Heston); local volatility; dividends; calibration to market option chains.
