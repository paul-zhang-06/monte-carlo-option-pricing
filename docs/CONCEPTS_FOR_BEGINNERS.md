# Concepts for beginners: Monte Carlo option pricing

## 1. Options and payoffs
A **call** pays `max(S_T - K, 0)` at expiry, a **put** pays `max(K - S_T, 0)`. Path-dependent options care about the whole price path:
* **Asian**: payoff uses the *average* price over the life (cheaper than European: averaging reduces volatility).
* **Barrier**: *knock-out* dies if the price touches a barrier; *knock-in* only comes alive if it does. Up/down says which side the barrier is on.
* **American**: can be exercised any day before expiry.

## 2. Risk-neutral pricing
The price today = discounted **expected payoff under the risk-neutral measure**, where the stock grows at the risk-free rate r: `price = e^{-rT} E[payoff]`. We estimate the expectation by simulating many possible futures and averaging.

## 3. Geometric Brownian motion
`dS = r S dt + sigma S dW`. In steps: `S_{t+dt} = S_t * exp((r - sigma^2/2) dt + sigma sqrt(dt) Z)` with `Z ~ N(0,1)`. The `-sigma^2/2` correction keeps the expected growth equal to r. This step is exact, not an approximation.

## 4. Monte Carlo error
Average of N payoffs has **standard error = std / sqrt(N)**. To halve the error you need 4 times more paths (a log-log plot of error vs N has slope -1/2). The 95% **confidence interval** is price +/- 1.96 SE.

## 5. Variance reduction
* **Antithetic variates**: for every path using shocks Z also simulate -Z. The two payoffs are negatively correlated, so their average has lower variance. The pairs (not single paths) are the independent samples.
* **Control variate**: pick a related quantity with a KNOWN expectation (e.g. the geometric Asian option, whose price has a formula) and correct the estimate: `X - beta (Y - E[Y])`. The more correlated, the bigger the gain (62x for the Asian option).

## 6. The time step and discrete monitoring
We check barriers and averages only at simulated times. A real barrier is monitored continuously, so discretely monitored knock-out options are worth *more* (the path may cross between observations). Rule here: half-day steps if expiry < 30 days, else daily. The Broadie-Glasserman-Kou correction shifts the barrier by `exp(0.5826 sigma sqrt(dt))` to mimic discrete monitoring in a continuous formula.

## 7. American options and Longstaff-Schwartz
At each date you compare **exercising now** with **waiting**. The value of waiting is unknown, so LSM estimates it by regressing future discounted cash flows on today's stock price (only using in-the-money paths). Work backwards from expiry. The estimated rule is then applied to fresh paths ("two-pass") so the result is a legitimate lower bound.

## 8. Early exercise
An American call on a stock with no dividends is never exercised early, so it equals the European call. An American put can be worth exercising early (to earn interest on the strike): the *early exercise premium*, bigger for deep in-the-money, long-dated options and high rates.

## 9. Greeks by bumping
Delta = how the price changes with S; gamma = how delta changes; vega = volatility; theta = time; rho = interest rate. Compute `(price(S+h) - price(S-h)) / 2h` with **the same random numbers** in both runs ("common random numbers"), otherwise noise swamps the tiny difference.

## 10. Validation mindset
Never trust a pricer without a benchmark: Black-Scholes (European), closed-form geometric Asian, barrier formulas, binomial tree (American), parity relations (call-put, knock-in + knock-out = vanilla).

## 11. Code map
`paths.py` (step rule, GBM) -> `pricer.py` (engine, variance reduction, Greeks) -> `lsm.py` (American) -> `bs.py`, `binomial.py` (benchmarks) -> `scripts/run_pipeline.py`.

## 12. Exercises
* Price a put-version of every style and check put-call parity for the European.
* Replace pseudo-random numbers by Sobol points (`scipy.stats.qmc`): how does the error slope change?
* Add a dividend yield q and see when an American call is exercised early.
* Implement a lookback option and validate against a known formula.
