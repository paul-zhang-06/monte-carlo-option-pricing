"""Longstaff-Schwartz least-squares Monte Carlo for American options (exercise at every monitoring step).

Backward induction on one set of paths estimates, at each step t, the CONTINUATION value as a polynomial in S_t
fitted by OLS on in-the-money paths only.  Exercise when intrinsic value > fitted continuation value.

Two estimators of the price:
  * in_sample : value the same paths used to fit the regressions. Over-fitting pushes it UP, a sub-optimal rule pushes it DOWN;
                in this project's tests the two roughly cancel (mean error about +0.02 on a $12 option for degree >= 3).
  * two_pass  : value an INDEPENDENT set of paths with the fitted exercise rule. The rule is feasible, so this is a LOWER BOUND
                (up to noise): measured mean error about -0.05 to -0.08 on a $12 option with 20k paths. Default.
"""
from __future__ import annotations

import numpy as np

from .paths import simulate


def _basis(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, degree + 1, increasing=True)     # 1, x, x^2, ...  (x = S / K keeps it well conditioned)


def _intrinsic(S, K, option_type):
    return np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)


def lsm_fit(paths: np.ndarray, K, r, dt, option_type, degree=3):
    """Backward pass. Returns (coefficients per step, in-sample discounted cash flows per path)."""
    n, m = paths.shape
    N = m - 1
    disc = np.exp(-r * dt)
    value = _intrinsic(paths[:, N], K, option_type)         # cash flow at maturity
    coefs = [None] * (N + 1)
    for t in range(N - 1, 0, -1):
        value = value * disc                                # continuation cash flows discounted to step t
        ex = _intrinsic(paths[:, t], K, option_type)
        itm = ex > 0
        if itm.sum() <= degree + 1:
            continue
        X = _basis(paths[itm, t] / K, degree)
        beta, *_ = np.linalg.lstsq(X, value[itm], rcond=None)
        coefs[t] = beta
        cont = X @ beta
        do = ex[itm] > cont
        idx = np.where(itm)[0][do]
        value[idx] = ex[itm][do]
    return coefs, value * disc


def lsm_value(paths: np.ndarray, coefs, K, r, dt, option_type, degree=3) -> np.ndarray:
    """Forward pass on (possibly independent) paths: discounted cash flow of the fitted exercise rule, per path."""
    n, m = paths.shape
    N = m - 1
    alive = np.ones(n, dtype=bool)
    cash = np.zeros(n)
    for t in range(1, N):
        if coefs[t] is None:
            continue
        ex = _intrinsic(paths[:, t], K, option_type)
        cand = alive & (ex > 0)
        if not cand.any():
            continue
        cont = _basis(paths[cand, t] / K, degree) @ coefs[t]
        do = ex[cand] > cont
        idx = np.where(cand)[0][do]
        cash[idx] = ex[idx] * np.exp(-r * dt * t)
        alive[idx] = False
    cash[alive] = _intrinsic(paths[alive, N], K, option_type) * np.exp(-r * dt * N)
    return cash


def lsm_price(S0, K, r, sigma, dt, n_steps, n_paths, rng, option_type="put", degree=3, two_pass=True,
              antithetic=False) -> dict:
    train = simulate(S0, r, sigma, dt, n_steps, n_paths, rng, antithetic)
    coefs, in_sample = lsm_fit(train, K, r, dt, option_type, degree)
    if two_pass:
        test = simulate(S0, r, sigma, dt, n_steps, n_paths, rng, antithetic)
        cf = lsm_value(test, coefs, K, r, dt, option_type, degree)
    else:
        cf = in_sample
    # early exercise at t = 0 (immediately) is never worse than waiting if intrinsic is larger
    return dict(samples=cf, in_sample_price=float(in_sample.mean()), intrinsic=float(_intrinsic(S0, K, option_type)))
