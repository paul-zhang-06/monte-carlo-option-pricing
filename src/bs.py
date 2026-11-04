"""Closed-form benchmarks used to VALIDATE the Monte Carlo engine (zero dividends, r and sigma constant).

* Black-Scholes price and Greeks for European options.
* Discrete-monitoring GEOMETRIC Asian option (exact: the geometric average of lognormals is lognormal).
* Continuous-monitoring barrier options (Reiner-Rubinstein / Haug tables, zero rebate) and the Broadie-Glasserman-Kou
  continuity correction that shifts the barrier by exp(+/- 0.5826 sigma sqrt(dt)) to approximate DISCRETE monitoring.
All times are in years; the project uses 252 trading days per year.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

BETA_BGK = 0.5826   # = -zeta(1/2)/sqrt(2 pi)


def bs_price(S0, K, T, r, sigma, option_type="call") -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def bs_greeks(S0, K, T, r, sigma, option_type="call", days_per_year=252) -> dict:
    """delta, gamma, vega (per 1 vol point), theta (per trading day), rho (per 1% rate)."""
    sq = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    call = option_type == "call"
    delta = norm.cdf(d1) if call else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S0 * sigma * sq)
    vega = S0 * norm.pdf(d1) * sq / 100
    first = -S0 * norm.pdf(d1) * sigma / (2 * sq)
    theta_year = first - r * K * np.exp(-r * T) * norm.cdf(d2) if call else first + r * K * np.exp(-r * T) * norm.cdf(-d2)
    rho = (K * T * np.exp(-r * T) * (norm.cdf(d2) if call else -norm.cdf(-d2))) / 100
    return dict(delta=float(delta), gamma=float(gamma), vega=float(vega), theta=float(theta_year / days_per_year), rho=float(rho))


def geometric_asian_price(S0, K, T, r, sigma, n_obs, option_type="call") -> float:
    """Exact price of a discretely monitored geometric-average option; observations at t_i = i T/n, i = 1..n."""
    n = n_obs
    dt = T / n
    mean = np.log(S0) + (r - 0.5 * sigma**2) * dt * (n + 1) / 2
    var = sigma**2 * dt * (n + 1) * (2 * n + 1) / (6 * n)
    sd = np.sqrt(var)
    d1 = (mean + var - np.log(K)) / sd
    d2 = d1 - sd
    disc = np.exp(-r * T)
    fwd = np.exp(mean + 0.5 * var)
    if option_type == "call":
        return float(disc * (fwd * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(disc * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1)))


def barrier_price_continuous(S0, K, H, T, r, sigma, option_type, barrier_type) -> float:
    """Continuously monitored barrier option, zero rebate, zero dividends (Haug's A, B, C, D decomposition).

    barrier_type in {"up-and-in", "up-and-out", "down-and-in", "down-and-out"}.
    """
    b = r
    mu = (b - 0.5 * sigma**2) / sigma**2
    sq = sigma * np.sqrt(T)
    phi = 1.0 if option_type == "call" else -1.0
    eta = 1.0 if barrier_type.startswith("down") else -1.0
    x1 = np.log(S0 / K) / sq + (1 + mu) * sq
    x2 = np.log(S0 / H) / sq + (1 + mu) * sq
    y1 = np.log(H**2 / (S0 * K)) / sq + (1 + mu) * sq
    y2 = np.log(H / S0) / sq + (1 + mu) * sq
    disc = np.exp(-r * T)
    N = norm.cdf
    A = phi * S0 * N(phi * x1) - phi * K * disc * N(phi * x1 - phi * sq)
    B = phi * S0 * N(phi * x2) - phi * K * disc * N(phi * x2 - phi * sq)
    C = phi * S0 * (H / S0) ** (2 * (mu + 1)) * N(eta * y1) - phi * K * disc * (H / S0) ** (2 * mu) * N(eta * y1 - eta * sq)
    D = phi * S0 * (H / S0) ** (2 * (mu + 1)) * N(eta * y2) - phi * K * disc * (H / S0) ** (2 * mu) * N(eta * y2 - eta * sq)
    ge = K > H
    table = {
        ("down-and-in", "call"): C if ge else A - B + D,
        ("up-and-in", "call"): A if ge else B - C + D,
        ("down-and-in", "put"): B - C + D if ge else A,
        ("up-and-in", "put"): A - B + D if ge else C,
        ("down-and-out", "call"): A - C if ge else B - D,
        ("up-and-out", "call"): 0.0 if ge else A - B + C - D,
        ("down-and-out", "put"): A - B + C - D if ge else 0.0,
        ("up-and-out", "put"): B - D if ge else A - C,
    }
    return float(table[(barrier_type, option_type)])


def barrier_price_bgk(S0, K, H, T, r, sigma, option_type, barrier_type, dt) -> float:
    """Discrete-monitoring approximation: move the barrier AWAY from spot by exp(beta sigma sqrt(dt))."""
    shift = np.exp(BETA_BGK * sigma * np.sqrt(dt))
    H_adj = H * shift if barrier_type.startswith("up") else H / shift
    return barrier_price_continuous(S0, K, H_adj, T, r, sigma, option_type, barrier_type)
