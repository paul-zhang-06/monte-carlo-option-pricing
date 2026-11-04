"""Monte Carlo option pricing engine: European, Asian, barrier (knock-in/out) and American (LSM) options under GBM.

    result = price_option(S0=100, K=100, expiry_days=90, r=0.05, sigma=0.2, option_type="call",
                          option_style="european", n_simulations=100_000, seed=42)
    result.price, result.std_err, result.ci_low, result.ci_high

Design notes
  * time step chosen automatically: 0.5 day if expiry < 30 days else 1 day (252 trading days / year);
  * barriers and Asian averages are monitored at EVERY simulated step (discrete monitoring);
  * `variance_reduction` in {"none", "antithetic", "control", "both"};
  * the standard error accounts for antithetic pairing (pairs are the independent samples) and for the control variate;
  * Greeks by central finite differences with COMMON RANDOM NUMBERS (same seed for every bumped run).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .bs import geometric_asian_price
from .lsm import lsm_price
from .paths import chunk_sizes, simulate, step_rule

STYLES = ("european", "asian", "barrier", "american")
BARRIERS = ("up-and-in", "up-and-out", "down-and-in", "down-and-out")


@dataclass
class PricingResult:
    price: float
    std_err: float
    ci_low: float
    ci_high: float
    n_simulations: int
    n_steps: int
    step_days: float
    runtime_s: float
    style: str
    variance_reduction: str
    greeks: dict | None = None
    extra: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"PricingResult(price={self.price:.4f}, se={self.std_err:.4f}, 95%CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
                f"{self.style}, steps={self.n_steps} x {self.step_days}d, {self.runtime_s:.2f}s)")


def _validate(S0, K, expiry_days, sigma, option_type, option_style, n, barrier, barrier_type, asian_average, vr):
    if S0 <= 0 or K <= 0 or sigma <= 0 or expiry_days <= 0:
        raise ValueError("S0, K, sigma and expiry_days must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if option_style not in STYLES:
        raise ValueError(f"option_style must be one of {STYLES}")
    if n < 2:
        raise ValueError("n_simulations must be at least 2")
    if vr not in ("none", "antithetic", "control", "both"):
        raise ValueError("variance_reduction must be none/antithetic/control/both")
    if option_style == "barrier":
        if barrier is None or barrier_type not in BARRIERS:
            raise ValueError(f"barrier options need barrier and barrier_type in {BARRIERS}")
        if barrier_type.startswith("up") and barrier <= S0:
            raise ValueError("an up barrier must be above S0")
        if barrier_type.startswith("down") and barrier >= S0:
            raise ValueError("a down barrier must be below S0")
    if option_style == "asian" and asian_average not in ("arithmetic", "geometric"):
        raise ValueError("asian_average must be 'arithmetic' or 'geometric'")


def _payoff(S, K, option_type):
    return np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)


def _samples(fn, S0, r, sigma, dt, n_steps, n, rng, antithetic):
    """Per-block payoffs -> independent samples (pair averages when antithetic). fn(paths) -> (x, y_or_None)."""
    xs, ys = [], []
    for c in chunk_sizes(n, n_steps, even=antithetic):
        x, y = fn(simulate(S0, r, sigma, dt, n_steps, c, rng, antithetic))
        if antithetic:
            h = c // 2
            x = 0.5 * (x[:h] + x[h:])
            y = None if y is None else 0.5 * (y[:h] + y[h:])
        xs.append(x)
        ys.append(y)
    x = np.concatenate(xs)
    y = None if ys[0] is None else np.concatenate(ys)
    return x, y


def _estimate(x, y=None, y_mean=None):
    """Mean and standard error; with a control variate y (known mean y_mean): x - beta (y - y_mean)."""
    if y is not None:
        beta = np.cov(x, y)[0, 1] / np.var(y, ddof=1)
        x = x - beta * (y - y_mean)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def price_option(S0, K, expiry_days, r, sigma, option_type="call", option_style="european", n_simulations=100_000,
                 barrier=None, barrier_type=None, asian_average="arithmetic", compute_greeks=False, seed=None,
                 variance_reduction="none", step_days=None, lsm_degree=3, lsm_two_pass=True,
                 n_steps_override=None) -> PricingResult:
    """Price one option by Monte Carlo. See module docstring."""
    _validate(S0, K, expiry_days, sigma, option_type, option_style, n_simulations, barrier, barrier_type, asian_average,
              variance_reduction)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_steps, dt, sd = step_rule(expiry_days, step_days)
    if n_steps_override is not None:     # used for theta: keep the SAME number of steps so common random numbers still match
        n_steps = n_steps_override
        dt = (expiry_days / 252) / n_steps
        sd = expiry_days / n_steps
    T = n_steps * dt
    disc = np.exp(-r * T)
    anti = variance_reduction in ("antithetic", "both")
    ctrl = variance_reduction in ("control", "both")
    n = n_simulations + (n_simulations % 2 if anti else 0)
    extra, y_mean = {}, None

    if option_style == "european":
        def fn(P):
            ST = P[:, -1]
            return disc * _payoff(ST, K, option_type), (disc * ST if ctrl else None)
        y_mean = S0                                              # E[e^{-rT} S_T] = S0 under the risk-neutral measure
        x, y = _samples(fn, S0, r, sigma, dt, n_steps, n, rng, anti)

    elif option_style == "asian":
        geo = asian_average == "geometric"

        def fn(P):
            m = P[:, 1:]
            avg = np.exp(np.log(m).mean(axis=1)) if geo else m.mean(axis=1)
            x = disc * _payoff(avg, K, option_type)
            y = None
            if ctrl and not geo:
                y = disc * _payoff(np.exp(np.log(m).mean(axis=1)), K, option_type)
            return x, y
        y_mean = geometric_asian_price(S0, K, T, r, sigma, n_steps, option_type)   # closed form control mean
        extra["geometric_closed_form"] = y_mean
        x, y = _samples(fn, S0, r, sigma, dt, n_steps, n, rng, anti)

    elif option_style == "barrier":
        up = barrier_type.startswith("up")
        knock_in = barrier_type.endswith("in")

        def fn(P):
            m = P[:, 1:]
            hit = (m.max(axis=1) >= barrier) if up else (m.min(axis=1) <= barrier)    # monitored at every step
            pay = disc * _payoff(P[:, -1], K, option_type)
            return (pay * hit if knock_in else pay * (~hit)), None
        x, y = _samples(fn, S0, r, sigma, dt, n_steps, n, rng, anti)

    else:   # american
        out = lsm_price(S0, K, r, sigma, dt, n_steps, n, rng, option_type, lsm_degree, lsm_two_pass, anti)
        cf = out["samples"]
        x = 0.5 * (cf[: n // 2] + cf[n // 2:]) if anti else cf
        y = None
        extra.update(in_sample_price=out["in_sample_price"], intrinsic=out["intrinsic"], lsm_degree=lsm_degree,
                     two_pass=lsm_two_pass)

    price, se = _estimate(x, y if ctrl and y is not None else None, y_mean)
    res = PricingResult(price=price, std_err=se, ci_low=price - 1.96 * se, ci_high=price + 1.96 * se, n_simulations=n,
                        n_steps=n_steps, step_days=sd, runtime_s=time.perf_counter() - t0, style=option_style,
                        variance_reduction=variance_reduction, extra=extra)
    if compute_greeks:
        res.greeks = finite_difference_greeks(S0, K, expiry_days, r, sigma, option_type, option_style, n_simulations,
                                              barrier, barrier_type, asian_average, seed, variance_reduction, step_days,
                                              lsm_degree, lsm_two_pass)
        res.runtime_s = time.perf_counter() - t0
    return res


def finite_difference_greeks(S0, K, expiry_days, r, sigma, option_type, option_style, n, barrier, barrier_type, asian_average,
                             seed, vr, step_days, lsm_degree, lsm_two_pass, rel_bump=0.01, common_random_numbers=True) -> dict:
    """Central-difference Greeks. Every bumped price re-uses the SAME seed (common random numbers), so the
    sampling error largely cancels in the difference. Units: delta per $1, gamma per $1^2, vega per vol point,
    theta per trading day, rho per 1% rate."""
    if seed is None:
        seed = 12345

    def P(**kw):
        a = dict(S0=S0, K=K, expiry_days=expiry_days, r=r, sigma=sigma)
        a.update(kw)
        s = seed if common_random_numbers else int(np.random.default_rng().integers(1 << 30))
        return price_option(a["S0"], a["K"], a["expiry_days"], a["r"], a["sigma"], option_type, option_style, n, barrier,
                            barrier_type, asian_average, False, s, vr, step_days, lsm_degree, lsm_two_pass, a.get("n_steps")).price

    h = rel_bump * S0
    up, dn, mid = P(S0=S0 + h), P(S0=S0 - h), P()
    delta = (up - dn) / (2 * h)
    gamma = (up - 2 * mid + dn) / h**2
    vega = (P(sigma=sigma + 0.01) - P(sigma=sigma - 0.01)) / 2          # per 1 vol point
    rho = (P(r=r + 0.01) - P(r=r - 0.01)) / 2                           # per 1% rate
    base_steps = step_rule(expiry_days, step_days)[0]
    theta = P(expiry_days=expiry_days - 1, n_steps=base_steps) - mid if expiry_days > 1 else float("nan")  # 1 trading day closer
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
