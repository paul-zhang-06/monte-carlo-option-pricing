"""Risk-neutral GBM path simulation and the project's time-step rule.

    dS = r S dt + sigma S dW      (zero dividends)   =>   S_{t+dt} = S_t exp( (r - sigma^2/2) dt + sigma sqrt(dt) Z )

The log-Euler step is EXACT for GBM, so the only discretisation error comes from path-dependent features
(barrier / Asian monitoring dates), not from the dynamics.

Time-step rule (252 trading days per year):
    expiry < 30 trading days  -> 0.5-day steps       expiry >= 30 days -> 1-day steps
"""
from __future__ import annotations

import numpy as np

DAYS_PER_YEAR = 252
SHORT_EXPIRY_DAYS = 30


def step_rule(expiry_days: float, step_days: float | None = None) -> tuple[int, float, float]:
    """Return (n_steps, dt_years, step_days). `step_days` overrides the rule (used only for studies)."""
    if expiry_days <= 0:
        raise ValueError("expiry_days must be positive")
    sd = step_days if step_days is not None else (0.5 if expiry_days < SHORT_EXPIRY_DAYS else 1.0)
    n = max(1, int(round(expiry_days / sd)))
    T = expiry_days / DAYS_PER_YEAR
    return n, T / n, sd


def normals(rng: np.random.Generator, n_paths: int, n_steps: int, antithetic: bool) -> np.ndarray:
    """Standard normals (n_paths, n_steps). With antithetic variates, rows i and i + n/2 are Z and -Z."""
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_simulations must be even with antithetic variates")
        z = rng.standard_normal((n_paths // 2, n_steps))
        return np.concatenate([z, -z])
    return rng.standard_normal((n_paths, n_steps))


def simulate(S0, r, sigma, dt, n_steps, n_paths, rng, antithetic=False) -> np.ndarray:
    """GBM paths, shape (n_paths, n_steps + 1); column 0 is S0."""
    z = normals(rng, n_paths, n_steps, antithetic)
    logret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = S0
    np.cumsum(logret, axis=1, out=out[:, 1:])
    out[:, 1:] = S0 * np.exp(out[:, 1:])
    return out


def chunk_sizes(n_paths: int, n_steps: int, budget: int = 4_000_000, even: bool = False) -> list[int]:
    """Split n_paths into blocks so a block holds at most ~budget array elements (keeps memory bounded)."""
    per = max(2 if even else 1, budget // (n_steps + 1))
    if even and per % 2:
        per -= 1
    sizes, left = [], n_paths
    while left > 0:
        s = min(per, left)
        sizes.append(s)
        left -= s
    return sizes
