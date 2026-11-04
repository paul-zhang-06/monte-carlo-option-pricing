"""Cox-Ross-Rubinstein binomial tree: independent benchmark for AMERICAN (and European) vanilla options."""
from __future__ import annotations

import numpy as np


def binomial_price(S0, K, T, r, sigma, option_type="put", american=True, n=2000) -> float:
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(n + 1)
    S = S0 * u ** (n - j) * d**j
    V = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
    for i in range(n - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            S = S0 * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
            ex = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
            V = np.maximum(V, ex)
    return float(V[0])
