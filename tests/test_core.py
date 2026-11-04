import numpy as np
import pytest

from src.binomial import binomial_price
from src.bs import (barrier_price_bgk, barrier_price_continuous, bs_greeks, bs_price, geometric_asian_price)
from src.paths import step_rule
from src.pricer import price_option

S0, K, R, SIG = 100.0, 100.0, 0.05, 0.2


def _z(res, truth):
    return (res.price - truth) / res.std_err


@pytest.mark.parametrize("days", [10, 90])
@pytest.mark.parametrize("otype", ["call", "put"])
def test_european_matches_black_scholes(days, otype):
    res = price_option(S0, K, days, R, SIG, otype, "european", 200_000, seed=7, variance_reduction="antithetic")
    assert abs(_z(res, bs_price(S0, K, days / 252, R, SIG, otype))) < 4


def test_put_call_parity_under_monte_carlo():
    c = price_option(S0, K, 90, R, SIG, "call", "european", 100_000, seed=3)
    p = price_option(S0, K, 90, R, SIG, "put", "european", 100_000, seed=3)
    assert c.price - p.price == pytest.approx(S0 - K * np.exp(-R * 90 / 252), abs=0.06)


def test_seed_reproducibility():
    a = price_option(S0, K, 60, R, SIG, n_simulations=20_000, seed=11)
    b = price_option(S0, K, 60, R, SIG, n_simulations=20_000, seed=11)
    c = price_option(S0, K, 60, R, SIG, n_simulations=20_000, seed=12)
    assert a.price == b.price and a.price != c.price


def test_time_step_rule():
    assert step_rule(29)[0] == 58 and step_rule(29)[2] == 0.5     # short expiry: half-day steps
    assert step_rule(30)[0] == 30 and step_rule(30)[2] == 1.0     # 30 days or more: daily steps
    assert step_rule(252)[0] == 252
    assert price_option(S0, K, 10, R, SIG, n_simulations=1000, seed=1).n_steps == 20


def test_variance_reduction_lowers_standard_error():
    se = {v: price_option(S0, K, 90, R, SIG, "call", "european", 100_000, seed=5, variance_reduction=v).std_err
          for v in ["none", "antithetic", "control", "both"]}
    assert se["antithetic"] < se["none"] and se["control"] < se["none"] and se["both"] < min(se["antithetic"], se["control"])


def test_standard_error_is_calibrated_with_antithetic_pairs():
    prices, ses = [], []
    for s in range(150):
        r = price_option(S0, K, 30, R, SIG, "call", "european", 2000, seed=500 + s, variance_reduction="antithetic")
        prices.append(r.price); ses.append(r.std_err)
    assert np.mean(ses) == pytest.approx(np.std(prices, ddof=1), rel=0.2)


def test_barrier_in_plus_out_equals_vanilla_on_same_paths():
    for bt_in, bt_out, H in [("up-and-in", "up-and-out", 115), ("down-and-in", "down-and-out", 85)]:
        i = price_option(S0, K, 60, R, SIG, "call", "barrier", 50_000, H, bt_in, seed=9)
        o = price_option(S0, K, 60, R, SIG, "call", "barrier", 50_000, H, bt_out, seed=9)
        v = price_option(S0, K, 60, R, SIG, "call", "european", 50_000, seed=9)
        assert i.price + o.price == pytest.approx(v.price, abs=1e-9)


def test_closed_form_barrier_parity_and_bgk_direction():
    for otype in ("call", "put"):
        for H, side in [(120, "up"), (80, "down")]:
            van = bs_price(S0, K, 0.25, R, SIG, otype)
            i = barrier_price_continuous(S0, K, H, 0.25, R, SIG, otype, f"{side}-and-in")
            o = barrier_price_continuous(S0, K, H, 0.25, R, SIG, otype, f"{side}-and-out")
            assert i + o == pytest.approx(van, abs=1e-9)
    # discrete monitoring is monitored less often, so knock-OUT options are worth MORE than the continuous formula
    cont = barrier_price_continuous(S0, K, 115, 0.25, R, SIG, "call", "up-and-out")
    assert barrier_price_bgk(S0, K, 115, 0.25, R, SIG, "call", "up-and-out", 1 / 252) > cont


def test_discrete_barrier_mc_close_to_bgk():
    res = price_option(S0, K, 90, R, SIG, "call", "barrier", 300_000, 115, "up-and-out", seed=2, variance_reduction="antithetic")
    assert abs(res.price - barrier_price_bgk(S0, K, 115, 90 / 252, R, SIG, "call", "up-and-out", 1 / 252)) < 0.03


def test_asian_ordering_and_geometric_closed_form():
    arith = price_option(S0, K, 90, R, SIG, "call", "asian", 100_000, seed=4, variance_reduction="control")
    geo = price_option(S0, K, 90, R, SIG, "call", "asian", 200_000, seed=4, asian_average="geometric")
    euro = bs_price(S0, K, 90 / 252, R, SIG, "call")
    assert abs(_z(geo, geometric_asian_price(S0, K, 90 / 252, R, SIG, 90))) < 4
    assert geo.price < arith.price < euro                       # AM-GM, and averaging reduces volatility


def test_asian_control_variate_is_consistent_with_plain():
    plain = price_option(S0, K, 90, R, SIG, "call", "asian", 300_000, seed=6)
    cv = price_option(S0, K, 90, R, SIG, "call", "asian", 100_000, seed=8, variance_reduction="control")
    assert abs(plain.price - cv.price) < 4 * np.hypot(plain.std_err, cv.std_err)
    assert cv.std_err < plain.std_err / 5


def test_american_put_vs_binomial_and_ordering():
    am = price_option(S0, 105, 90, R, SIG, "put", "american", 100_000, seed=1)
    tree = binomial_price(S0, 105, 90 / 252, R, SIG, "put", True, 2000)
    euro = bs_price(S0, 105, 90 / 252, R, SIG, "put")
    assert abs(am.price - tree) < 4 * am.std_err + 0.01
    assert am.price > euro


def test_american_call_without_dividends_equals_european():
    am = price_option(S0, K, 90, R, SIG, "call", "american", 100_000, seed=2)
    assert abs(am.price - bs_price(S0, K, 90 / 252, R, SIG, "call")) < 4 * am.std_err + 0.02


def test_greeks_match_black_scholes():
    g = price_option(S0, K, 90, R, SIG, "call", "european", 200_000, seed=5, compute_greeks=True).greeks
    b = bs_greeks(S0, K, 90 / 252, R, SIG, "call")
    assert g["delta"] == pytest.approx(b["delta"], abs=0.01)
    assert g["gamma"] == pytest.approx(b["gamma"], abs=0.003)
    assert g["vega"] == pytest.approx(b["vega"], abs=0.01)
    assert g["theta"] == pytest.approx(b["theta"], abs=0.003)
    assert g["rho"] == pytest.approx(b["rho"], abs=0.01)


def test_input_validation():
    with pytest.raises(ValueError):
        price_option(S0, K, 90, R, -0.1)
    with pytest.raises(ValueError):
        price_option(S0, K, 90, R, SIG, option_style="barrier", barrier=90, barrier_type="up-and-out")
    with pytest.raises(ValueError):
        price_option(S0, K, 90, R, SIG, option_style="lookback")
