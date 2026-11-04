"""Run every experiment and write tables to outputs/tables and figures to outputs/figures.  ~8-12 minutes."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.binomial import binomial_price
from src.bs import (barrier_price_bgk, barrier_price_continuous, bs_greeks, bs_price, geometric_asian_price)
from src.paths import simulate, step_rule
from src.pricer import _samples, finite_difference_greeks, price_option

FIG, TAB = Path("outputs/figures"), Path("outputs/tables")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
S0, K, R, SIG = 100.0, 100.0, 0.05, 0.20
SUMMARY: dict = {}


def save(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def yrs(d):
    return d / 252


# 1 ---------------------------------------------------------------------------------------------------------------------
def validation():
    rows = []
    for T in (10, 29, 30, 90, 252):
        for k in (90, 100, 110):
            for ot in ("call", "put"):
                r = price_option(S0, k, T, R, SIG, ot, "european", 200_000, seed=T * 7 + k, variance_reduction="antithetic")
                bs = bs_price(S0, k, yrs(T), R, SIG, ot)
                rows.append(dict(expiry_days=T, K=k, type=ot, steps=r.n_steps, step_days=r.step_days, mc=r.price, se=r.std_err,
                                 bs=bs, error=r.price - bs, z=(r.price - bs) / r.std_err, inside_ci=abs(r.price - bs) <= 1.96 * r.std_err))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "01_european_validation.csv", index=False)
    SUMMARY["validation"] = dict(n=len(df), frac_inside_95ci=float(df["inside_ci"].mean()), max_abs_z=float(df["z"].abs().max()),
                                 max_abs_error=float(df["error"].abs().max()))
    # calibration of the standard error: z-scores over many independent seeds
    zs = []
    for sd in range(300):
        r = price_option(S0, K, 30, R, SIG, "call", "european", 20_000, seed=9000 + sd)
        zs.append((r.price - bs_price(S0, K, yrs(30), R, SIG, "call")) / r.std_err)
    zs = np.array(zs)
    SUMMARY["z_calibration"] = dict(mean=float(zs.mean()), sd=float(zs.std()), frac_outside_196=float((abs(zs) > 1.96).mean()))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(zs, bins=25, density=True, alpha=.7)
    x = np.linspace(-4, 4, 200)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "r")
    ax.set(title="Monte Carlo z-scores over 300 seeds vs N(0,1): unbiased, SE calibrated", xlabel="(MC - BS) / SE")
    save(fig, "01_z_score_calibration.png")


# 2 ---------------------------------------------------------------------------------------------------------------------
def convergence():
    Ns = [1000, 3000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    rows = []
    for vr in ("none", "antithetic", "control", "both"):
        for n in Ns:
            r = price_option(S0, K, 90, R, SIG, "call", "european", n, seed=21, variance_reduction=vr)
            rows.append(dict(vr=vr, n=n, price=r.price, se=r.std_err, runtime=r.runtime_s))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "02_convergence_european.csv", index=False)
    bs = bs_price(S0, K, yrs(90), R, SIG, "call")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for vr, g in df.groupby("vr"):
        axes[0].errorbar(g["n"], g["price"], yerr=1.96 * g["se"], marker="o", ms=3, capsize=2, label=vr)
        axes[1].loglog(g["n"], g["se"], "o-", label=vr)
        axes[2].loglog(g["runtime"], g["se"], "o-", label=vr)
    axes[0].axhline(bs, color="k", ls="--", label="Black-Scholes")
    axes[0].set(xscale="log", title="Price estimate and 95% CI vs number of simulations", xlabel="N", ylabel="price")
    ref = df[df.vr == "none"].iloc[0]
    axes[1].loglog(Ns, ref["se"] * np.sqrt(Ns[0] / np.array(Ns)), "k:", label="slope -1/2")
    axes[1].set(title="Standard error vs N (log-log)", xlabel="N", ylabel="SE")
    axes[2].set(title="Error vs runtime", xlabel="seconds", ylabel="SE")
    for a in axes:
        a.legend(fontsize=7)
    save(fig, "02_convergence_and_variance_reduction.png")
    # running mean for 'expected value against sample' chart
    rng = np.random.default_rng(5)
    n_steps, dt, _ = step_rule(90)
    disc = np.exp(-R * 90 / 252)
    fn = lambda P: (disc * np.maximum(P[:, -1] - K, 0), None)
    x_plain, _ = _samples(fn, S0, R, SIG, dt, n_steps, 400_000, np.random.default_rng(5), False)
    x_anti, _ = _samples(fn, S0, R, SIG, dt, n_steps, 400_000, np.random.default_rng(5), True)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for x, lab, ns in [(x_plain, "plain", np.arange(1, len(x_plain) + 1)), (x_anti, "antithetic (pairs)", 2 * np.arange(1, len(x_anti) + 1))]:
        rm = np.cumsum(x) / np.arange(1, len(x) + 1)
        ax.plot(ns[100:], rm[100:], label=lab, lw=1)
    ax.axhline(bs, color="k", ls="--", label="Black-Scholes")
    ax.set(xscale="log", xlabel="simulated paths", ylabel="running mean of discounted payoff", ylim=(bs - .25, bs + .25),
           title="Convergence of the running mean (European call, 90d)")
    ax.legend()
    save(fig, "03_running_mean_convergence.png")
    # variance-reduction efficiency table for several styles
    rows = []
    cases = {"european call": dict(option_style="european"), "asian call (arithmetic)": dict(option_style="asian"),
             "up-and-out call (H=115)": dict(option_style="barrier", barrier=115, barrier_type="up-and-out"),
             "american put": dict(option_style="american", option_type="put", n_simulations=50_000)}
    for name, kw in cases.items():
        base = None
        for vr in ("none", "antithetic", "control", "both"):
            if name.startswith("american") and vr in ("control", "both"):
                continue
            if name.startswith("up-and-out") and vr in ("control", "both"):
                continue
            kk = dict(S0=S0, K=K, expiry_days=90, r=R, sigma=SIG, option_type="call", option_style="european", n_simulations=100_000)
            kk.update(kw)
            r = price_option(**kk, seed=33, variance_reduction=vr)
            eff = 1 / (r.std_err**2 * r.runtime_s)
            base = base or eff
            rows.append(dict(case=name, method=vr, price=r.price, se=r.std_err, runtime_s=r.runtime_s, efficiency_vs_plain=eff / base))
    eff_df = pd.DataFrame(rows)
    eff_df["se_reduction_x"] = eff_df.groupby("case")["se"].transform(lambda s: s.iloc[0] / s)
    eff_df.to_csv(TAB / "03_variance_reduction_efficiency.csv", index=False)
    SUMMARY["variance_reduction"] = eff_df.set_index(["case", "method"])[["se_reduction_x", "efficiency_vs_plain"]].round(2).reset_index().to_dict("records")
    slope = np.polyfit(np.log(df[df.vr == "none"]["n"]), np.log(df[df.vr == "none"]["se"]), 1)[0]
    SUMMARY["se_loglog_slope_plain"] = float(slope)


# 3 ---------------------------------------------------------------------------------------------------------------------
def asian():
    rows = []
    for sig in (0.1, 0.2, 0.3, 0.4):
        eu = bs_price(S0, K, yrs(90), R, sig, "call")
        a = price_option(S0, K, 90, R, sig, "call", "asian", 200_000, seed=3, variance_reduction="control")
        g = price_option(S0, K, 90, R, sig, "call", "asian", 200_000, seed=3, asian_average="geometric", variance_reduction="antithetic")
        rows.append(dict(sigma=sig, european=eu, asian_arithmetic=a.price, se_arith=a.std_err, asian_geometric=g.price,
                         geometric_closed_form=geometric_asian_price(S0, K, yrs(90), R, sig, 90), asian_over_european=a.price / eu))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "04_asian_vs_european.csv", index=False)
    SUMMARY["asian"] = df.round(4).to_dict("records")
    sigs = np.linspace(0.05, 0.5, 10)
    kks = np.arange(80, 121, 5)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for lab, f in [("European", lambda s: bs_price(S0, K, yrs(90), R, s, "call")),
                   ("Asian (arith, MC+CV)", lambda s: price_option(S0, K, 90, R, s, "call", "asian", 60_000, seed=1, variance_reduction="control").price)]:
        axes[0].plot(sigs, [f(s) for s in sigs], "o-", label=lab)
    axes[0].set(title="Price vs volatility", xlabel="sigma", ylabel="call price")
    axes[0].legend()
    for lab, f in [("European", lambda k: bs_price(S0, k, yrs(90), R, SIG, "call")),
                   ("Asian", lambda k: price_option(S0, k, 90, R, SIG, "call", "asian", 60_000, seed=1, variance_reduction="control").price)]:
        axes[1].plot(kks, [f(k) for k in kks], "o-", label=lab)
    axes[1].set(title="Price vs strike", xlabel="K")
    axes[1].legend()
    for lab, f in [("European", lambda t: bs_price(S0, K, yrs(t), R, SIG, "call")),
                   ("Asian", lambda t: price_option(S0, K, t, R, SIG, "call", "asian", 60_000, seed=1, variance_reduction="control").price)]:
        axes[2].plot(range(20, 260, 30), [f(t) for t in range(20, 260, 30)], "o-", label=lab)
    axes[2].set(title="Price vs maturity", xlabel="trading days")
    axes[2].legend()
    save(fig, "04_asian_vs_european.png")


# 4 ---------------------------------------------------------------------------------------------------------------------
def barrier():
    rows = []
    for ot in ("call", "put"):
        for bt, H in [("up-and-in", 115), ("up-and-out", 115), ("down-and-in", 85), ("down-and-out", 85)]:
            r = price_option(S0, K, 90, R, SIG, ot, "barrier", 400_000, H, bt, seed=17, variance_reduction="antithetic")
            c = barrier_price_continuous(S0, K, H, yrs(90), R, SIG, ot, bt)
            b = barrier_price_bgk(S0, K, H, yrs(90), R, SIG, ot, bt, 1 / 252)
            rows.append(dict(type=ot, barrier_type=bt, H=H, mc_discrete=r.price, se=r.std_err, closed_form_continuous=c,
                             closed_form_bgk=b, vanilla=bs_price(S0, K, yrs(90), R, SIG, ot), z_vs_bgk=(r.price - b) / r.std_err))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "05_barrier_all_types.csv", index=False)
    SUMMARY["barrier_types"] = dict(max_abs_gap_mc_bgk=float((df["mc_discrete"] - df["closed_form_bgk"]).abs().max()),
                                    max_abs_gap_mc_continuous=float((df["mc_discrete"] - df["closed_form_continuous"]).abs().max()))
    # barrier level sweep
    Hs_up, Hs_dn = np.arange(102, 141, 4), np.arange(60, 99, 4)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    van = bs_price(S0, K, yrs(90), R, SIG, "call")
    for bt, Hs, ax in [("up", Hs_up, axes[0]), ("down", Hs_dn, axes[1])]:
        io = [price_option(S0, K, 90, R, SIG, "call", "barrier", 80_000, H, f"{bt}-and-in", seed=2).price for H in Hs]
        oo = [price_option(S0, K, 90, R, SIG, "call", "barrier", 80_000, H, f"{bt}-and-out", seed=2).price for H in Hs]
        ax.plot(Hs, io, "o-", label="knock-in")
        ax.plot(Hs, oo, "s-", label="knock-out")
        ax.plot(Hs, np.array(io) + np.array(oo), "k--", label="in + out")
        ax.axhline(van, color="grey", lw=.8, label="vanilla")
        ax.set(title=f"{bt}-barrier call price vs barrier level", xlabel="barrier H")
        ax.legend(fontsize=7)
    save(fig, "05_barrier_price_vs_level.png")
    # monitoring-frequency study
    rows = []
    for sd in (4.0, 2.0, 1.0, 0.5, 0.25):
        r = price_option(S0, K, 20, R, SIG, "call", "barrier", 400_000, 108, "up-and-out", seed=5, variance_reduction="antithetic", step_days=sd)
        rows.append(dict(step_days=sd, mc=r.price, se=r.std_err, continuous=barrier_price_continuous(S0, K, 108, yrs(20), R, SIG, "call", "up-and-out"),
                         bgk=barrier_price_bgk(S0, K, 108, yrs(20), R, SIG, "call", "up-and-out", sd / 252)))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "06_barrier_monitoring_frequency.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(df["step_days"], df["mc"], yerr=1.96 * df["se"], fmt="o-", label="MC (discrete monitoring)")
    ax.plot(df["step_days"], df["bgk"], "s--", label="BGK-corrected closed form")
    ax.axhline(df["continuous"].iloc[0], color="k", ls=":", label="continuous monitoring")
    ax.set(xscale="log", title="Up-and-out call (20d, H=108): price vs monitoring step", xlabel="step (trading days)", ylabel="price")
    ax.legend()
    save(fig, "06_barrier_monitoring_frequency.png")
    SUMMARY["monitoring"] = df.round(4).to_dict("records")


# 5 ---------------------------------------------------------------------------------------------------------------------
def time_step_rule():
    rows = []
    for T in (15, 20, 25, 29, 30, 31, 35, 40):
        for sd, lab in [(None, "rule"), (1.0, "1-day steps"), (0.5, "0.5-day steps")]:
            a = price_option(S0, K, T, R, SIG, "call", "asian", 200_000, seed=3, variance_reduction="control", step_days=sd)
            b = price_option(S0, K, T, R, SIG, "call", "barrier", 300_000, 108, "up-and-out", seed=3, variance_reduction="antithetic", step_days=sd)
            rows.append(dict(expiry_days=T, steps=lab, step_used=a.step_days, asian=a.price, asian_se=a.std_err, uo_call=b.price, uo_se=b.std_err))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "07_time_step_rule.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    for lab, g in df.groupby("steps"):
        axes[0].plot(g["expiry_days"], g["asian"], "o-", label=lab)
        axes[1].plot(g["expiry_days"], g["uo_call"], "o-", label=lab)
    for ax, t in zip(axes, ["Asian call", "Up-and-out call (H=108)"]):
        ax.axvline(30, color="grey", ls=":")
        ax.set(title=f"{t}: effect of the step size around the 30-day switch", xlabel="expiry (trading days)")
        ax.legend(fontsize=7)
    save(fig, "07_time_step_rule.png")


# 6 ---------------------------------------------------------------------------------------------------------------------
def american():
    rows = []
    for T in (90, 252):
        for s0 in (80, 90, 100, 110, 120):
            r = price_option(s0, K, T, R, SIG, "put", "american", 100_000, seed=1, variance_reduction="antithetic")
            tree = binomial_price(s0, K, yrs(T), R, SIG, "put", True, 2000)
            eu = bs_price(s0, K, yrs(T), R, SIG, "put")
            rows.append(dict(expiry_days=T, S0=s0, lsm=r.price, se=r.std_err, binomial=tree, european=eu, intrinsic=max(K - s0, 0),
                             early_exercise_premium=tree - eu, lsm_minus_tree=r.price - tree, in_sample_lsm=r.extra["in_sample_price"]))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "08_american_put.csv", index=False)
    SUMMARY["american"] = dict(max_abs_gap_vs_tree=float(df["lsm_minus_tree"].abs().max()), mean_gap=float(df["lsm_minus_tree"].mean()),
                               max_premium=float(df["early_exercise_premium"].max()))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    s_grid = np.arange(70, 131, 5)
    for T, c in [(90, "C0"), (252, "C1")]:
        axes[0].plot(s_grid, [binomial_price(s, K, yrs(T), R, SIG, "put", True, 1000) - bs_price(s, K, yrs(T), R, SIG, "put") for s in s_grid],
                     color=c, label=f"{T}d")
    axes[0].set(title="Early-exercise premium of an American put (tree - Black-Scholes)", xlabel="S0", ylabel="premium")
    axes[0].legend()
    for sig, c in [(0.15, "C2"), (0.25, "C3"), (0.40, "C4")]:
        axes[1].plot([0.0, 0.02, 0.05, 0.08, 0.10],
                     [binomial_price(100, K, yrs(252), rr, sig, "put", True, 800) - bs_price(100, K, yrs(252), rr, sig, "put") for rr in [0.0, 0.02, 0.05, 0.08, 0.10]],
                     "o-", color=c, label=f"sigma={sig}")
    axes[1].set(title="Premium vs interest rate (ATM, 1y)", xlabel="r")
    axes[1].legend()
    # LSM basis degree and in-sample vs two-pass: averaged over 12 seeds (a single seed is mostly noise)
    rows2 = []
    tree = binomial_price(100, 110, yrs(252), R, SIG, "put", True, 2000)
    for deg in (1, 2, 3, 4, 5):
        for tp in (True, False):
            e = [price_option(100, 110, 252, R, SIG, "put", "american", 20_000, seed=700 + sd, lsm_degree=deg, lsm_two_pass=tp).price - tree for sd in range(12)]
            rows2.append(dict(degree=deg, estimator="two-pass (independent paths)" if tp else "in-sample", mean_error=float(np.mean(e)),
                              se_of_mean=float(np.std(e, ddof=1) / np.sqrt(len(e))), tree=tree))
    d2 = pd.DataFrame(rows2)
    d2.to_csv(TAB / "09_lsm_basis_and_bias.csv", index=False)
    for est, g in d2.groupby("estimator"):
        axes[2].errorbar(g["degree"], g["mean_error"], yerr=g["se_of_mean"], fmt="o-", capsize=3, label=est)
    axes[2].axhline(0, color="k", lw=.8)
    axes[2].set(title="LSM bias vs binomial (put K=110, 1y, 20k paths, 12 seeds)", xlabel="polynomial degree", ylabel="mean(LSM - tree)")
    axes[2].legend(fontsize=7)
    save(fig, "08_american_put_lsm.png")
    c = price_option(S0, K, 252, R, SIG, "call", "american", 100_000, seed=2)
    SUMMARY["american_call_vs_european"] = dict(lsm=c.price, se=c.std_err, bs=bs_price(S0, K, 1.0, R, SIG, "call"))


# 7 ---------------------------------------------------------------------------------------------------------------------
def greeks():
    rows = []
    for ot in ("call", "put"):
        g = price_option(S0, K, 90, R, SIG, ot, "european", 400_000, seed=5, compute_greeks=True, variance_reduction="antithetic").greeks
        b = bs_greeks(S0, K, yrs(90), R, SIG, ot)
        for k in g:
            rows.append(dict(type=ot, greek=k, mc=g[k], black_scholes=b[k], abs_error=g[k] - b[k], rel_error=(g[k] - b[k]) / abs(b[k])))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "10_greeks_european.csv", index=False)
    SUMMARY["greeks_max_rel_err"] = float(df["rel_error"].abs().max())
    # bump size and common random numbers
    rows = []
    bs_d = bs_greeks(S0, K, yrs(90), R, SIG, "call")["delta"]
    for rel in (0.0005, 0.002, 0.01, 0.05):
        for crn in (True, False):
            errs = []
            for sd in range(12):
                g = finite_difference_greeks(S0, K, 90, R, SIG, "call", "european", 50_000, None, None, "arithmetic", 100 + sd, "none", None, 3, True,
                                             rel_bump=rel, common_random_numbers=crn)
                errs.append(g["delta"] - bs_d)
            rows.append(dict(rel_bump=rel, common_random_numbers=crn, rmse=float(np.sqrt(np.mean(np.square(errs)))), bias=float(np.mean(errs))))
    d2 = pd.DataFrame(rows)
    d2.to_csv(TAB / "11_delta_bump_study.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for crn, g in d2.groupby("common_random_numbers"):
        ax.loglog(g["rel_bump"], g["rmse"], "o-", label="common random numbers" if crn else "independent seeds")
    ax.set(title="Delta RMSE vs bump size (50k paths)", xlabel="bump / S0", ylabel="RMSE of delta")
    ax.legend()
    save(fig, "09_delta_bump_and_crn.png")
    # delta of each style vs spot
    s_grid = np.arange(80, 121, 5)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    specs = {"european call": dict(option_style="european"), "asian call": dict(option_style="asian"),
             "up-and-out call H=125": dict(option_style="barrier", barrier=125, barrier_type="up-and-out"),
             "american put": dict(option_style="american", option_type="put")}
    for lab, kw in specs.items():
        ot = kw.get("option_type", "call")
        d = []
        for s0 in s_grid:
            g = finite_difference_greeks(s0, K, 90, R, SIG, ot, kw["option_style"], 40_000, kw.get("barrier"), kw.get("barrier_type"), "arithmetic", 7,
                                         "antithetic", None, 3, True)
            d.append(g["delta"])
        ax.plot(s_grid, d, "o-", label=lab)
    ax.set(title="Monte Carlo delta by option style (common random numbers)", xlabel="S0", ylabel="delta")
    ax.legend(fontsize=8)
    save(fig, "10_delta_by_style.png")


# 8 ---------------------------------------------------------------------------------------------------------------------
def runtime_and_plots():
    rows = []
    for style, kw in [("european", {}), ("asian", {}), ("barrier", dict(barrier=115, barrier_type="up-and-out")), ("american", dict(option_type="put"))]:
        for n in (10_000, 50_000, 100_000, 200_000):
            for T in (10, 90, 252):
                if style == "american" and n > 100_000:
                    continue
                r = price_option(S0, K, T, R, SIG, kw.pop("option_type", "call") if False else kw.get("option_type", "call"), style, n, kw.get("barrier"),
                                 kw.get("barrier_type"), seed=1)
                rows.append(dict(style=style, n=n, expiry_days=T, steps=r.n_steps, runtime_s=r.runtime_s))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "12_runtime.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    sub = df[df["expiry_days"] == 90]
    for st, g in sub.groupby("style"):
        axes[0].loglog(g["n"], g["runtime_s"], "o-", label=st)
    axes[0].set(title="Runtime vs simulations (90d)", xlabel="N", ylabel="seconds")
    axes[0].legend()
    sub = df[df["n"] == 100_000]
    for st, g in sub.groupby("style"):
        axes[1].loglog(g["steps"], g["runtime_s"], "o-", label=st)
    axes[1].set(title="Runtime vs number of time steps (100k paths)", xlabel="steps", ylabel="seconds")
    axes[1].legend()
    save(fig, "11_runtime_scaling.png")
    # sample paths and payoff distributions
    rng = np.random.default_rng(3)
    n_steps, dt, _ = step_rule(60)
    P = simulate(S0, R, SIG, dt, n_steps, 25, rng)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for p in P:
        hit = p.max() >= 115
        ax.plot(np.arange(n_steps + 1), p, color="C3" if hit else "C0", lw=.9, alpha=.8)
    ax.axhline(115, color="k", ls="--", label="up barrier 115 (red paths knock out)")
    ax.axhline(K, color="grey", ls=":")
    ax.set(title="Simulated risk-neutral GBM paths (60 days, 1-day steps)", xlabel="step", ylabel="S")
    ax.legend()
    save(fig, "12_sample_paths.png")
    P = simulate(S0, R, SIG, dt, n_steps, 200_000, rng)
    disc = np.exp(-R * 60 / 252)
    pay = {"European call": np.maximum(P[:, -1] - K, 0), "Asian call": np.maximum(P[:, 1:].mean(axis=1) - K, 0),
           "Up-and-out call": np.maximum(P[:, -1] - K, 0) * (P[:, 1:].max(axis=1) < 115),
           "Up-and-in call": np.maximum(P[:, -1] - K, 0) * (P[:, 1:].max(axis=1) >= 115)}
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=False)
    for ax, (k, v) in zip(axes, pay.items()):
        ax.hist(disc * v, bins=60, color="C0")
        ax.set(title=f"{k}\n(P(payoff=0) = {np.mean(v == 0):.2f})", xlabel="discounted payoff")
        ax.set_yscale("log")
    save(fig, "13_payoff_distributions.png")
    SUMMARY["p_zero_payoff"] = {k: float(np.mean(v == 0)) for k, v in pay.items()}


if __name__ == "__main__":
    t0 = time.time()
    ALL = dict(validation=validation, convergence=convergence, asian=asian, barrier=barrier, time_step_rule=time_step_rule,
               american=american, greeks=greeks, runtime_and_plots=runtime_and_plots)
    chosen = [ALL[a] for a in sys.argv[1:]] if len(sys.argv) > 1 else list(ALL.values())
    if len(sys.argv) > 1 and Path("outputs/summary.json").exists():
        SUMMARY.update(json.loads(Path("outputs/summary.json").read_text()))
    for fn in chosen:
        print(fn.__name__, flush=True)
        fn()
        print(f"  done at {time.time() - t0:.0f}s", flush=True)
    Path("outputs/summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))
    print(json.dumps({k: v for k, v in SUMMARY.items() if k not in ("asian", "variance_reduction", "monitoring")}, indent=2, default=str))
