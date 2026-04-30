"""ROBOT family — feasibility analysis for 4 proposed optimizations.

H1. EWMA vs Welford: regime detection — does the spread mean drift over time?
H2. Hybrid Maker-Taker: implied fair value from regression — useful?
H3. Dynamic Limit Allocation: signal strength varies — concurrent pair contention?
H4. Volatility-Adjusted Z: short-window vol changes — does it predict trade quality?
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/robot")
ROBOTS = ["ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY", "ROBOT_MOPPING", "ROBOT_VACUUMING"]
DAYS = [2, 3, 4]

# H25 pairs from BEST.py
PAIRS = [
    ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
    ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
    ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
    ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
    ("ROBOT_MOPPING", "ROBOT_VACUUMING", "sum", 1.2),
]


def load_mids() -> pd.DataFrame:
    parts = []
    for d in DAYS:
        p = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
        p = p[p["product"].isin(ROBOTS)][["day", "timestamp", "product", "mid_price"]]
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    df["t_global"] = df["day"] * 1_000_000 + df["timestamp"]
    return df.pivot_table(index="t_global", columns="product", values="mid_price").sort_index()


def signal(mid, a, b, sign):
    return mid[a] - mid[b] if sign == "spread" else mid[a] + mid[b]


def h1_regime_drift(mid: pd.DataFrame):
    """Compute Welford global mean vs rolling mean (window=2000) of each signal.
    If they differ a lot, EWMA helps."""
    print("="*70)
    print("H1 — Regime drift (EWMA vs static mean)")
    print("="*70)
    rows = []
    for a, b, sign, _ in PAIRS:
        sig = signal(mid, a, b, sign).dropna()
        global_mean = sig.mean()
        global_std = sig.std()
        # Rolling 2000 mean
        rmean = sig.rolling(2000, min_periods=500).mean()
        rstd = sig.rolling(2000, min_periods=500).std()
        # How much does rolling mean deviate from global mean?
        drift_max = (rmean - global_mean).abs().max() / global_std if global_std else 0
        drift_avg = (rmean - global_mean).abs().mean() / global_std if global_std else 0
        # Rolling std variation (volatility regime)
        vol_ratio = rstd.max() / rstd.min() if rstd.min() and not np.isnan(rstd.min()) else 1
        rows.append({
            "pair": f"{a}|{b}|{sign}",
            "global_mean": global_mean,
            "global_std": global_std,
            "drift_max_z": drift_max,
            "drift_avg_z": drift_avg,
            "vol_ratio": vol_ratio,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nInterpretation: drift_max_z >> 1 means EWMA helps (mean is non-stationary)")
    print("vol_ratio >> 2 means H4 (vol-adjusted z) might help")
    df.to_csv(OUT / "h1_h4_regime.csv", index=False)
    return df


def h2_fair_value(mid: pd.DataFrame):
    """For each ROBOT symbol, fit regression sym ~ others. Compute resid std.
    If resid std small, MM at fair value is feasible."""
    print("\n"+"="*70)
    print("H2 — Hybrid Maker-Taker: implied fair value")
    print("="*70)
    M = mid.dropna()
    rows = []
    for target in ROBOTS:
        X = M.drop(columns=[target]).values
        y = M[target].values
        X_ = np.c_[np.ones(len(X)), X]
        beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
        yhat = X_ @ beta
        resid = y - yhat
        ss_res = (resid**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = 1 - ss_res/ss_tot
        # How often is |resid| > 1 (worth a taker trade)?
        edge_freq = (np.abs(resid) > 1).mean()
        edge_freq_2 = (np.abs(resid) > 2).mean()
        rows.append({
            "target": target,
            "r2": r2,
            "resid_std": resid.std(),
            "resid_p95": np.percentile(np.abs(resid), 95),
            "edge_gt_1_freq": edge_freq,
            "edge_gt_2_freq": edge_freq_2,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nFor PEBBLES the equiv fair has resid_std~0 (R²=1.0)")
    print("If r2 > 0.9 and resid_std < 100, fair-value MM viable.")
    df.to_csv(OUT / "h2_fair_value.csv", index=False)
    return df


def h3_pair_concurrency(mid: pd.DataFrame):
    """Simulate trades: how often are multiple pairs active on the same leg
    simultaneously? If always alone, dynamic alloc has no value."""
    print("\n"+"="*70)
    print("H3 — Dynamic Limit Allocation: pair concurrency on shared legs")
    print("="*70)
    # For each pair, compute online (Welford-like) z and detect entry events
    EXIT_Z = 0.3
    WARMUP = 500
    leg_concurrent_active = {s: [] for s in ROBOTS}  # list of n_pairs active per tick

    # Track running stats
    stats = {(a,b,s): {"n":0,"mean":0.0,"M2":0.0} for (a,b,s,_) in PAIRS}
    states = {(a,b,s): 0 for (a,b,s,_) in PAIRS}

    M = mid.dropna()
    leg_active_per_tick = []
    for t, row in M.iterrows():
        active_legs = {s: 0 for s in ROBOTS}
        for a, b, sign, ez in PAIRS:
            sig = row[a] - row[b] if sign == "spread" else row[a] + row[b]
            st = stats[(a,b,sign)]
            st["n"] += 1; n = st["n"]
            delta = sig - st["mean"]
            st["mean"] += delta/n
            st["M2"] += delta * (sig - st["mean"])
            if n < WARMUP: continue
            sd = (st["M2"]/(n-1))**0.5 if n > 1 else 0
            if sd < 1e-9: continue
            z = (sig - st["mean"])/sd
            cur = states[(a,b,sign)]
            if cur == 0:
                if z > ez: cur = -1
                elif z < -ez: cur = +1
            else:
                if abs(z) < EXIT_Z: cur = 0
            states[(a,b,sign)] = cur
            if cur != 0:
                active_legs[a] += 1
                active_legs[b] += 1
        leg_active_per_tick.append(active_legs)

    df_active = pd.DataFrame(leg_active_per_tick)
    print("\nLeg contention statistics (n_pairs simultaneously active per leg):")
    summary = []
    for s in ROBOTS:
        col = df_active[s]
        summary.append({
            "leg": s,
            "frac_idle": (col == 0).mean(),
            "frac_1pair": (col == 1).mean(),
            "frac_2pairs": (col == 2).mean(),
            "frac_3+": (col >= 3).mean(),
            "max": col.max(),
            "mean_when_active": col[col>0].mean() if (col>0).any() else 0,
        })
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    print("\nInterpretation: high frac_2pairs/3+ means LIMIT=10 is contended,")
    print("dynamic allocation by signal strength could matter.")
    df_sum.to_csv(OUT / "h3_concurrency.csv", index=False)
    return df_sum


def h4_vol_regimes(mid: pd.DataFrame):
    """Already covered by H1 vol_ratio. Show short vs long window."""
    print("\n"+"="*70)
    print("H4 — Volatility-Adjusted Z: short-window vol vs entry z")
    print("="*70)
    rows = []
    for a, b, sign, ez in PAIRS:
        sig = signal(mid, a, b, sign).dropna()
        std_short = sig.rolling(500).std()
        std_long = sig.rolling(5000).std()
        ratio = (std_short / std_long).dropna()
        rows.append({
            "pair": f"{a}|{b}|{sign}",
            "vol_ratio_p10": ratio.quantile(0.1),
            "vol_ratio_p50": ratio.quantile(0.5),
            "vol_ratio_p90": ratio.quantile(0.9),
            "vol_ratio_p99": ratio.quantile(0.99),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nIf vol_ratio_p99 / p10 > 3, vol regimes differ enough to matter.")
    df.to_csv(OUT / "h4_vol_regimes.csv", index=False)
    return df


if __name__ == "__main__":
    mid = load_mids()
    print(f"Loaded {len(mid)} ticks across {len(DAYS)} days")
    h1 = h1_regime_drift(mid)
    h2 = h2_fair_value(mid)
    h3 = h3_pair_concurrency(mid)
    h4 = h4_vol_regimes(mid)
