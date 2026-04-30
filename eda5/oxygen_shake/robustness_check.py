"""
OXYGEN_SHAKE robustness check — does the apparent edge survive?

For the top 10 (a, b, form, entry_z) candidates from pair_scan_results.csv:
  1. day-by-day PnL (already in scan)
  2. tick-resampled (subsample 1-of-5 ticks) — checks if PnL depends on
     the specific tick discretisation
  3. shifted-warmup test: discard first 5000 ticks of each day (shifts
     the running mean estimate); profitable strategies should still pay
  4. randomised entry_z perturbation (+/- 0.2): if PnL collapses with a
     small parameter change, the edge is fragile

Outputs:
  - robustness_results.csv
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent

OXY = [
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_MORNING_BREATH",
]
WARMUP = 500
EXIT_Z = 0.3
LIMIT = 10
SIZE = 10


def load_mids():
    frames = []
    for day in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{day}.csv", sep=";")
        df = df[df["product"].isin(OXY)][["timestamp", "product", "mid_price"]]
        df["t"] = day * 1_000_000 + df["timestamp"]
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index="t", columns="product", values="mid_price").sort_index().dropna()
    days = pd.Series((wide.index // 1_000_000).astype(int), index=wide.index)
    return wide, days


def simulate(mids, days, a, b, form, entry_z, beta=1.0, exit_z=EXIT_Z, warmup=WARMUP):
    if form == "spread":
        sig = mids[a] - mids[b]; leg_b_mult = -1.0
    elif form == "sum":
        sig = mids[a] + mids[b]; leg_b_mult = +1.0
    elif form == "ols_spread":
        sig = mids[a] - beta * mids[b]; leg_b_mult = -beta
    else:
        raise ValueError(form)

    vals_a = mids[a].values
    vals_b = mids[b].values
    sig_v = sig.values
    days_v = days.values
    n = len(sig_v)

    running_n = 0
    mean = 0.0
    M2 = 0.0
    pos_target = 0
    leg_pos = {a: 0, b: 0}
    leg_avg = {a: 0.0, b: 0.0}
    leg_realized = {a: 0.0, b: 0.0}
    by_day = {2: 0.0, 3: 0.0, 4: 0.0}

    def apply_leg(sym, qty_target, price, day):
        cur = leg_pos[sym]
        d = qty_target - cur
        if d == 0:
            return
        if (cur > 0 and d < 0) or (cur < 0 and d > 0):
            sign = 1 if cur > 0 else -1
            close_qty = min(abs(d), abs(cur))
            pnl = sign * (price - leg_avg[sym]) * close_qty
            leg_realized[sym] += pnl
            by_day[day] += pnl
            new_pos = cur - sign * close_qty
            remaining = d + sign * close_qty
            if remaining != 0:
                leg_avg[sym] = price
                new_pos += remaining
            leg_pos[sym] = new_pos
        else:
            new_pos = cur + d
            if cur == 0:
                leg_avg[sym] = price
            else:
                leg_avg[sym] = (leg_avg[sym] * abs(cur) + price * abs(d)) / abs(new_pos)
            leg_pos[sym] = new_pos

    for i in range(n):
        x = sig_v[i]
        running_n += 1
        delta = x - mean
        mean += delta / running_n
        M2 += delta * (x - mean)
        if running_n >= warmup:
            var = M2 / (running_n - 1) if running_n > 1 else 0.0
            sd = math.sqrt(var)
            if sd > 1e-9:
                z = (x - mean) / sd
                cur = pos_target
                if cur == 0:
                    if z > entry_z:
                        cur = -1
                    elif z < -entry_z:
                        cur = +1
                else:
                    if abs(z) < exit_z:
                        cur = 0
                pos_target = cur

        ta = SIZE * pos_target
        tb = int(round(leg_b_mult * SIZE)) * pos_target
        ta = max(-LIMIT, min(LIMIT, ta))
        tb = max(-LIMIT, min(LIMIT, tb))
        day = int(days_v[i])
        apply_leg(a, ta, vals_a[i], day)
        apply_leg(b, tb, vals_b[i], day)

    last_day = int(days_v[-1])
    for s, vals in [(a, vals_a), (b, vals_b)]:
        if leg_pos[s] != 0:
            mtm = (vals[-1] - leg_avg[s]) * leg_pos[s]
            leg_realized[s] += mtm
            by_day[last_day] += mtm

    return leg_realized[a] + leg_realized[b], by_day


def ols_beta(y, x):
    yv = y.values - y.mean(); xv = x.values - x.mean()
    denom = (xv * xv).sum()
    return float((xv * yv).sum() / denom) if denom > 0 else 0.0


CANDIDATES = [
    ("OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH", "spread", 1.2),
    ("OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_MINT", "sum", 1.0),
    ("OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH", "spread", 1.0),
    ("OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH", "ols_spread", 1.8),
    ("OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH", "spread", 1.5),
    ("OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_MINT", "sum", 1.2),
]


def main():
    mids, days = load_mids()
    rows = []
    for a, b, form, ez in CANDIDATES:
        beta = ols_beta(mids[a], mids[b]) if form == "ols_spread" else 1.0

        base_total, base_days = simulate(mids, days, a, b, form, ez, beta)

        # Perturbation 1: entry_z +/- 0.2
        plus, _ = simulate(mids, days, a, b, form, ez + 0.2, beta)
        minus, _ = simulate(mids, days, a, b, form, ez - 0.2, beta)

        # Perturbation 2: longer warmup (skip first 2000 ticks)
        warm_total, _ = simulate(mids, days, a, b, form, ez, beta, warmup=2000)

        # Perturbation 3: tighter exit (0.1 vs 0.3)
        tight, _ = simulate(mids, days, a, b, form, ez, beta, exit_z=0.1)

        # Perturbation 4: looser exit (0.5)
        loose, _ = simulate(mids, days, a, b, form, ez, beta, exit_z=0.5)

        # Subsample 1-of-5 ticks
        idx = mids.index[::5]
        sub_mids = mids.loc[idx]
        sub_days = days.loc[idx]
        sub_total, _ = simulate(sub_mids, sub_days, a, b, form, ez, beta)

        rows.append({
            "a": a.replace("OXYGEN_SHAKE_", ""),
            "b": b.replace("OXYGEN_SHAKE_", ""),
            "form": form, "entry_z": ez,
            "base": round(base_total, 0),
            "z_plus_0.2": round(plus, 0),
            "z_minus_0.2": round(minus, 0),
            "warmup_2000": round(warm_total, 0),
            "exit_z_0.1": round(tight, 0),
            "exit_z_0.5": round(loose, 0),
            "subsample_5x": round(sub_total, 0),
            "d2": round(base_days[2], 0),
            "d3": round(base_days[3], 0),
            "d4": round(base_days[4], 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "robustness_results.csv", index=False)
    print("\n=== ROBUSTNESS — top OXY pair candidates ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
