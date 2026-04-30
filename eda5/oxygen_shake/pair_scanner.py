"""
OXYGEN_SHAKE pair scanner — same simulator used for SLEEP_POD pair_scanner_v2,
ported to OXYGEN_SHAKE. Tests every (a, b, form, entry_z) and reports PnL on
days 2/3/4 with the same execution model FINAL_GLAUCO uses.

Forms tested: spread, sum, ols_spread.
"""
from __future__ import annotations

import itertools
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
ENTRY_Z_GRID = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
FORMS = ["spread", "sum", "ols_spread"]


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


def ols_beta(y: pd.Series, x: pd.Series) -> float:
    yv = y.values - y.mean()
    xv = x.values - x.mean()
    denom = (xv * xv).sum()
    return float((xv * yv).sum() / denom) if denom > 0 else 0.0


def simulate_pair(mids, days, a, b, form, entry_z, beta=1.0):
    if form == "spread":
        sig = mids[a] - mids[b]
        leg_b_mult = -1.0
    elif form == "sum":
        sig = mids[a] + mids[b]
        leg_b_mult = +1.0
    elif form == "ols_spread":
        sig = mids[a] - beta * mids[b]
        leg_b_mult = -beta
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
    n_trades = 0
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
        if running_n >= WARMUP:
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
                    if abs(z) < EXIT_Z:
                        cur = 0
                if cur != pos_target:
                    n_trades += 1
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

    return leg_realized[a] + leg_realized[b], n_trades, by_day


def main():
    print("Loading mids...")
    mids, days = load_mids()
    print(f"  ticks={len(mids)}  days={sorted(days.unique())}")

    rows = []
    pairs = list(itertools.combinations(OXY, 2))
    print(f"Scanning {len(pairs)} pairs x {len(FORMS)} forms x {len(ENTRY_Z_GRID)} entry_z")
    for a, b in pairs:
        beta_ab = ols_beta(mids[a], mids[b])
        for form in FORMS:
            beta = beta_ab if form == "ols_spread" else 1.0
            for ez in ENTRY_Z_GRID:
                total, n_tr, by_day = simulate_pair(mids, days, a, b, form, ez, beta)
                pos_days = sum(1 for v in by_day.values() if v > 0)
                rows.append({
                    "a": a, "b": b, "form": form, "beta": round(beta, 3),
                    "entry_z": ez, "n_trades": n_tr, "total_pnl": round(total, 1),
                    "pnl_d2": round(by_day[2], 1),
                    "pnl_d3": round(by_day[3], 1),
                    "pnl_d4": round(by_day[4], 1),
                    "pos_days": pos_days,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "pair_scan_results.csv", index=False)
    cols = ["a", "b", "form", "beta", "entry_z", "n_trades",
            "total_pnl", "pnl_d2", "pnl_d3", "pnl_d4", "pos_days"]

    print("\n=== TOP 20 by total PnL ===")
    print(df.sort_values("total_pnl", ascending=False).head(20)[cols].to_string(index=False))

    print("\n=== Robust: pos_days >= 2, top 15 ===")
    rb = df[df["pos_days"] >= 2].sort_values("total_pnl", ascending=False).head(15)
    print(rb[cols].to_string(index=False))

    print("\n=== Fully robust: pos_days == 3, top 15 ===")
    full = df[df["pos_days"] == 3].sort_values("total_pnl", ascending=False).head(15)
    print(full[cols].to_string(index=False))

    print(f"\nSaved {len(df)} combos to {OUT_DIR/'pair_scan_results.csv'}")


if __name__ == "__main__":
    main()
