"""
Combine top SLEEP_POD candidates and simulate them together with proper
leg aggregation + LIMIT clamping (matches FINAL_GLAUCO multi-pair behavior).
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Data_ROUND_5"
SLEEP_POD = [
    "SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE",
]
WARMUP = 500
EXIT_Z = 0.3
LIMIT = 10
SIZE = 10


def load_mids():
    frames = []
    for day in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{day}.csv", sep=";")
        df = df[df["product"].isin(SLEEP_POD)][["timestamp", "product", "mid_price"]]
        df["t"] = day * 1_000_000 + df["timestamp"]
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index="t", columns="product", values="mid_price").sort_index().dropna()
    days = pd.Series((wide.index // 1_000_000).astype(int), index=wide.index)
    return wide, days


def run_combo(mids, days, pairs, label):
    """
    pairs: list of (a, b, form, entry_z)
    Simulates all pairs running together. Each pair has its own running stats
    and pos target. Leg targets aggregate across pairs, then clamped to LIMIT.
    Realized PnL via avg-cost basis on closes/flips, MTM at end.
    """
    n = len(mids)
    days_v = days.values
    vals = {s: mids[s].values for s in SLEEP_POD}

    pair_state = [{"n": 0, "mean": 0.0, "M2": 0.0, "pos": 0} for _ in pairs]
    leg_pos = {s: 0 for s in SLEEP_POD}
    leg_avg = {s: 0.0 for s in SLEEP_POD}
    by_day = {2: 0.0, 3: 0.0, 4: 0.0}
    leg_realized = {s: 0.0 for s in SLEEP_POD}

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
        leg_targets = {s: 0 for s in SLEEP_POD}
        for pi, (a, b, form, ez) in enumerate(pairs):
            sig_v = vals[a][i] - vals[b][i] if form == "spread" else vals[a][i] + vals[b][i]
            st = pair_state[pi]
            st["n"] += 1
            d = sig_v - st["mean"]
            st["mean"] += d / st["n"]
            st["M2"] += d * (sig_v - st["mean"])
            cur = st["pos"]
            if st["n"] >= WARMUP:
                var = st["M2"] / (st["n"] - 1) if st["n"] > 1 else 0.0
                sd = math.sqrt(var)
                if sd > 1e-9:
                    z = (sig_v - st["mean"]) / sd
                    if cur == 0:
                        if z > ez:
                            cur = -1
                        elif z < -ez:
                            cur = +1
                    else:
                        if abs(z) < EXIT_Z:
                            cur = 0
            st["pos"] = cur
            ta = SIZE * cur
            tb = (-SIZE if form == "spread" else SIZE) * cur
            leg_targets[a] += ta
            leg_targets[b] += tb

        day = int(days_v[i])
        for s, tgt in leg_targets.items():
            tgt = max(-LIMIT, min(LIMIT, tgt))
            apply_leg(s, tgt, vals[s][i], day)

    last_day = int(days_v[-1])
    for s in SLEEP_POD:
        if leg_pos[s] != 0:
            mtm = (vals[s][-1] - leg_avg[s]) * leg_pos[s]
            leg_realized[s] += mtm
            by_day[last_day] += mtm

    total = sum(leg_realized.values())
    print(f"\n=== {label} ===")
    print(f"  Total: {total:+.0f}  | d2: {by_day[2]:+.0f}  d3: {by_day[3]:+.0f}  d4: {by_day[4]:+.0f}")
    for s in SLEEP_POD:
        if abs(leg_realized[s]) > 1:
            print(f"    {s}: {leg_realized[s]:+.0f}")
    return total, by_day


def main():
    mids, days = load_mids()
    print(f"Loaded {len(mids)} ticks")

    # Top single-pair winners (in PRODUCTION simulator):
    #   COTTON-POLYESTER spread @ z=1.0 -> +49150
    #   POLYESTER-SUEDE spread  @ z=1.8 -> +35540
    #   COTTON-SUEDE spread     @ z=1.0 -> +25960  (or @1.8 -> +24720)
    #   LAMB_WOOL+SUEDE sum     @ z=2.0 -> +26745  (positive on 2 of 3 days)

    # Baseline (current production)
    run_combo(mids, days, [
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.8),
    ], "BASELINE: 1 pair (POLY-COTTON spread, z=1.8)")

    # Just lower z on baseline
    run_combo(mids, days, [
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.0),
    ], "BASELINE-TUNED: same pair, z=1.0")

    # Add POLY-SUEDE
    run_combo(mids, days, [
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.8),
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE", "spread", 1.8),
    ], "2 pairs: POLY-COTTON z=1.8 + POLY-SUEDE z=1.8")

    # POLY-COTTON z=1.0 + POLY-SUEDE z=1.8
    run_combo(mids, days, [
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.0),
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE", "spread", 1.8),
    ], "2 pairs (TUNED): POLY-COTTON z=1.0 + POLY-SUEDE z=1.8")

    # 3 pairs all using POLYESTER as hub
    run_combo(mids, days, [
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.0),
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE", "spread", 1.8),
        ("SLEEP_POD_COTTON", "SLEEP_POD_SUEDE", "spread", 1.0),
    ], "3 pairs: POLY-COTTON + POLY-SUEDE + COTTON-SUEDE (all z tuned)")

    # 3 pairs without POLY-COTTON conflict
    run_combo(mids, days, [
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE", "spread", 1.8),
        ("SLEEP_POD_COTTON", "SLEEP_POD_SUEDE", "spread", 1.8),
        ("SLEEP_POD_LAMB_WOOL", "SLEEP_POD_SUEDE", "sum", 2.0),
    ], "3 pairs SUEDE-hub: POLY-SUEDE + COTTON-SUEDE + LAMB+SUEDE")

    # Add 4th pair: COTTON-NYLON (only fully robust on all 3 days)
    run_combo(mids, days, [
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.0),
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE", "spread", 1.8),
        ("SLEEP_POD_COTTON", "SLEEP_POD_SUEDE", "spread", 1.0),
        ("SLEEP_POD_COTTON", "SLEEP_POD_NYLON", "spread", 1.0),
    ], "4 pairs: + COTTON-NYLON")


if __name__ == "__main__":
    main()
