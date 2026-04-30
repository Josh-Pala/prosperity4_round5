"""
Check pairwise correlation of the top SLEEP_POD pair signals to ensure
they're not all the same trade in disguise. Also test combined PnL when
all are run together (with position-limit aware leg targets).
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
    return long.pivot_table(index="t", columns="product", values="mid_price").sort_index().dropna()


def signal(mids, a, b, form):
    return mids[a] - mids[b] if form == "spread" else mids[a] + mids[b]


def run_pairs(mids, pairs):
    """Simulate a SET of pairs simultaneously, mimicking FINAL_GLAUCO leg
    aggregation + position limit clamping. Returns (total_pnl, per_leg_pnl)."""
    # Per-pair running stats
    pair_state = {i: {"n": 0, "mean": 0.0, "M2": 0.0, "pos_target": 0}
                  for i in range(len(pairs))}
    leg_pos = {s: 0 for s in SLEEP_POD}
    leg_realized_pnl = {s: 0.0 for s in SLEEP_POD}
    last_leg_avg_price = {s: 0.0 for s in SLEEP_POD}

    n = len(mids)
    vals = {s: mids[s].values for s in SLEEP_POD}

    for i in range(n):
        # Update each pair stat & decide pair target
        leg_targets = {s: 0 for s in SLEEP_POD}
        for pi, (a, b, form, ez) in enumerate(pairs):
            x = (vals[a][i] - vals[b][i]) if form == "spread" else (vals[a][i] + vals[b][i])
            st = pair_state[pi]
            st["n"] += 1
            d = x - st["mean"]
            st["mean"] += d / st["n"]
            st["M2"] += d * (x - st["mean"])
            cur = st["pos_target"]
            if st["n"] >= WARMUP:
                var = st["M2"] / (st["n"] - 1) if st["n"] > 1 else 0.0
                sd = math.sqrt(var)
                if sd > 1e-9:
                    z = (x - st["mean"]) / sd
                    if cur == 0:
                        if z > ez:
                            cur = -1
                        elif z < -ez:
                            cur = +1
                    else:
                        if abs(z) < EXIT_Z:
                            cur = 0
            st["pos_target"] = cur
            ta = SIZE * cur
            tb = (-SIZE if form == "spread" else SIZE) * cur
            leg_targets[a] += ta
            leg_targets[b] += tb

        # Apply leg targets (clamp to LIMIT) and compute realized PnL via avg cost
        for s, tgt in leg_targets.items():
            tgt = max(-LIMIT, min(LIMIT, tgt))
            current = leg_pos[s]
            d = tgt - current
            if d == 0:
                continue
            price = vals[s][i]
            # Closing or flipping portion
            if (current > 0 and d < 0) or (current < 0 and d > 0):
                # closing some/all
                close_qty = min(abs(d), abs(current))
                # PnL = (price - avg_entry) * sign(current) * close_qty
                sign = 1 if current > 0 else -1
                leg_realized_pnl[s] += sign * (price - last_leg_avg_price[s]) * close_qty
                new_pos = current - sign * close_qty
                # remaining d may open new direction
                remaining = d - (-sign * close_qty)
                if remaining != 0:
                    last_leg_avg_price[s] = price
                    new_pos = new_pos + remaining
                leg_pos[s] = new_pos
            else:
                # adding to existing position — update avg cost
                new_pos = current + d
                if current == 0:
                    last_leg_avg_price[s] = price
                else:
                    last_leg_avg_price[s] = (
                        last_leg_avg_price[s] * abs(current) + price * abs(d)
                    ) / abs(new_pos)
                leg_pos[s] = new_pos

    # Mark-to-market remaining at last price
    for s, pos in leg_pos.items():
        if pos != 0:
            mtm = (vals[s][-1] - last_leg_avg_price[s]) * pos
            leg_realized_pnl[s] += mtm

    total = sum(leg_realized_pnl.values())
    return total, leg_realized_pnl


def main():
    mids = load_mids()
    print(f"Loaded {len(mids)} ticks\n")

    # Top candidates from grid search
    candidates = [
        ("SLEEP_POD_COTTON", "SLEEP_POD_POLYESTER", "sum", 1.5, "TOP1"),
        ("SLEEP_POD_LAMB_WOOL", "SLEEP_POD_SUEDE", "spread", 1.0, "TOP2"),
        ("SLEEP_POD_COTTON", "SLEEP_POD_SUEDE", "sum", 1.0, "TOP3"),
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.8, "BASELINE"),
    ]

    print("=== Signal correlations ===")
    sigs = {label: signal(mids, a, b, form) for a, b, form, ez, label in candidates}
    sigs_df = pd.DataFrame(sigs)
    print(sigs_df.corr().round(3).to_string())

    # Z-score correlations (more meaningful for trade timing)
    print("\n=== Z-score correlations (rolling 2000) ===")
    zs = {}
    for label, s in sigs.items():
        m = s.rolling(2000).mean()
        sd = s.rolling(2000).std()
        zs[label] = ((s - m) / sd).fillna(0)
    print(pd.DataFrame(zs).corr().round(3).to_string())

    print("\n=== Combined sim: top 3 candidates run together ===")
    top3 = [(a, b, form, ez) for a, b, form, ez, _ in candidates[:3]]
    total, by_leg = run_pairs(mids, top3)
    print(f"Total PnL (combined, with LIMIT={LIMIT} and SIZE={SIZE}): {total:+.1f}")
    print("Per-leg breakdown:")
    for s, v in by_leg.items():
        print(f"  {s}: {v:+.1f}")

    print("\n=== Combined sim: baseline only ===")
    base = [(a, b, form, ez) for a, b, form, ez, lbl in candidates if lbl == "BASELINE"]
    total_b, by_leg_b = run_pairs(mids, base)
    print(f"Total PnL (baseline only): {total_b:+.1f}")
    for s, v in by_leg_b.items():
        print(f"  {s}: {v:+.1f}")

    print("\n=== Combined sim: baseline + top 3 (4 pairs total) ===")
    all4 = [(a, b, form, ez) for a, b, form, ez, _ in candidates]
    total_a, by_leg_a = run_pairs(mids, all4)
    print(f"Total PnL (4 pairs combined): {total_a:+.1f}")
    for s, v in by_leg_a.items():
        print(f"  {s}: {v:+.1f}")

    print("\n=== Combined sim: top 3 WITHOUT baseline (drop POLY-COTTON) ===")
    top3_no_base = [(a, b, form, ez) for a, b, form, ez, lbl in candidates
                    if lbl != "BASELINE"]
    total_t3, by_leg_t3 = run_pairs(mids, top3_no_base)
    print(f"Total PnL: {total_t3:+.1f}")


if __name__ == "__main__":
    main()
