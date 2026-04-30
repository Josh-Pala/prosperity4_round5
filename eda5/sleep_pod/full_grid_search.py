"""
Full grid search on SLEEP_POD pair candidates using PRODUCTION logic
(running mean/M2, single warmup, z entry/exit thresholds matching FINAL_GLAUCO).

For each (pair, form, entry_z) combo:
  - Simulate across all 3 days CONCATENATED (mimicking the running stats
    that persist via traderData in production)
  - Report total PnL, n_trades, AND per-day PnL breakdown.
  - Filter: only keep combos where per-day PnL is positive on >=2 of 3 days.
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent

SLEEP_POD = [
    "SLEEP_POD_COTTON",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_SUEDE",
]
WARMUP = 500
EXIT_Z = 0.3
ENTRY_Z_GRID = [1.0, 1.2, 1.5, 1.8, 2.0]
FORMS = ["spread", "sum"]


def load_mids():
    frames = []
    for day in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{day}.csv", sep=";")
        df = df[df["product"].isin(SLEEP_POD)][["timestamp", "product", "mid_price"]]
        df["day"] = day
        df["t"] = day * 1_000_000 + df["timestamp"]
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index="t", columns="product", values="mid_price").sort_index().dropna()
    days = (wide.index // 1_000_000).astype(int)
    return wide, days


def simulate(sig: pd.Series, days: pd.Series, entry_z: float, exit_z: float = EXIT_Z):
    """Running-stats z-score sim. Returns (total_pnl, n_trades, per_day_pnl)."""
    vals = sig.values
    n = len(vals)
    running_n = 0
    mean = 0.0
    M2 = 0.0
    pos = 0
    last_entry = 0.0
    pnls_per_day = {2: 0.0, 3: 0.0, 4: 0.0}
    n_trades = 0
    for i in range(n):
        x = vals[i]
        running_n += 1
        delta = x - mean
        mean += delta / running_n
        M2 += delta * (x - mean)
        if running_n < WARMUP:
            continue
        var = M2 / (running_n - 1) if running_n > 1 else 0.0
        sd = math.sqrt(var)
        if sd <= 1e-9:
            continue
        z = (x - mean) / sd
        new_pos = pos
        if pos == 0:
            if z > entry_z:
                new_pos = -1
            elif z < -entry_z:
                new_pos = +1
        else:
            if abs(z) < exit_z:
                new_pos = 0
        if new_pos != pos:
            if pos == 0:
                last_entry = x
            else:
                trade_pnl = pos * (last_entry - x)
                pnls_per_day[int(days.iloc[i])] += trade_pnl
                n_trades += 1
            pos = new_pos
    total = sum(pnls_per_day.values())
    return total, n_trades, pnls_per_day


def main():
    print("Loading data...")
    mids, days = load_mids()
    days_idx = pd.Series(days, index=mids.index)
    print(f"Loaded {len(mids)} ticks")

    rows = []
    for a, b in itertools.combinations(SLEEP_POD, 2):
        for form in FORMS:
            sig = mids[a] - mids[b] if form == "spread" else mids[a] + mids[b]
            for ez in ENTRY_Z_GRID:
                total, n_tr, by_day = simulate(sig, days_idx, ez)
                pos_days = sum(1 for v in by_day.values() if v > 0)
                rows.append({
                    "a": a, "b": b, "form": form, "entry_z": ez,
                    "n_trades": n_tr, "total_pnl": total,
                    "pnl_d2": by_day[2], "pnl_d3": by_day[3], "pnl_d4": by_day[4],
                    "pos_days": pos_days,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "full_grid_search.csv", index=False)

    print("\n=== TOP 20 by total PnL (any consistency) ===")
    cols = ["a", "b", "form", "entry_z", "n_trades", "total_pnl",
            "pnl_d2", "pnl_d3", "pnl_d4", "pos_days"]
    print(df.sort_values("total_pnl", ascending=False).head(20)[cols].to_string(index=False))

    print("\n=== TOP 15 with consistency (positive on >= 2 of 3 days) ===")
    consistent = df[df["pos_days"] >= 2].sort_values("total_pnl", ascending=False)
    print(consistent.head(15)[cols].to_string(index=False))

    print("\n=== TOP 10 with FULL consistency (positive on all 3 days) ===")
    full = df[df["pos_days"] == 3].sort_values("total_pnl", ascending=False)
    print(full.head(10)[cols].to_string(index=False))

    print(f"\nSaved {len(df)} combos to {OUT_DIR/'full_grid_search.csv'}")


if __name__ == "__main__":
    main()
