"""
Sanity-check: simulate POLYESTER-COTTON pair with the SAME running-stats
logic that FINAL_GLAUCO uses, on day 2/3/4 data, to see what PnL is expected.

If this matches the production backtest, our scanner setup is consistent.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Data_ROUND_5"
ENTRY_Z = 1.8
EXIT_Z = 0.3
WARMUP = 500


def load_pair(a: str, b: str) -> pd.DataFrame:
    frames = []
    for day in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{day}.csv", sep=";")
        df = df[df["product"].isin([a, b])][["timestamp", "product", "mid_price"]]
        df["t"] = day * 1_000_000 + df["timestamp"]
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index="t", columns="product", values="mid_price").sort_index().dropna()
    return wide


def simulate_running(sig: pd.Series, label: str):
    """Match FINAL_GLAUCO logic exactly: running mean/M2, z entry/exit."""
    vals = sig.values
    n = len(vals)
    running_n = 0
    mean = 0.0
    M2 = 0.0
    pos = 0
    last_entry = 0.0
    pnls = []
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
            if z > ENTRY_Z:
                new_pos = -1
            elif z < -ENTRY_Z:
                new_pos = +1
        else:
            if abs(z) < EXIT_Z:
                new_pos = 0
        if new_pos != pos:
            if pos == 0:
                last_entry = x
            else:
                pnls.append(pos * (last_entry - x))
            pos = new_pos
    arr = np.array(pnls) if pnls else np.array([0.0])
    print(f"{label}: trades={len(pnls)} total_pnl={arr.sum():+.1f} "
          f"mean={arr.mean():+.2f} std={arr.std():.2f} "
          f"final_mean={mean:.2f} final_sd={math.sqrt(M2/(running_n-1)):.2f}")
    return arr.sum(), len(pnls)


def main():
    candidates = [
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread"),  # current production
        ("SLEEP_POD_COTTON", "SLEEP_POD_POLYESTER", "sum"),
        ("SLEEP_POD_LAMB_WOOL", "SLEEP_POD_SUEDE", "sum"),
        ("SLEEP_POD_NYLON", "SLEEP_POD_SUEDE", "spread"),
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE", "sum"),
    ]
    print(f"\n{'='*60}\nRUNNING-STATS sim (matches FINAL_GLAUCO)\n{'='*60}")
    for a, b, form in candidates:
        wide = load_pair(a, b)
        if form == "spread":
            sig = wide[a] - wide[b]
        else:
            sig = wide[a] + wide[b]
        simulate_running(sig, f"{a} {form} {b}")


if __name__ == "__main__":
    main()
