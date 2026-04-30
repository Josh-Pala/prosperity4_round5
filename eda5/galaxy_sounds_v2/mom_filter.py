"""GALAXY_SOUNDS — Filtered momentum: only trade when |mom| exceeds spread cost.

Frictionless momentum was very profitable (~1.25M with size 10 over 3 days).
Aggressive trading lost everything because round-trip costs ~14 points.

Test: only enter when |mom| > spread + alpha. Hold M ticks. Then exit aggressive.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
GS = ["GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
      "GALAXY_SOUNDS_SOLAR_WINDS"]
LIMIT = 10


def load_prices() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
        df = df[df["product"].isin(GS)].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def simulate(g: pd.DataFrame, N: int, th: float, hold: int, size: int = LIMIT) -> dict:
    bid = g["bid_price_1"].to_numpy()
    ask = g["ask_price_1"].to_numpy()
    mid = g["mid_price"].to_numpy()
    n = len(mid)
    if n < N + hold + 5:
        return {"pnl": 0.0, "trades": 0}

    pos = 0
    cash = 0.0
    trades = 0
    hold_left = 0
    for t in range(N, n - 1):
        if hold_left > 0:
            hold_left -= 1
            if hold_left == 0 and pos != 0:
                # exit aggressive
                if pos > 0:
                    cash += pos * bid[t]
                else:
                    cash += pos * ask[t]
                pos = 0
                trades += 1
            continue
        if pos != 0:
            continue  # safety

        mom = mid[t] - mid[t - N]
        if mom > th:
            # enter long aggressive
            cash -= size * ask[t]
            pos = size
            hold_left = hold
            trades += 1
        elif mom < -th:
            cash += size * bid[t]
            pos = -size
            hold_left = hold
            trades += 1

    if pos != 0:
        cash += pos * mid[-1]
    return {"pnl": cash, "trades": trades}


def main() -> None:
    OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/galaxy_sounds_v2")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("Loading...")
    prices = load_prices()

    rows = []
    Ns = [50, 100, 200, 500]
    ths = [10, 15, 20, 30, 50, 75, 100]
    holds = [10, 20, 50, 100, 200]
    for sym in GS:
        for N, th, hd in product(Ns, ths, holds):
            day = []
            n_tr = []
            for d in (2, 3, 4):
                g = prices[(prices["product"] == sym) & (prices["day"] == d)].sort_values("timestamp").reset_index(drop=True)
                r = simulate(g, N=N, th=th, hold=hd)
                day.append(r["pnl"])
                n_tr.append(r["trades"])
            rows.append({
                "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "N": N, "th": th, "hold": hd,
                "d2": day[0], "d3": day[1], "d4": day[2], "total": sum(day),
                "tr": sum(n_tr),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mom_filter_grid.csv", index=False)

    print("\n--- Top 10 per symbol ---")
    for sym in df["sym"].unique():
        sub = df[df["sym"] == sym].sort_values("total", ascending=False).head(10)
        print(f"\n{sym}:")
        print(sub.to_string(index=False))

    print("\n--- Top uniform across 5 symbols ---")
    by = df.groupby(["N", "th", "hold"])[["total", "tr"]].sum().reset_index()
    print(by.sort_values("total", ascending=False).head(15).to_string(index=False))

    print("\n--- Best per-symbol ---")
    best = df.sort_values("total", ascending=False).groupby("sym").head(1)
    print(best.to_string(index=False))
    print(f"SUM best per-sym = {best['total'].sum():,.0f}")


if __name__ == "__main__":
    main()
