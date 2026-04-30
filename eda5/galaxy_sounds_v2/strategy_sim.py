"""GALAXY_SOUNDS — Realistic strategy simulator.

Approach: directional market-maker.
- Maintain rolling momentum signal: mom_t = mid_t - mid_{t-N}
- Maintain rolling imbalance: imb_t = (bv1 - av1) / (bv1 + av1)
- Combined score s = mom + alpha * imb_scale
- If s > +threshold: bull regime → quote bid aggressive (size MAX), ask wide
- If s < -threshold: bear regime → quote ask aggressive, bid wide
- Else: balanced two-sided MM

Execution model:
- We submit limit orders at b+1 / a-1 (passive entry).
- A buy@b+1 is filled at tick t+1 if ask_{t+1} <= b+1.
- A sell@a-1 is filled at tick t+1 if bid_{t+1} >= a-1.
- Market exit: if hold time exceeded or signal flips, exit aggressive.

This emulates the IMC backtester behavior reasonably for spread>=2 markets.

Goal: find params that yield 200k+ across 3 days for 5 symbols.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/galaxy_sounds_v2")
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


def simulate_directional_mm(
    g: pd.DataFrame, N: int, threshold: float,
    use_imb: bool = True, imb_alpha: float = 5.0,
    quote_size: int = 5, exit_on_flip: bool = True,
    take_when_strong: bool = False, strong_mult: float = 2.0,
) -> dict:
    """Simulate one (sym, day). Returns pnl, trades, mean_pos."""
    bid = g["bid_price_1"].to_numpy()
    ask = g["ask_price_1"].to_numpy()
    mid = g["mid_price"].to_numpy()
    bv = g["bid_volume_1"].to_numpy().astype(float)
    av = g["ask_volume_1"].to_numpy().astype(float)
    n = len(mid)
    if n < N + 10:
        return {"pnl": 0.0, "trades": 0, "mean_pos": 0.0, "max_pos": 0, "min_pos": 0}

    pos = 0
    cash = 0.0
    trades = 0
    pos_track = []

    for t in range(N, n - 1):
        mom = mid[t] - mid[t - N]
        imb = (bv[t] - av[t]) / max(bv[t] + av[t], 1.0) if use_imb else 0.0
        score = mom + imb_alpha * imb

        # decide quotes for next tick
        b_quote = int(bid[t]) + 1
        a_quote = int(ask[t]) - 1

        # capacity
        buy_cap = LIMIT - pos
        sell_cap = LIMIT + pos

        bull = score > threshold
        bear = score < -threshold

        # passive fills (depend on next-tick book move)
        # buy_fill: ask_{t+1} <= b_quote
        ask_next = ask[t + 1]
        bid_next = bid[t + 1]
        buy_filled = ask_next <= b_quote
        sell_filled = bid_next >= a_quote

        if bull:
            # Quote big on bid, small/none on ask (we want to BUILD long)
            buy_qty = min(quote_size, buy_cap)
            sell_qty = 0  # don't quote ask (or wide)
            if buy_filled and buy_qty > 0:
                pos += buy_qty; cash -= buy_qty * b_quote; trades += 1
            # optional aggressive take if very strong
            if take_when_strong and score > strong_mult * threshold and buy_cap > 0:
                # cross spread, take what's available at ask
                take_qty = min(buy_cap, int(av[t]))
                if take_qty > 0:
                    pos += take_qty; cash -= take_qty * ask[t]; trades += 1
        elif bear:
            buy_qty = 0
            sell_qty = min(quote_size, sell_cap)
            if sell_filled and sell_qty > 0:
                pos -= sell_qty; cash += sell_qty * a_quote; trades += 1
            if take_when_strong and score < -strong_mult * threshold and sell_cap > 0:
                take_qty = min(sell_cap, int(bv[t]))
                if take_qty > 0:
                    pos -= take_qty; cash += take_qty * bid[t]; trades += 1
        else:
            # balanced two-sided MM
            buy_qty = min(quote_size, buy_cap)
            sell_qty = min(quote_size, sell_cap)
            if buy_filled and buy_qty > 0:
                pos += buy_qty; cash -= buy_qty * b_quote; trades += 1
            if sell_filled and sell_qty > 0:
                pos -= sell_qty; cash += sell_qty * a_quote; trades += 1

        # exit on signal flip
        if exit_on_flip:
            if pos > 0 and bear:
                # cross to flat
                cash += pos * bid[t]; pos = 0; trades += 1
            elif pos < 0 and bull:
                cash += pos * ask[t]; pos = 0; trades += 1

        pos_track.append(pos)

    # close at last mid
    if pos != 0:
        cash += pos * mid[-1]
        pos = 0

    return {
        "pnl": cash,
        "trades": trades,
        "mean_pos": float(np.mean(pos_track)) if pos_track else 0.0,
        "max_pos": int(np.max(pos_track)) if pos_track else 0,
        "min_pos": int(np.min(pos_track)) if pos_track else 0,
    }


def grid_search(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    Ns = [20, 50, 100, 200]
    thresholds = [1.0, 2.0, 4.0, 8.0]
    quote_sizes = [5, 10]
    use_imb_opts = [False, True]
    take_opts = [False, True]

    for sym in GS:
        for N, th, qs, ui, tk in product(Ns, thresholds, quote_sizes, use_imb_opts, take_opts):
            day_pnls = []
            for d in (2, 3, 4):
                g = prices[(prices["product"] == sym) & (prices["day"] == d)].sort_values("timestamp").reset_index(drop=True)
                r = simulate_directional_mm(
                    g, N=N, threshold=th, use_imb=ui, imb_alpha=5.0,
                    quote_size=qs, exit_on_flip=True,
                    take_when_strong=tk, strong_mult=2.0,
                )
                day_pnls.append(r["pnl"])
            rows.append({
                "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "N": N, "th": th, "qs": qs, "imb": ui, "take": tk,
                "d2": day_pnls[0], "d3": day_pnls[1], "d4": day_pnls[2],
                "total": sum(day_pnls),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("Loading...")
    prices = load_prices()

    print("\n=== Grid search (directional MM) ===")
    df = grid_search(prices)
    df.to_csv(OUT / "grid_strategy.csv", index=False)

    print("\n--- Top 20 configs by total per symbol ---")
    for sym in df["sym"].unique():
        sub = df[df["sym"] == sym].sort_values("total", ascending=False).head(5)
        print(f"\n{sym}:")
        print(sub.to_string(index=False))

    print("\n--- Best uniform config across symbols (use same params) ---")
    by_cfg = df.groupby(["N", "th", "qs", "imb", "take"]).agg(
        total=("total", "sum"),
        n_neg_days=("d2", lambda s: (s < 0).sum()),  # placeholder
    ).reset_index()
    print(by_cfg.sort_values("total", ascending=False).head(15).to_string(index=False))

    print("\n--- Best per-symbol config (allow heterogeneous) ---")
    bps = df.sort_values("total", ascending=False).groupby("sym").head(1)
    print(bps.to_string(index=False))
    print(f"\nSUM of best per-symbol: {bps['total'].sum():,.0f}")


if __name__ == "__main__":
    main()
