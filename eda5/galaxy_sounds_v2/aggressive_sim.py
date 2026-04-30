"""GALAXY_SOUNDS — Aggressive trend-following simulator.

Different model: WHEN signal is strong, CROSS THE SPREAD.
This is realistic against a backtester that fills aggressive orders at touch.

Strategy:
- mom_t = mid_t - mid_{t-N}
- if mom > entry_threshold: target = +LIMIT (long)
- if mom < -entry_threshold: target = -LIMIT (short)
- otherwise: hold current position
- exit threshold (smaller) for closing: target=0 if |mom|<exit_threshold
- aggressive at touch (buy at ask, sell at bid)

Costs:
- Round-trip ~ ask-bid (~13). Need momentum continuation > spread + slip.

Tune:
- N: lookback
- entry_th, exit_th
- size scaling
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


def simulate_aggressive(g: pd.DataFrame, N: int, entry_th: float, exit_th: float, size: int = LIMIT) -> dict:
    bid = g["bid_price_1"].to_numpy()
    ask = g["ask_price_1"].to_numpy()
    mid = g["mid_price"].to_numpy()
    n = len(mid)
    if n < N + 10:
        return {"pnl": 0.0, "trades": 0, "mean_pos": 0.0}

    pos = 0
    cash = 0.0
    trades = 0
    pos_track = []
    target = 0
    for t in range(N, n - 1):
        mom = mid[t] - mid[t - N]
        if mom > entry_th:
            target = size
        elif mom < -entry_th:
            target = -size
        else:
            if abs(mom) < exit_th:
                target = 0
        # execute aggressive
        if target != pos:
            d = target - pos
            if d > 0:
                cash -= d * ask[t]
                pos += d
                trades += 1
            elif d < 0:
                cash += (-d) * bid[t]
                pos += d
                trades += 1
        pos_track.append(pos)

    if pos != 0:
        cash += pos * mid[-1]
        pos = 0

    return {"pnl": cash, "trades": trades, "mean_pos": float(np.mean(pos_track)) if pos_track else 0.0}


def simulate_passive_lean(g: pd.DataFrame, N: int, entry_th: float, exit_th: float, size: int = LIMIT) -> dict:
    """Passive entry: when bullish, post bid at b+1 (buy passively).
    When bearish, post ask at a-1.
    Exit aggressive (cross spread) when target flips."""
    bid = g["bid_price_1"].to_numpy()
    ask = g["ask_price_1"].to_numpy()
    mid = g["mid_price"].to_numpy()
    n = len(mid)
    if n < N + 10:
        return {"pnl": 0.0, "trades": 0, "mean_pos": 0.0}

    pos = 0
    cash = 0.0
    trades = 0
    pos_track = []
    target = 0
    for t in range(N, n - 1):
        mom = mid[t] - mid[t - N]
        if mom > entry_th:
            target = size
        elif mom < -entry_th:
            target = -size
        else:
            if abs(mom) < exit_th:
                target = 0

        d = target - pos
        if d > 0:
            # need to buy
            # PASSIVE entry: try b+1, fill if ask_next <= b+1
            bq = int(bid[t]) + 1
            if ask[t + 1] <= bq:
                buy = min(d, size)
                cash -= buy * bq
                pos += buy
                trades += 1
            elif target * pos < 0 or abs(target) > abs(pos):
                # target swap or expand: take aggressive only if signal flipped
                # For now, only passive — skip if not filled
                pass
        elif d < 0:
            aq = int(ask[t]) - 1
            if bid[t + 1] >= aq:
                sell = min(-d, size)
                cash += sell * aq
                pos -= sell
                trades += 1
        else:
            # at target, do nothing
            pass

        # If signal flipped strongly and we're against it, exit aggressive
        if (target == 0 or (target > 0) != (pos > 0)) and pos != 0:
            # immediate exit on opposite signal
            if target * pos < 0 or (target == 0 and pos != 0 and abs(mom) < exit_th):
                # close
                if pos > 0:
                    cash += pos * bid[t]
                else:
                    cash += pos * ask[t]
                trades += 1
                pos = 0

        pos_track.append(pos)

    if pos != 0:
        cash += pos * mid[-1]
        pos = 0

    return {"pnl": cash, "trades": trades, "mean_pos": float(np.mean(pos_track)) if pos_track else 0.0}


def simulate_hybrid(g: pd.DataFrame, N: int, entry_th: float, exit_th: float, size: int = LIMIT,
                    cooldown: int = 0) -> dict:
    """Hybrid: passive entry attempt for `cooldown` ticks, then aggressive if not filled.
    Exit always aggressive."""
    bid = g["bid_price_1"].to_numpy()
    ask = g["ask_price_1"].to_numpy()
    mid = g["mid_price"].to_numpy()
    n = len(mid)
    if n < N + 10:
        return {"pnl": 0.0, "trades": 0, "mean_pos": 0.0}

    pos = 0
    cash = 0.0
    trades = 0
    pos_track = []
    target = 0
    waited = 0  # how long we've been trying passive
    for t in range(N, n - 1):
        mom = mid[t] - mid[t - N]
        new_target = target
        if mom > entry_th:
            new_target = size
        elif mom < -entry_th:
            new_target = -size
        else:
            if abs(mom) < exit_th:
                new_target = 0

        if new_target != target:
            target = new_target
            waited = 0

        d = target - pos
        if d != 0:
            # Try passive first
            filled = False
            if d > 0:
                bq = int(bid[t]) + 1
                if ask[t + 1] <= bq:
                    buy = min(d, size)
                    cash -= buy * bq
                    pos += buy
                    trades += 1
                    filled = True
            else:
                aq = int(ask[t]) - 1
                if bid[t + 1] >= aq:
                    sell = min(-d, size)
                    cash += sell * aq
                    pos -= sell
                    trades += 1
                    filled = True

            if not filled:
                waited += 1
                # Aggressive after cooldown
                if waited >= cooldown:
                    if d > 0:
                        cash -= d * ask[t]
                        pos += d
                        trades += 1
                    else:
                        cash += (-d) * bid[t]
                        pos += d
                        trades += 1
                    waited = 0
            else:
                waited = 0
        pos_track.append(pos)

    if pos != 0:
        cash += pos * mid[-1]
    return {"pnl": cash, "trades": trades, "mean_pos": float(np.mean(pos_track)) if pos_track else 0.0}


def grid_search(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    Ns = [50, 100, 200, 500]
    entry_ths = [3, 5, 8, 12, 16, 20, 30, 50]
    exit_ths_factor = [0.0, 0.3, 0.6]  # exit_th = entry_th * factor
    cooldowns = [0, 1, 5, 20]  # 0 = pure aggressive
    sizes = [10]

    for sym in GS:
        for N, ent, exf, cd, sz in product(Ns, entry_ths, exit_ths_factor, cooldowns, sizes):
            ext = ent * exf
            day_pnls = []
            for d in (2, 3, 4):
                g = prices[(prices["product"] == sym) & (prices["day"] == d)].sort_values("timestamp").reset_index(drop=True)
                if cd == 0:
                    r = simulate_aggressive(g, N=N, entry_th=ent, exit_th=ext, size=sz)
                else:
                    r = simulate_hybrid(g, N=N, entry_th=ent, exit_th=ext, size=sz, cooldown=cd)
                day_pnls.append(r["pnl"])
            rows.append({
                "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "mode": "aggr" if cd == 0 else f"hybr{cd}",
                "N": N, "entry": ent, "exit": ext, "cd": cd, "sz": sz,
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

    print("\n=== Grid search aggressive/hybrid ===")
    df = grid_search(prices)
    df.to_csv(OUT / "grid_aggressive.csv", index=False)

    print("\n--- Top 5 per symbol ---")
    for sym in df["sym"].unique():
        sub = df[df["sym"] == sym].sort_values("total", ascending=False).head(5)
        print(f"\n{sym}:")
        print(sub.to_string(index=False))

    print("\n--- Best per-symbol summed ---")
    best = df.sort_values("total", ascending=False).groupby("sym").head(1)
    print(best.to_string(index=False))
    print(f"\nSUM best per-symbol: {best['total'].sum():,.0f}")

    print("\n--- Best uniform config ---")
    by_cfg = df.groupby(["mode", "N", "entry", "exit", "cd", "sz"])["total"].sum().reset_index()
    print(by_cfg.sort_values("total", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
