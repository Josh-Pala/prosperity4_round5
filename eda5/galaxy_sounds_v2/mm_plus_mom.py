"""GALAXY_SOUNDS — MM (always-on, k=1) + parallel directional overlay.

Two layers, executed independently then combined for position cap.
- Layer A (MM): quote bid+1, ask-1, size 5, no skew. Independent target_pos_A.
- Layer B (Momentum directional): when |mom_N| > th, target_pos_B = +/- size_mom.
- Net target = clip(A_pos + B_pos, ±LIMIT).

Implementation: track two virtual positions; merge for execution.
But we have only one real position. So we route fills:
  - MM fills come from passive at b+1/a-1.
  - Momentum overlay: aggressive take, but only if it doesn't violate cap.

Simulator simplification: when momentum signal active, instead of trading aggressively,
INCREASE the MM quote_size on the favored side (more bid quote when bullish).
This gathers passive fills in the right direction without crossing spread.
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


def load_trades() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA / f"trades_round_5_day_{d}.csv", sep=";")
        df = df[df["symbol"].isin(GS)].copy()
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def simulate(
    prices_g: pd.DataFrame, trades_g: pd.DataFrame,
    mm_size: int = 5, mom_size: int = 5,
    mom_N: int = 200, mom_th_enter: float = 30, mom_th_exit: float = 10,
    inv_skew: float = 0.0,  # punish positions: shifts fair against us
    take_aggressive: bool = False,
    take_threshold: float = 60,  # only take if very strong momentum
) -> dict:
    g = prices_g.sort_values("timestamp").reset_index(drop=True)
    bid = g["bid_price_1"].to_numpy()
    ask = g["ask_price_1"].to_numpy()
    mid = g["mid_price"].to_numpy()
    ts_arr = g["timestamp"].to_numpy()

    trade_by_ts: dict[int, list[tuple[float, int]]] = {}
    for ts, p, q in zip(trades_g["timestamp"].to_numpy(), trades_g["price"].to_numpy(), trades_g["quantity"].to_numpy()):
        trade_by_ts.setdefault(int(ts), []).append((float(p), int(q)))

    n = len(g)
    pos = 0
    cash = 0.0
    n_buys = 0; n_sells = 0
    mom_state = 0  # -1 / 0 / +1
    pos_track = []

    for t in range(n):
        bb = int(bid[t]); aa = int(ask[t])

        # Momentum state machine
        if t >= mom_N:
            mom = mid[t] - mid[t - mom_N]
            if mom_state == 0:
                if mom > mom_th_enter: mom_state = 1
                elif mom < -mom_th_enter: mom_state = -1
            else:
                if abs(mom) < mom_th_exit:
                    mom_state = 0
                elif mom_state == 1 and mom < -mom_th_enter:
                    mom_state = -1
                elif mom_state == -1 and mom > mom_th_enter:
                    mom_state = 1
        else:
            mom = 0

        # MM quote sizes by side; momentum tilts the size
        if mom_state == 1:
            # Bullish — quote big on bid, small on ask
            buy_quote_size = mm_size + mom_size
            sell_quote_size = max(0, mm_size - mom_size)
        elif mom_state == -1:
            buy_quote_size = max(0, mm_size - mom_size)
            sell_quote_size = mm_size + mom_size
        else:
            buy_quote_size = mm_size
            sell_quote_size = mm_size

        # Inventory skew: fair = mid - inv_skew * pos
        # bq = bb+1 - some adjustment, aq = aa-1 + some adjustment
        # Simplest: when long, push bid quote DOWN (less aggressive bid), ask DOWN (more aggressive sell)
        # When short, opposite. Here: shift both quotes by -inv_skew*pos rounded
        shift = -int(round(inv_skew * pos))
        bq = bb + 1 + shift
        aq = aa - 1 + shift
        bq = max(bb + 1, min(bq, aa - 1))
        aq = max(bb + 1, min(aq, aa - 1))
        if bq >= aq:
            mid_int = (bb + aa) // 2
            bq = mid_int; aq = mid_int + 1
            bq = max(bb + 1, min(bq, aa - 1))
            aq = max(bb + 1, min(aq, aa - 1))

        buy_cap = LIMIT - pos
        sell_cap = LIMIT + pos

        # Aggressive take if very strong momentum
        if take_aggressive and t >= mom_N:
            if mom > take_threshold and buy_cap > 0:
                tq = min(buy_cap, mom_size)
                if tq > 0:
                    cash -= tq * aa
                    pos += tq
                    buy_cap -= tq
                    n_buys += 1
            elif mom < -take_threshold and sell_cap > 0:
                tq = min(sell_cap, mom_size)
                if tq > 0:
                    cash += tq * bb
                    pos -= tq
                    sell_cap -= tq
                    n_sells += 1

        # Passive fills against market trades at this tick
        ts_trades = trade_by_ts.get(int(ts_arr[t]), [])
        for tp, tq in ts_trades:
            if tp <= bq and buy_cap > 0 and buy_quote_size > 0:
                fill = min(buy_cap, tq, buy_quote_size)
                if fill > 0:
                    cash -= fill * bq
                    pos += fill
                    buy_cap -= fill
                    n_buys += 1
            if tp >= aq and sell_cap > 0 and sell_quote_size > 0:
                fill = min(sell_cap, tq, sell_quote_size)
                if fill > 0:
                    cash += fill * aq
                    pos -= fill
                    sell_cap -= fill
                    n_sells += 1
        pos_track.append(pos)

    if pos != 0:
        cash += pos * mid[-1]
    return {"pnl": cash, "buys": n_buys, "sells": n_sells,
            "mean_pos": float(np.mean(pos_track)) if pos_track else 0.0}


def main() -> None:
    OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/galaxy_sounds_v2")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)

    prices = load_prices()
    trades = load_trades()

    rows = []
    mm_sizes = [5]
    mom_sizes = [0, 3, 5]
    mom_Ns = [100, 200]
    mom_th_enters = [20, 50, 100]
    mom_th_exit_facs = [0.3]
    inv_skews = [0.0, 1.0]
    takes = [False]
    take_ths = [99999]

    print("Running grid...")
    cnt = 0
    for sym in GS:
        for mm_sz, m_sz, mn, te, tf, isk, tk in product(mm_sizes, mom_sizes, mom_Ns, mom_th_enters, mom_th_exit_facs, inv_skews, takes):
            if tk:
                tk_ths = take_ths
            else:
                tk_ths = [99999]
            for tk_th in tk_ths:
                day = []
                for d in (2, 3, 4):
                    pg = prices[(prices["product"] == sym) & (prices["day"] == d)]
                    tg = trades[(trades["symbol"] == sym) & (trades["day"] == d)]
                    r = simulate(
                        pg, tg, mm_size=mm_sz, mom_size=m_sz,
                        mom_N=mn, mom_th_enter=te, mom_th_exit=te * tf,
                        inv_skew=isk, take_aggressive=tk, take_threshold=tk_th,
                    )
                    day.append(r["pnl"])
                rows.append({
                    "sym": sym.replace("GALAXY_SOUNDS_", ""),
                    "mm_sz": mm_sz, "m_sz": m_sz, "mN": mn, "te": te, "tf": tf,
                    "isk": isk, "take": tk, "tk_th": tk_th,
                    "d2": day[0], "d3": day[1], "d4": day[2], "total": sum(day),
                })
                cnt += 1
    print(f"  {cnt} configs evaluated")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mm_plus_mom_grid.csv", index=False)

    print("\n--- Top 5 per symbol ---")
    for sym in df["sym"].unique():
        sub = df[df["sym"] == sym].sort_values("total", ascending=False).head(5)
        print(f"\n{sym}:")
        print(sub.to_string(index=False))

    print("\n--- Best uniform across 5 symbols ---")
    by = df.groupby(["mm_sz", "m_sz", "mN", "te", "tf", "isk", "take", "tk_th"])["total"].sum().reset_index()
    print(by.sort_values("total", ascending=False).head(15).to_string(index=False))

    print("\n--- Best per-symbol heterogeneous ---")
    bps = df.sort_values("total", ascending=False).groupby("sym").head(1)
    print(bps.to_string(index=False))
    print(f"SUM best per-sym = {bps['total'].sum():,.0f}")

    print("\n--- Configs where ALL 3 days POSITIVE (per symbol) ---")
    for sym in df["sym"].unique():
        sub = df[df["sym"] == sym].copy()
        sub["min_day"] = sub[["d2", "d3", "d4"]].min(axis=1)
        sub = sub[sub["min_day"] >= 0].sort_values("total", ascending=False)
        if not sub.empty:
            print(f"\n{sym} top-3:")
            print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
