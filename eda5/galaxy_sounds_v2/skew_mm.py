"""GALAXY_SOUNDS — MM with aggressive inventory management.

Key insight: 41k baseline comes mostly from 1-2 lucky days per symbol.
Other days lose due to inventory drift (1-sided fills when prices trend).

Solution: skew quotes hard against inventory.
  When long N units: pull bid back (less aggressive buy), push ask up close to mid (more aggressive sell).
  Quote 'fair = mid - skew*pos'.
  Fair price determines bq=fair-half, aq=fair+half.

Also: hard inventory cap (e.g., max 5 instead of 10) → soft-position management.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
import sys
sys.path.insert(0, str(Path(__file__).parent))
from realistic_mm import simulate_realistic_mm, load_prices, load_trades, GS, LIMIT


def simulate_mm_skewed(
    prices_g: pd.DataFrame, trades_g: pd.DataFrame,
    half_spread: int, skew_per_unit: float,
    quote_size: int = 5, max_inventory: int = 10,
    use_mom: bool = False, mom_N: int = 100, mom_th: float = 8,
    mom_shift: float = 2.0,
) -> dict:
    """Quote bq = round(fair - half_spread), aq = round(fair + half_spread)
    where fair = mid - skew_per_unit * pos +/- mom_shift if momentum.
    Hard cap at max_inventory.
    """
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
    pos_track = []

    for t in range(n):
        bb = int(bid[t]); aa = int(ask[t])

        # Fair price with inventory skew
        fair = mid[t] - skew_per_unit * pos
        if use_mom and t >= mom_N:
            mom = mid[t] - mid[t - mom_N]
            if mom > mom_th:
                fair += mom_shift  # raise our fair (we expect up)
            elif mom < -mom_th:
                fair -= mom_shift

        bq = int(round(fair - half_spread))
        aq = int(round(fair + half_spread))
        # constrain in spread
        bq = max(bb + 1, min(bq, aa - 1))
        aq = max(bb + 1, min(aq, aa - 1))
        if bq >= aq:
            mid_int = (bb + aa) // 2
            bq = mid_int; aq = mid_int + 1
            bq = max(bb + 1, min(bq, aa - 1))
            aq = max(bb + 1, min(aq, aa - 1))

        # capacity with hard inventory cap
        buy_cap = max(0, max_inventory - pos)
        sell_cap = max(0, max_inventory + pos)

        ts_trades = trade_by_ts.get(int(ts_arr[t]), [])
        for tp, tq in ts_trades:
            if tp <= bq and buy_cap > 0:
                fill = min(buy_cap, tq, quote_size)
                if fill > 0:
                    cash -= fill * bq
                    pos += fill
                    buy_cap -= fill
                    n_buys += 1
            if tp >= aq and sell_cap > 0:
                fill = min(sell_cap, tq, quote_size)
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

    print("\n=== Skewed MM grid ===")
    rows = []
    half_sps = [4, 5, 6, 7, 8]            # how far from fair
    skews = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # skew per unit
    sizes = [5, 10]
    max_invs = [3, 5, 7, 10]

    for sym in GS:
        for hs, sk, sz, mi in product(half_sps, skews, sizes, max_invs):
            day = []
            for d in (2, 3, 4):
                pg = prices[(prices["product"] == sym) & (prices["day"] == d)]
                tg = trades[(trades["symbol"] == sym) & (trades["day"] == d)]
                r = simulate_mm_skewed(pg, tg, half_spread=hs, skew_per_unit=sk,
                                       quote_size=sz, max_inventory=mi)
                day.append(r["pnl"])
            rows.append({
                "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "hs": hs, "skew": sk, "sz": sz, "maxI": mi,
                "d2": day[0], "d3": day[1], "d4": day[2], "total": sum(day),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mm_skew_grid.csv", index=False)

    print("\n--- Top 5 per symbol ---")
    for sym in df["sym"].unique():
        sub = df[df["sym"] == sym].sort_values("total", ascending=False).head(5)
        print(f"\n{sym}:")
        print(sub.to_string(index=False))

    print("\n--- Best uniform config across 5 symbols ---")
    by = df.groupby(["hs", "skew", "sz", "maxI"])["total"].sum().reset_index()
    print(by.sort_values("total", ascending=False).head(10).to_string(index=False))

    print("\n--- Best per-symbol heterogeneous ---")
    bps = df.sort_values("total", ascending=False).groupby("sym").head(1)
    print(bps.to_string(index=False))
    print(f"SUM best per-sym = {bps['total'].sum():,.0f}")

    print("\n--- For each symbol, max guaranteed (worst day >= 0) configs ---")
    for sym in df["sym"].unique():
        sub = df[df["sym"] == sym].copy()
        sub["min_day"] = sub[["d2", "d3", "d4"]].min(axis=1)
        sub = sub[sub["min_day"] >= 0].sort_values("total", ascending=False)
        if not sub.empty:
            print(f"\n{sym} (positive every day):")
            print(sub.head(3).to_string(index=False))
        else:
            print(f"\n{sym}: no config with all days positive.")


if __name__ == "__main__":
    main()
