"""GALAXY_SOUNDS — Realistic MM simulator that mirrors backtester logic.

Backtester fill rules (default match_trades=all):
  BUY  @ p: filled by market_trades with trade.price <= p (volume cap = market sell_qty).
  SELL @ p: filled by market_trades with trade.price >= p (volume cap = market buy_qty).
  Fill price = OUR p (not the market trade price).

Discovery: GALAXY_SOUNDS market trades are 51% at bid, 49% at ask, 0% inside.
So:
  BUY @ (bid + k) with k in [0..spread): filled by 51% (at-bid) trades.
  SELL @ (ask - k) similarly filled by 49% (at-ask) trades.

Every increment of k brings:
  - +1 unit of edge cost (we pay 1 more per unit).
  - But also +1 chance of being adversely selected (price moved against us).

Test: vary k_buy, k_sell from 1 to 6+. Use full size 10. Track inventory and exit at end of day.
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


def load_trades() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA / f"trades_round_5_day_{d}.csv", sep=";")
        df = df[df["symbol"].isin(GS)].copy()
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def simulate_realistic_mm(
    prices_g: pd.DataFrame, trades_g: pd.DataFrame,
    k_buy: int, k_sell: int,
    quote_size: int = 5, skew_per_unit: float = 0.0,
    use_momentum: bool = False, mom_N: int = 100, mom_th: float = 8,
    inventory_haircut: float = 0.0,  # bias quotes when inventory high
) -> dict:
    """Simulate one symbol+day. trades_g indexed by timestamp."""
    g = prices_g.sort_values("timestamp").reset_index(drop=True)
    bid = g["bid_price_1"].to_numpy()
    ask = g["ask_price_1"].to_numpy()
    mid = g["mid_price"].to_numpy()
    ts_arr = g["timestamp"].to_numpy()

    # Index trades by timestamp
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

        # determine side preference (momentum)
        boost_buy = 0.0; boost_sell = 0.0
        if use_momentum and t >= mom_N:
            mom = mid[t] - mid[t - mom_N]
            if mom > mom_th:
                # bullish: quote bid more aggressive (want long)
                boost_buy = +1.0
                boost_sell = -1.0  # less aggressive on sell side
            elif mom < -mom_th:
                boost_buy = -1.0
                boost_sell = +1.0

        # inventory skew: if pos high, quote ask more aggressive
        inv_skew = pos * skew_per_unit  # positive when long

        # final quote prices (passive in the spread)
        # k_buy = how far above bid (1 = best+1 standard MM, 2 = +2, etc.)
        # boost_buy positive = pull bid more towards mid (more aggressive)
        kb = int(round(k_buy + boost_buy + inv_skew))
        ks = int(round(k_sell + boost_sell - inv_skew))
        kb = max(1, kb); ks = max(1, ks)
        bq = bb + kb
        aq = aa - ks
        if bq >= aa: bq = aa - 1
        if aq <= bb: aq = bb + 1
        if bq >= aq:
            mid_int = (bb + aa) // 2
            bq = mid_int; aq = mid_int + 1

        buy_cap = LIMIT - pos
        sell_cap = LIMIT + pos

        # Determine fills using market trades at this timestamp
        ts_trades = trade_by_ts.get(int(ts_arr[t]), [])
        for tp, tq in ts_trades:
            # Each trade is one entity. It can fill our buy if tp <= bq, our sell if tp >= aq.
            # Order of preference matches backtester (it tries to match the order against trades in sequence).
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

    # mark-to-market at last mid
    if pos != 0:
        cash += pos * mid[-1]
    return {
        "pnl": cash,
        "buys": n_buys, "sells": n_sells,
        "mean_pos": float(np.mean(pos_track)) if pos_track else 0.0,
        "max_pos": int(np.max(pos_track)) if pos_track else 0,
        "min_pos": int(np.min(pos_track)) if pos_track else 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("Loading...")
    prices = load_prices()
    trades = load_trades()

    print("\n=== Sweep k_buy, k_sell symmetric, no momentum ===")
    rows = []
    ks = list(range(1, 8))
    for sym in GS:
        for k in ks:
            for sz in [5, 10]:
                day = []
                for d in (2, 3, 4):
                    pg = prices[(prices["product"] == sym) & (prices["day"] == d)]
                    tg = trades[(trades["symbol"] == sym) & (trades["day"] == d)]
                    r = simulate_realistic_mm(pg, tg, k_buy=k, k_sell=k, quote_size=sz)
                    day.append(r["pnl"])
                rows.append({"sym": sym.replace("GALAXY_SOUNDS_", ""), "k": k, "sz": sz,
                             "d2": day[0], "d3": day[1], "d4": day[2], "total": sum(day)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mm_sweep_k.csv", index=False)
    print(df.to_string(index=False))

    print("\n--- 3-day total per (k, size), summed over 5 symbols ---")
    print(df.groupby(["k", "sz"])["total"].sum().reset_index().sort_values("total", ascending=False).to_string(index=False))

    print("\n=== Sweep with momentum boost ===")
    rows2 = []
    for sym in GS:
        for k in [3, 4, 5, 6]:
            for sz in [5, 10]:
                for mn in [50, 100, 200]:
                    for mth in [3, 6, 10, 15]:
                        day = []
                        for d in (2, 3, 4):
                            pg = prices[(prices["product"] == sym) & (prices["day"] == d)]
                            tg = trades[(trades["symbol"] == sym) & (trades["day"] == d)]
                            r = simulate_realistic_mm(
                                pg, tg, k_buy=k, k_sell=k, quote_size=sz,
                                use_momentum=True, mom_N=mn, mom_th=mth,
                            )
                            day.append(r["pnl"])
                        rows2.append({
                            "sym": sym.replace("GALAXY_SOUNDS_", ""),
                            "k": k, "sz": sz, "mN": mn, "mth": mth,
                            "d2": day[0], "d3": day[1], "d4": day[2], "total": sum(day),
                        })
    df2 = pd.DataFrame(rows2)
    df2.to_csv(OUT / "mm_sweep_mom.csv", index=False)

    print("\n--- Top 5 per symbol ---")
    for sym in df2["sym"].unique():
        sub = df2[df2["sym"] == sym].sort_values("total", ascending=False).head(5)
        print(f"\n{sym}:")
        print(sub.to_string(index=False))

    print("\n--- Best uniform configuration over 5 symbols ---")
    by = df2.groupby(["k", "sz", "mN", "mth"])["total"].sum().reset_index().sort_values("total", ascending=False)
    print(by.head(15).to_string(index=False))

    print("\n--- Best per-symbol heterogeneous ---")
    bps = df2.sort_values("total", ascending=False).groupby("sym").head(1)
    print(bps.to_string(index=False))
    print(f"SUM best per-sym = {bps['total'].sum():,.0f}")


if __name__ == "__main__":
    main()
