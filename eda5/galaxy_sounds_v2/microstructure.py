"""GALAXY_SOUNDS — Microstructure deep dive.

Goal: capire perche' il MM puro fa solo 41k/3gg e dove sta l'edge mancante.
Output: dist spread, depth, frequenza top-of-book change, % tick con spread>=N,
PnL teorico per livello di passive entry.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/galaxy_sounds_v2")
GS = ["GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
      "GALAXY_SOUNDS_SOLAR_WINDS"]


def load_all_prices() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
        df = df[df["product"].isin(GS)].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_all_trades() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA / f"trades_round_5_day_{d}.csv", sep=";")
        df = df[df["symbol"].isin(GS)].copy()
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def microstructure(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, g in prices.groupby("product"):
        g = g.sort_values(["day", "timestamp"]).reset_index(drop=True)
        spread = g["ask_price_1"] - g["bid_price_1"]
        bid1 = g["bid_volume_1"]
        ask1 = g["ask_volume_1"]
        # depth on top-of-book
        # OFI proxy: changes in bid/ask top quotes weighted by volumes
        # spread freq
        for d in (2, 3, 4):
            gd = g[g["day"] == d]
            sp = gd["ask_price_1"] - gd["bid_price_1"]
            ticks = len(gd)
            row = {
                "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "day": d,
                "ticks": ticks,
                "spread_mean": float(sp.mean()),
                "spread_med": float(sp.median()),
                "spread_p90": float(sp.quantile(0.9)),
                "spread_max": int(sp.max()),
                "pct_spread_eq1": float((sp == 1).mean() * 100),
                "pct_spread_eq2": float((sp == 2).mean() * 100),
                "pct_spread_ge3": float((sp >= 3).mean() * 100),
                "pct_spread_ge4": float((sp >= 4).mean() * 100),
                "pct_spread_ge5": float((sp >= 5).mean() * 100),
                "bid1_vol_mean": float(gd["bid_volume_1"].mean()),
                "ask1_vol_mean": float(gd["ask_volume_1"].mean()),
                "bid1_vol_p10": float(gd["bid_volume_1"].quantile(0.1)),
                "ask1_vol_p10": float(gd["ask_volume_1"].quantile(0.1)),
                "mid_std": float(gd["mid_price"].diff().std()),
                "mid_chg_per100": float(gd["mid_price"].diff().abs().sum() / max(1, len(gd)) * 100),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def trade_stats(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Trade flow per symbol+day; signed by mid at trade time."""
    px_keys = prices.set_index(["day", "timestamp", "product"])[
        ["bid_price_1", "ask_price_1", "mid_price"]
    ]
    rows = []
    for sym in GS:
        for d in (2, 3, 4):
            t = trades[(trades["symbol"] == sym) & (trades["day"] == d)]
            if t.empty:
                continue
            # try to align to nearest tick
            joined = t.merge(
                prices[(prices["day"] == d) & (prices["product"] == sym)][
                    ["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]
                ],
                left_on=["day", "timestamp"], right_on=["day", "timestamp"], how="left"
            )
            # signed: trade above mid -> buyer initiated (+), below -> seller (-)
            mid = joined["mid_price"]
            sgn = np.sign(joined["price"] - mid)
            qty = joined["quantity"]
            sgn_qty = sgn * qty
            row = {
                "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "day": d,
                "n_trades": int(len(t)),
                "total_qty": int(qty.sum()),
                "buyer_init_qty": int(qty[sgn > 0].sum()),
                "seller_init_qty": int(qty[sgn < 0].sum()),
                "midline_qty": int(qty[sgn == 0].sum()),
                "net_signed_qty": int(sgn_qty.sum()),
                "mean_trade_size": float(qty.mean()),
                "trade_vs_mid_mean_bps": float(((joined["price"] - mid) / mid * 1e4).mean()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def passive_fill_simulation(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Per ogni simbolo/giorno, simula: per ogni tick t con spread>=2,
    quotare bid=best_bid+1 e ask=best_ask-1 (size 5).
    Fill se nel tick t+1 il mid si muove ATTRAVERSO il quote.
    Misura: % fill rate, edge realizzato (mid_t+1 - quote_price), PnL atteso.
    Questo isola la qualita' del MM puro."""
    rows = []
    for sym in GS:
        for d in (2, 3, 4):
            g = prices[(prices["day"] == d) & (prices["product"] == sym)].sort_values("timestamp").reset_index(drop=True)
            if len(g) < 100:
                continue
            sp = g["ask_price_1"] - g["bid_price_1"]
            mid = g["mid_price"]
            mid_next = mid.shift(-1)
            bid_q = g["bid_price_1"] + 1  # our bid
            ask_q = g["ask_price_1"] - 1  # our ask
            # Fill se mid_next < bid_q (per il bid: qualcuno ha venduto)
            # Pi precisamente: ask_next <= bid_q
            ask_next = g["ask_price_1"].shift(-1)
            bid_next = g["bid_price_1"].shift(-1)
            mask_quote = sp >= 2
            buy_fill = mask_quote & (ask_next <= bid_q)
            sell_fill = mask_quote & (bid_next >= ask_q)
            # edge per fill (in punti, tipicamente 1)
            buy_edge = (mid_next - bid_q)[buy_fill]
            sell_edge = (ask_q - mid_next)[sell_fill]
            row = {
                "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "day": d,
                "ticks_with_quote": int(mask_quote.sum()),
                "buy_fills": int(buy_fill.sum()),
                "sell_fills": int(sell_fill.sum()),
                "buy_fill_rate_%": float(buy_fill.sum() / max(1, mask_quote.sum()) * 100),
                "sell_fill_rate_%": float(sell_fill.sum() / max(1, mask_quote.sum()) * 100),
                "buy_edge_mean": float(buy_edge.mean()) if not buy_edge.empty else 0.0,
                "sell_edge_mean": float(sell_edge.mean()) if not sell_edge.empty else 0.0,
                "naive_pnl_size1": float(buy_edge.sum() + sell_edge.sum()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading prices...")
    prices = load_all_prices()
    print(f"  {len(prices):,} rows for GS")
    print("Loading trades...")
    trades = load_all_trades()
    print(f"  {len(trades):,} trades for GS")

    print("\n=== Microstructure ===")
    ms = microstructure(prices)
    ms.to_csv(OUT / "microstructure.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(ms.to_string(index=False))

    print("\n=== Trade flow ===")
    tf = trade_stats(trades, prices)
    tf.to_csv(OUT / "trade_flow.csv", index=False)
    print(tf.to_string(index=False))

    print("\n=== Passive MM fill simulation (size 1, naive) ===")
    pf = passive_fill_simulation(prices, trades)
    pf.to_csv(OUT / "passive_fill_sim.csv", index=False)
    print(pf.to_string(index=False))

    print("\n=== Aggregate by symbol (3-day) ===")
    agg = pf.groupby("sym").agg(
        ticks_with_quote=("ticks_with_quote", "sum"),
        buy_fills=("buy_fills", "sum"),
        sell_fills=("sell_fills", "sum"),
        naive_pnl_size1=("naive_pnl_size1", "sum"),
    ).reset_index()
    agg["naive_pnl_size5"] = agg["naive_pnl_size1"] * 5
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
