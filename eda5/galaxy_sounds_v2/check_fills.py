"""Check at which prices market trades occur for GALAXY_SOUNDS,
relative to the bid/ask. This tells us whether passive quotes at b+1/a-1 fill."""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
GS = ["GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
      "GALAXY_SOUNDS_SOLAR_WINDS"]


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    rows = []
    for d in (2, 3, 4):
        prices = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
        trades = pd.read_csv(DATA / f"trades_round_5_day_{d}.csv", sep=";")
        prices = prices[prices["product"].isin(GS)]
        trades = trades[trades["symbol"].isin(GS)]

        # Notably trades file across symbols looks identical — confirm
        sample = trades.head(20)
        print(f"\n--- Day {d} sample trades ---")
        print(sample.to_string(index=False))

        # Check whether trades file truly identical across symbols
        # (Same ts, qty, price) per symbol
        for sym in GS:
            sub = trades[trades["symbol"] == sym]
            print(f"  {sym}: n_trades={len(sub)}, qty_total={sub['quantity'].sum()}")
        print()

        # Join trades to book at same timestamp
        for sym in GS:
            t = trades[trades["symbol"] == sym].copy()
            p = prices[prices["product"] == sym][["timestamp", "bid_price_1", "ask_price_1", "mid_price"]]
            j = t.merge(p, on="timestamp", how="left")
            if j.empty:
                continue
            j["bid_dist"] = j["price"] - j["bid_price_1"]   # how far above bid
            j["ask_dist"] = j["ask_price_1"] - j["price"]   # how far below ask
            j["spread"] = j["ask_price_1"] - j["bid_price_1"]
            rows.append({
                "day": d, "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "n_trades": len(j),
                "spread_mean": float(j["spread"].mean()),
                "trade_at_bid": int((j["price"] == j["bid_price_1"]).sum()),
                "trade_below_bid": int((j["price"] < j["bid_price_1"]).sum()),
                "trade_at_ask": int((j["price"] == j["ask_price_1"]).sum()),
                "trade_above_ask": int((j["price"] > j["ask_price_1"]).sum()),
                "trade_inside": int(((j["price"] > j["bid_price_1"]) & (j["price"] < j["ask_price_1"])).sum()),
                "bid_dist_mean": float(j["bid_dist"].mean()),
                "ask_dist_mean": float(j["ask_dist"].mean()),
                "trade_at_or_below_b+1": int((j["price"] <= j["bid_price_1"] + 1).sum()),
                "trade_at_or_above_a-1": int((j["price"] >= j["ask_price_1"] - 1).sum()),
                "trade_at_or_below_b+2": int((j["price"] <= j["bid_price_1"] + 2).sum()),
                "trade_at_or_above_a-2": int((j["price"] >= j["ask_price_1"] - 2).sum()),
                "trade_at_or_below_b+3": int((j["price"] <= j["bid_price_1"] + 3).sum()),
                "trade_at_or_above_a-3": int((j["price"] >= j["ask_price_1"] - 3).sum()),
            })

    df = pd.DataFrame(rows)
    print("\n=== Trade locations vs bid/ask ===")
    print(df.to_string(index=False))

    print("\n=== Aggregate over 3 days per symbol ===")
    agg = df.groupby("sym").agg({
        "n_trades": "sum",
        "trade_at_bid": "sum",
        "trade_below_bid": "sum",
        "trade_at_ask": "sum",
        "trade_above_ask": "sum",
        "trade_inside": "sum",
        "trade_at_or_below_b+1": "sum",
        "trade_at_or_above_a-1": "sum",
        "trade_at_or_below_b+2": "sum",
        "trade_at_or_above_a-2": "sum",
    }).reset_index()
    agg["pct_buyfill_b+1"] = agg["trade_at_or_below_b+1"] / agg["n_trades"] * 100
    agg["pct_sellfill_a-1"] = agg["trade_at_or_above_a-1"] / agg["n_trades"] * 100
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
