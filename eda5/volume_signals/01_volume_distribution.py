"""
01 - Volume distribution analysis.

Goal: trovare 'firme' di trader informati. Per ogni prodotto, capire:
- distribuzione dei volumi nei trades market
- distribuzione dei volumi resting nell'order book (bid/ask level 1,2,3)
- volumi 'unusual' (>= P90, >= P95, >= P99) e taglie ricorrenti (50, 100, 200...)
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/volume_signals")
OUT.mkdir(exist_ok=True)

DAYS = [2, 3, 4]


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"trades_round_5_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    prices = load_prices()
    trades = load_trades()

    print(f"prices rows: {len(prices):,}  symbols: {prices['product'].nunique()}")
    print(f"trades rows: {len(trades):,}  symbols: {trades['symbol'].nunique()}")

    # OB volumes (collapsed across levels)
    ob_vol_records = []
    for col in ["bid_volume_1", "bid_volume_2", "bid_volume_3",
                "ask_volume_1", "ask_volume_2", "ask_volume_3"]:
        side = "bid" if "bid" in col else "ask"
        sub = prices[["product", col]].dropna()
        sub = sub[sub[col] > 0]
        sub = sub.rename(columns={col: "volume"})
        sub["side"] = side
        sub["level"] = col[-1]
        ob_vol_records.append(sub)
    ob_vol = pd.concat(ob_vol_records, ignore_index=True)

    # ---- 1. Trades volume distribution per symbol ----
    trade_stats = (
        trades.groupby("symbol")["quantity"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
            p95=lambda s: s.quantile(0.95),
            p99=lambda s: s.quantile(0.99),
            max="max",
        )
        .round(2)
        .sort_values("p99", ascending=False)
    )
    trade_stats.to_csv(OUT / "trade_volume_stats.csv")
    print("\n=== TRADE volume p99 (top 10) ===")
    print(trade_stats.head(10))

    # ---- 2. OB resting volume distribution per symbol ----
    ob_stats = (
        ob_vol.groupby("product")["volume"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
            p95=lambda s: s.quantile(0.95),
            p99=lambda s: s.quantile(0.99),
            max="max",
        )
        .round(2)
        .sort_values("p99", ascending=False)
    )
    ob_stats.to_csv(OUT / "ob_volume_stats.csv")
    print("\n=== OB resting volume p99 (top 10) ===")
    print(ob_stats.head(10))

    # ---- 3. Round-number / cluster sizes ----
    # Per ogni prodotto, conta quanto spesso compaiono volumi 'tondi' (10,20,50,100,200,...)
    # tra i top 1% di trades.
    candidates = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 500]
    rows = []
    for sym, g in trades.groupby("symbol"):
        thr99 = g["quantity"].quantile(0.99)
        big = g[g["quantity"] >= thr99]
        for v in candidates:
            cnt_total = (g["quantity"] == v).sum()
            cnt_big = (big["quantity"] == v).sum()
            if cnt_total == 0 and cnt_big == 0:
                continue
            rows.append({
                "symbol": sym,
                "size": v,
                "count_all": int(cnt_total),
                "count_top1pct": int(cnt_big),
                "share_top1pct": cnt_big / max(len(big), 1),
                "p99": thr99,
            })
    cluster = pd.DataFrame(rows).sort_values(
        ["share_top1pct", "count_top1pct"], ascending=[False, False]
    )
    cluster.to_csv(OUT / "trade_round_sizes.csv", index=False)
    print("\n=== Round-number sizes most frequent in top1pct trades ===")
    print(cluster.head(20))

    # OB resting: stessa cosa
    rows = []
    for sym, g in ob_vol.groupby("product"):
        thr99 = g["volume"].quantile(0.99)
        big = g[g["volume"] >= thr99]
        for v in candidates:
            cnt_big = (big["volume"] == v).sum()
            if cnt_big == 0:
                continue
            rows.append({
                "symbol": sym,
                "size": v,
                "count_top1pct": int(cnt_big),
                "share_top1pct": cnt_big / max(len(big), 1),
                "p99": thr99,
            })
    ob_cluster = pd.DataFrame(rows).sort_values(
        ["share_top1pct", "count_top1pct"], ascending=[False, False]
    )
    ob_cluster.to_csv(OUT / "ob_round_sizes.csv", index=False)
    print("\n=== OB resting round sizes in top1pct ===")
    print(ob_cluster.head(20))


if __name__ == "__main__":
    main()
