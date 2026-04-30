"""
05 - Drill-down su TRANSLATOR_ASTRO_BLACK e TRANSLATOR_ECLIPSE_CHARCOAL.

Verifica che il pattern 'big bid resting → mid up' sia robusto:
- Test su tutti i percentili 80, 90, 95, 99 della bid_volume_1
- Test horizon 30, 50, 100, 200
- Cross-check: anche big_ask_resting → short funziona?
- Quanti eventi al giorno → frequenza pratica per il bot

Inoltre: caratterizza il segnale: dimensione media, prezzo, spread.
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/volume_signals")
DAYS = [2, 3, 4]
SYMBOLS = ["TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
           "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE"]


def load() -> pd.DataFrame:
    frames = [pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";") for d in DAYS]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return df


def grid_for(g: pd.DataFrame, sym: str) -> pd.DataFrame:
    g = g.copy()
    spread = float((g["ask_price_1"] - g["bid_price_1"]).median())

    # Distribution of bid_volume_1 / ask_volume_1
    bv1 = g["bid_volume_1"].dropna()
    av1 = g["ask_volume_1"].dropna()
    print(f"\n--- {sym} ---")
    print(f"  bid_volume_1: median={bv1.median():.0f}  p90={bv1.quantile(0.9):.0f}  "
          f"p95={bv1.quantile(0.95):.0f}  p99={bv1.quantile(0.99):.0f}  max={bv1.max():.0f}")
    print(f"  ask_volume_1: median={av1.median():.0f}  p90={av1.quantile(0.9):.0f}  "
          f"p95={av1.quantile(0.95):.0f}  p99={av1.quantile(0.99):.0f}  max={av1.max():.0f}")
    print(f"  median spread: {spread:.0f}")

    rows = []
    for pct in [80, 90, 95, 99]:
        thr_b = bv1.quantile(pct / 100)
        thr_a = av1.quantile(pct / 100)
        big_b = g["bid_volume_1"] >= thr_b
        big_a = g["ask_volume_1"] >= thr_a

        for h in [30, 50, 100, 200]:
            fut = (g["mid_price"].shift(-h) - g["mid_price"]).values

            # Long con big bid only
            mask = (big_b & ~big_a).values
            pnl = fut[mask] - spread
            pnl = pnl[~np.isnan(pnl)]
            if len(pnl) > 30:
                rows.append({
                    "symbol": sym, "side": "BIG_BID_ONLY_long", "pct": pct, "h": h,
                    "thr": int(thr_b), "n": len(pnl),
                    "mean": round(pnl.mean(), 2),
                    "tot": round(pnl.sum(), 1),
                    "win": round((pnl > 0).mean(), 3),
                })
            # Short con big ask only
            mask = (big_a & ~big_b).values
            pnl = -fut[mask] - spread
            pnl = pnl[~np.isnan(pnl)]
            if len(pnl) > 30:
                rows.append({
                    "symbol": sym, "side": "BIG_ASK_ONLY_short", "pct": pct, "h": h,
                    "thr": int(thr_a), "n": len(pnl),
                    "mean": round(pnl.mean(), 2),
                    "tot": round(pnl.sum(), 1),
                    "win": round((pnl > 0).mean(), 3),
                })
    return pd.DataFrame(rows)


def per_day_stability(g: pd.DataFrame, sym: str, pct: int, h: int, side: str,
                      spread: float) -> dict:
    out = {}
    for d in DAYS:
        sub = g[g["day"] == d]
        bv1 = sub["bid_volume_1"].dropna()
        av1 = sub["ask_volume_1"].dropna()
        # Use full-period thresholds (would be in-sample for day-by-day; use global)
        if bv1.empty:
            continue
        # Use global thresholds from concatenated, then evaluate on day
        global_b = g["bid_volume_1"].quantile(pct / 100)
        global_a = g["ask_volume_1"].quantile(pct / 100)
        big_b = sub["bid_volume_1"] >= global_b
        big_a = sub["ask_volume_1"] >= global_a
        fut = (sub["mid_price"].shift(-h) - sub["mid_price"]).values
        if side == "BIG_BID_ONLY_long":
            mask = (big_b & ~big_a).values
            pnl = fut[mask] - spread
        else:
            mask = (big_a & ~big_b).values
            pnl = -fut[mask] - spread
        pnl = pnl[~np.isnan(pnl)]
        out[d] = {
            "n": int(len(pnl)),
            "mean": round(float(pnl.mean()), 2) if len(pnl) else None,
            "tot": round(float(pnl.sum()), 1) if len(pnl) else 0,
            "win": round(float((pnl > 0).mean()), 3) if len(pnl) else None,
        }
    return out


def main() -> None:
    prices = load()
    all_grids = []
    for sym in SYMBOLS:
        g = prices[prices["product"] == sym]
        all_grids.append(grid_for(g, sym))
    grid = pd.concat(all_grids, ignore_index=True)
    grid.to_csv(OUT / "translator_grid.csv", index=False)

    # Filter profitable rules
    profitable = grid[(grid["mean"] > 0) & (grid["n"] >= 100)].copy()
    profitable = profitable.sort_values("tot", ascending=False)
    print("\n=== Translator profitable rules (n>=100, mean>0 after spread cost) ===")
    print(profitable.head(20).to_string(index=False))

    # Stability per day for top 6
    print("\n=== Per-day stability for top 6 translator rules ===")
    rows = []
    for _, r in profitable.head(6).iterrows():
        g_sym = prices[prices["product"] == r["symbol"]]
        spread = float((g_sym["ask_price_1"] - g_sym["bid_price_1"]).median())
        st = per_day_stability(g_sym, r["symbol"], int(r["pct"]), int(r["h"]),
                               r["side"], spread)
        rec = {
            "symbol": r["symbol"], "side": r["side"], "pct": r["pct"], "h": r["h"],
            "all_tot": r["tot"], "all_n": r["n"], "all_win": r["win"],
        }
        for d in DAYS:
            d_data = st.get(d, {})
            rec[f"d{d}_tot"] = d_data.get("tot")
            rec[f"d{d}_n"] = d_data.get("n")
            rec[f"d{d}_mean"] = d_data.get("mean")
            rec[f"d{d}_win"] = d_data.get("win")
        rows.append(rec)
    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.to_csv(OUT / "translator_top_stability.csv", index=False)


if __name__ == "__main__":
    main()
