"""
04 - Day-by-day stability check.

Per i top candidati, controlla che il segnale tenga in tutti e 3 i giorni.
Un segnale che funziona solo su 1 giorno e fallisce sugli altri è overfit.
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/volume_signals")
DAYS = [2, 3, 4]

CANDIDATES = [
    # (symbol, signal_kind, threshold_pctile, direction, horizon)
    ("ROBOT_IRONING", "BIG_BOTH", 95, -1, 100),
    ("MICROCHIP_OVAL", "BIG_BID_ONLY", 95, -1, 100),
    ("ROBOT_DISHES", "BIG_BOTH", 95, +1, 100),
    ("MICROCHIP_RECTANGLE", "BIG_BOTH", 95, +1, 100),
    ("TRANSLATOR_GRAPHITE_MIST", "BIG_BID_P99", 99, +1, 100),
    ("TRANSLATOR_ECLIPSE_CHARCOAL", "BIG_BID_ONLY", 95, +1, 100),
    ("TRANSLATOR_ASTRO_BLACK", "BIG_BID_P99", 99, +1, 100),
    ("PEBBLES_XS", "BIG_BOTH", 95, -1, 100),
    ("SLEEP_POD_NYLON", "BIG_BID_ONLY", 95, +1, 100),
    ("OXYGEN_SHAKE_GARLIC", "BIG_BOTH", 95, +1, 100),
    ("ROBOT_DISHES", "BIG_BOTH", 95, +1, 30),
    ("ROBOT_VACUUMING", "BIG_ASK_ONLY", 95, -1, 100),
]


def load_prices() -> pd.DataFrame:
    frames = [pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";") for d in DAYS]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return df


def evaluate(g_day: pd.DataFrame, kind: str, pct: int, direction: int, h: int,
             cost: float) -> dict:
    bv1 = g_day["bid_volume_1"].dropna()
    av1 = g_day["ask_volume_1"].dropna()
    if bv1.empty or av1.empty:
        return {}
    thr_b = bv1.quantile(pct / 100)
    thr_a = av1.quantile(pct / 100)
    big_b = g_day["bid_volume_1"] >= thr_b
    big_a = g_day["ask_volume_1"] >= thr_a

    if kind == "BIG_BOTH":
        mask = (big_b & big_a).values
    elif kind == "BIG_BID_ONLY":
        mask = (big_b & ~big_a).values
    elif kind == "BIG_ASK_ONLY":
        mask = (big_a & ~big_b).values
    elif kind == "BIG_BID_P99":
        mask = big_b.values
    elif kind == "BIG_ASK_P99":
        mask = big_a.values
    else:
        return {}

    fut = (g_day["mid_price"].shift(-h) - g_day["mid_price"]).values
    pnl = direction * fut[mask] - cost
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 20:
        return {"n": int(len(pnl)), "mean": np.nan, "tot": 0.0, "win": np.nan}
    return {
        "n": int(len(pnl)),
        "mean": round(float(pnl.mean()), 3),
        "tot": round(float(pnl.sum()), 1),
        "win": round(float((pnl > 0).mean()), 3),
    }


def main() -> None:
    prices = load_prices()
    rows = []
    for sym, kind, pct, direction, h in CANDIDATES:
        g_sym = prices[prices["product"] == sym]
        spread_med = (g_sym["ask_price_1"] - g_sym["bid_price_1"]).median()
        cost = float(spread_med)
        full = {"day": "ALL"} | evaluate(g_sym, kind, pct, direction, h, cost)
        per_day = {}
        for d in DAYS:
            r = evaluate(g_sym[g_sym["day"] == d], kind, pct, direction, h, cost)
            per_day[d] = r
        rows.append({
            "symbol": sym,
            "rule": f"{kind}_p{pct}_{'long' if direction>0 else 'short'}_h{h}",
            "spread": cost,
            "all_n": full.get("n"),
            "all_mean": full.get("mean"),
            "all_tot": full.get("tot"),
            "all_win": full.get("win"),
            **{f"d{d}_n": per_day[d].get("n") for d in DAYS},
            **{f"d{d}_mean": per_day[d].get("mean") for d in DAYS},
            **{f"d{d}_tot": per_day[d].get("tot") for d in DAYS},
            **{f"d{d}_win": per_day[d].get("win") for d in DAYS},
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stability_per_day.csv", index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
