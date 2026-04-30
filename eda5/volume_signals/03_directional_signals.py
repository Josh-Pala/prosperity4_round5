"""
03 - Directional / asymmetric signals.

Approfondimento dei pattern emersi:

1. Per i top candidati, separa BIG_BID e BIG_ASK e vedi se il segnale è
   davvero direzionale (es. big bid solo → up?) o non-direzionale.

2. Testa il segnale BIG_BID_AND_NOT_BIG_ASK e viceversa.

3. Misura l'edge in basis-point sul mid e in 'XIRECs per trade ipotetico':
       edge = abs(mean_fut_ret) * shares ipotetiche
   Per ogni segnale stampa anche la frequenza (n_on / n_total).

4. Costruisce una *trading rule* semplice e calcola PnL backtest naive:
       quando segnale ON, apri 1 contratto in direzione predetta,
       chiudi dopo h tick.
   Cap: 10 contratti (pos limit). Slippage: paga half-spread = 1.

Output: top trading rules per Sharpe (mean/std del PnL per evento).
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/volume_signals")
DAYS = [2, 3, 4]
HORIZONS = [5, 10, 30, 100]


def load_prices() -> pd.DataFrame:
    frames = [pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";") for d in DAYS]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return df


def half_spread(row: pd.Series) -> float:
    bp = row.get("bid_price_1")
    ap = row.get("ask_price_1")
    if pd.isna(bp) or pd.isna(ap):
        return np.nan
    return (ap - bp) / 2


def evaluate_rule(g: pd.DataFrame, mask: np.ndarray, h: int, direction: int,
                  cost_per_round_trip: float) -> dict:
    """direction: +1 long, -1 short."""
    g = g.copy()
    fut = g["mid_price"].shift(-h) - g["mid_price"]
    pnl_per_event = direction * fut[mask].values - cost_per_round_trip
    pnl_per_event = pnl_per_event[~np.isnan(pnl_per_event)]
    if len(pnl_per_event) < 30:
        return {}
    sharpe = pnl_per_event.mean() / pnl_per_event.std(ddof=1) if pnl_per_event.std() > 0 else 0.0
    return {
        "n_events": len(pnl_per_event),
        "mean_pnl": round(float(pnl_per_event.mean()), 3),
        "tot_pnl": round(float(pnl_per_event.sum()), 1),
        "sharpe_per_event": round(float(sharpe), 3),
        "win_rate": round(float((pnl_per_event > 0).mean()), 3),
    }


def analyze_symbol(g: pd.DataFrame) -> list[dict]:
    g = g.copy()
    g["bid_tot"] = g[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].fillna(0).sum(axis=1)
    g["ask_tot"] = g[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].fillna(0).sum(axis=1)
    g["imbal"] = (g["bid_tot"] - g["ask_tot"]) / (g["bid_tot"] + g["ask_tot"]).replace(0, np.nan)
    g["spread"] = g["ask_price_1"] - g["bid_price_1"]
    half_sp = float(g["spread"].median()) / 2.0  # cost per leg
    cost = 2 * half_sp  # round trip = 2 half-spreads (entry + exit aggressively)

    bv1_p95 = g["bid_volume_1"].dropna().quantile(0.95)
    av1_p95 = g["ask_volume_1"].dropna().quantile(0.95)
    bv1_p99 = g["bid_volume_1"].dropna().quantile(0.99)
    av1_p99 = g["ask_volume_1"].dropna().quantile(0.99)

    big_bid = g["bid_volume_1"] >= bv1_p95
    big_ask = g["ask_volume_1"] >= av1_p95
    big_bid_only = big_bid & ~big_ask
    big_ask_only = big_ask & ~big_bid
    big_both = big_bid & big_ask
    big_bid99 = g["bid_volume_1"] >= bv1_p99
    big_ask99 = g["ask_volume_1"] >= av1_p99

    imb_long = g["imbal"] > 0.4
    imb_short = g["imbal"] < -0.4

    rules = {
        "BIG_BID_ONLY_long": (big_bid_only.values, +1),
        "BIG_BID_ONLY_short": (big_bid_only.values, -1),
        "BIG_ASK_ONLY_long": (big_ask_only.values, +1),
        "BIG_ASK_ONLY_short": (big_ask_only.values, -1),
        "BIG_BOTH_long": (big_both.values, +1),
        "BIG_BOTH_short": (big_both.values, -1),
        "BIG_BID99_long": (big_bid99.values, +1),
        "BIG_ASK99_short": (big_ask99.values, -1),
        "IMBAL>0.4_long": (imb_long.values, +1),
        "IMBAL<-0.4_short": (imb_short.values, -1),
    }

    rows = []
    for name, (mask, d) in rules.items():
        for h in HORIZONS:
            r = evaluate_rule(g, mask, h, d, cost)
            if r:
                r.update({"rule": name, "horizon": h, "cost": round(cost, 2),
                          "spread_med": round(2 * half_sp, 2)})
                rows.append(r)
    return rows


def main() -> None:
    prices = load_prices()
    all_rows = []
    for sym, g in prices.groupby("product", sort=False):
        for r in analyze_symbol(g):
            r["symbol"] = sym
            all_rows.append(r)
    df = pd.DataFrame(all_rows)
    df = df[["symbol", "rule", "horizon", "n_events", "mean_pnl", "tot_pnl",
             "sharpe_per_event", "win_rate", "cost", "spread_med"]]
    df.to_csv(OUT / "trading_rules_pnl.csv", index=False)

    # Filtra: almeno 100 eventi, mean_pnl > 0
    profitable = df[(df["n_events"] >= 100) & (df["mean_pnl"] > 0)].copy()
    profitable = profitable.sort_values("tot_pnl", ascending=False)
    print("\n=== Top 30 profitable rules (n_events>=100, after spread cost) ===")
    print(profitable.head(30).to_string(index=False))

    # Per simbolo: best rule
    if not profitable.empty:
        best_per_sym = profitable.sort_values("tot_pnl", ascending=False).groupby("symbol").head(1)
        print("\n=== Best profitable rule per symbol (top 20 by tot_pnl) ===")
        print(best_per_sym.head(20).to_string(index=False))
        best_per_sym.to_csv(OUT / "best_rule_per_symbol.csv", index=False)


if __name__ == "__main__":
    main()
