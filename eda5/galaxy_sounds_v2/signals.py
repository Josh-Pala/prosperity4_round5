"""GALAXY_SOUNDS — Signal validation.

Goals:
1. Trade flow analysis (corretta, con merge per symbol).
2. Momentum/trend signal: confirm Hurst ~1 → look-back vs forward returns.
3. OFI (Order Flow Imbalance) intra-tick.
4. Best-of-book imbalance (bid_vol vs ask_vol).
5. Combined signal backtest (size up to 10).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/galaxy_sounds_v2")
GS = ["GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
      "GALAXY_SOUNDS_SOLAR_WINDS"]


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


def trade_flow_correct(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Join trades to prices on (day, timestamp, symbol)."""
    px = prices.rename(columns={"product": "symbol"})[
        ["day", "timestamp", "symbol", "bid_price_1", "ask_price_1", "mid_price"]
    ]
    j = trades.merge(px, on=["day", "timestamp", "symbol"], how="left")
    # Lee-Ready: above mid -> buyer init (+1), below -> seller (-1)
    sgn = np.sign(j["price"] - j["mid_price"]).fillna(0)
    j["sgn"] = sgn
    j["sgn_qty"] = sgn * j["quantity"]

    rows = []
    for sym in GS:
        for d in (2, 3, 4):
            sub = j[(j["symbol"] == sym) & (j["day"] == d)]
            if sub.empty:
                continue
            rows.append({
                "sym": sym.replace("GALAXY_SOUNDS_", ""),
                "day": d,
                "n_trades": len(sub),
                "qty_total": int(sub["quantity"].sum()),
                "qty_buy": int(sub.loc[sub["sgn"] > 0, "quantity"].sum()),
                "qty_sell": int(sub.loc[sub["sgn"] < 0, "quantity"].sum()),
                "qty_mid": int(sub.loc[sub["sgn"] == 0, "quantity"].sum()),
                "net_signed": int(sub["sgn_qty"].sum()),
                "mean_sz": float(sub["quantity"].mean()),
                "max_sz": int(sub["quantity"].max()),
                "p90_sz": float(sub["quantity"].quantile(0.9)),
            })
    return pd.DataFrame(rows)


def momentum_validation(prices: pd.DataFrame) -> pd.DataFrame:
    """For each look-back N, compute per-day Sharpe of strategy:
    long if mid_t > mid_(t-N), short otherwise; hold M ticks; size 1.
    Pure directional bet on mid (no execution friction)."""
    rows = []
    for sym in GS:
        for d in (2, 3, 4):
            g = prices[(prices["product"] == sym) & (prices["day"] == d)].sort_values("timestamp").reset_index(drop=True)
            mid = g["mid_price"].to_numpy()
            for N in (5, 10, 20, 50, 100):
                for M in (1, 5, 10, 20):
                    if N + M >= len(mid):
                        continue
                    # signal at t: sign(mid[t] - mid[t-N])
                    sig = np.sign(mid[N:] - mid[:-N])  # length = len-N
                    # holding return: mid[t+M] - mid[t]
                    fwd = mid[N + M:] - mid[N:-M]  # align with sig (drop last M)
                    sig = sig[: len(fwd)]
                    if len(fwd) < 100:
                        continue
                    pnl = sig * fwd
                    sh = pnl.mean() / pnl.std() * np.sqrt(len(pnl)) if pnl.std() > 0 else 0
                    wr = float((pnl > 0).mean())
                    total = float(pnl.sum())  # size 1 PnL across the day
                    rows.append({
                        "sym": sym.replace("GALAXY_SOUNDS_", ""),
                        "day": d,
                        "N": N,
                        "M": M,
                        "n_signals": len(pnl),
                        "win_rate": wr,
                        "mean_pnl": float(pnl.mean()),
                        "sharpe": float(sh),
                        "total_pnl_size1": total,
                    })
    return pd.DataFrame(rows)


def book_imbalance_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """At each tick, imbalance = (bid_v1 - ask_v1) / (bid_v1 + ask_v1).
    Test if it predicts mid move over next M ticks."""
    rows = []
    for sym in GS:
        for d in (2, 3, 4):
            g = prices[(prices["product"] == sym) & (prices["day"] == d)].sort_values("timestamp").reset_index(drop=True)
            mid = g["mid_price"].to_numpy()
            bv = g["bid_volume_1"].to_numpy().astype(float)
            av = g["ask_volume_1"].to_numpy().astype(float)
            imb = (bv - av) / np.maximum(bv + av, 1.0)
            for M in (1, 5, 10):
                if M >= len(mid):
                    continue
                fwd = mid[M:] - mid[:-M]
                im = imb[:-M]
                # Pearson corr
                if im.std() == 0 or fwd.std() == 0:
                    continue
                r = float(np.corrcoef(im, fwd)[0, 1])
                # naive: long if imb>0 else short
                sig = np.sign(im)
                pnl = sig * fwd
                sh = pnl.mean() / pnl.std() * np.sqrt(len(pnl)) if pnl.std() > 0 else 0
                rows.append({
                    "sym": sym.replace("GALAXY_SOUNDS_", ""),
                    "day": d,
                    "M": M,
                    "corr_imb_fwdret": r,
                    "win_rate": float((pnl > 0).mean()),
                    "mean_pnl": float(pnl.mean()),
                    "sharpe": float(sh),
                    "total_pnl_size1": float(pnl.sum()),
                })
    return pd.DataFrame(rows)


def execution_aware_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """Trend-following with realistic execution:
    - Signal: sign(mid_t - mid_(t-N))
    - Entry: aggressive (cross spread), exit: aggressive
    - Cost: spread/2 entry + spread/2 exit
    - Hold M ticks, size = full LIMIT (10)
    - Compute true tradable PnL.
    """
    rows = []
    LIMIT = 10
    for sym in GS:
        for d in (2, 3, 4):
            g = prices[(prices["product"] == sym) & (prices["day"] == d)].sort_values("timestamp").reset_index(drop=True)
            bid = g["bid_price_1"].to_numpy()
            ask = g["ask_price_1"].to_numpy()
            mid = g["mid_price"].to_numpy()
            n = len(mid)
            for N in (10, 20, 50, 100, 200):
                for M in (5, 10, 20, 50, 100):
                    if N + M >= n:
                        continue
                    # state machine: at each step recompute, hold for M steps
                    pnl_sum = 0.0
                    n_trades = 0
                    pos = 0
                    entry_price = 0.0
                    hold_left = 0
                    for t in range(N, n - 1):
                        if hold_left > 0:
                            hold_left -= 1
                            if hold_left == 0 and pos != 0:
                                # exit aggressive
                                exit_p = bid[t] if pos > 0 else ask[t]
                                pnl_sum += pos * (exit_p - entry_price)
                                pos = 0
                            continue
                        sig = 1 if mid[t] > mid[t - N] else (-1 if mid[t] < mid[t - N] else 0)
                        if sig != 0:
                            pos = LIMIT * sig
                            entry_price = ask[t] if sig > 0 else bid[t]
                            hold_left = M
                            n_trades += 1
                    rows.append({
                        "sym": sym.replace("GALAXY_SOUNDS_", ""),
                        "day": d,
                        "N": N,
                        "M": M,
                        "n_trades": n_trades,
                        "pnl": pnl_sum,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("Loading...")
    prices = load_prices()
    trades = load_trades()

    print("\n=== Trade flow (correct) ===")
    tf = trade_flow_correct(prices, trades)
    tf.to_csv(OUT / "trade_flow_correct.csv", index=False)
    print(tf.to_string(index=False))

    print("\n=== Momentum signal (frictionless mid-to-mid) ===")
    mm = momentum_validation(prices)
    mm.to_csv(OUT / "momentum.csv", index=False)
    # Best per symbol+day
    print("\nBest (N,M) per symbol+day by sharpe:")
    best = mm.sort_values(["sym", "day", "sharpe"], ascending=[True, True, False]).groupby(["sym", "day"]).head(3)
    print(best.to_string(index=False))
    print("\nGrand total of frictionless momentum (size 1 → multiply ×10 for LIMIT):")
    total_by_NM = mm.groupby(["N", "M"])["total_pnl_size1"].sum().sort_values(ascending=False).head(15)
    print(total_by_NM)

    print("\n=== Book imbalance signal ===")
    bi = book_imbalance_signal(prices)
    bi.to_csv(OUT / "imbalance.csv", index=False)
    print(bi.to_string(index=False))

    print("\n=== Execution-aware momentum (LIMIT=10, aggressive entry+exit) ===")
    em = execution_aware_momentum(prices)
    em.to_csv(OUT / "exec_momentum.csv", index=False)
    print("\nTotal 3-day PnL by (N, M):")
    total = em.groupby(["N", "M"])["pnl"].sum().reset_index().sort_values("pnl", ascending=False)
    print(total.head(15).to_string(index=False))
    print("\nBest config 3-day total per symbol:")
    by_sym = em.groupby(["sym", "N", "M"])["pnl"].sum().reset_index()
    for sym in by_sym["sym"].unique():
        sub = by_sym[by_sym["sym"] == sym].sort_values("pnl", ascending=False).head(3)
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
