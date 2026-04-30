"""MICROCHIP pair sweep: explore all pairs × signal types × entry_z thresholds.

Goal: find pair set that beats the current 3-pair (TRIANGLE-anchored) baseline.

Signal types tested:
  - spread:  m_a - m_b
  - sum:     m_a + m_b
  - product: m_a * m_b / 1e4   (normalized)
  - ratio:   m_a / m_b

For each (a, b, signal_type, entry_z) we run a simple z-score mean-reversion
simulation matching the live trader's logic:
  - WARMUP=500 ticks Welford accumulation
  - When |z| > entry_z & flat -> enter (-1 if z>0, +1 if z<0)
  - When |z| < EXIT_Z & in pos -> flatten
  - Position size: SIZE=10, clipped to LIMIT=10 per leg
  - Aggressive entry: cross spread at best_ask (buy) / best_bid (sell)

PnL is naive: assume fills at best_ask for buys, best_bid for sells, no fees,
no liquidity cap. This is a *relative* ranking tool, not absolute PnL.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

DATA_DIR = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
DAYS = [2, 3, 4]
SYMBOLS = [
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
]

WARMUP = 500
EXIT_Z = 0.3
SIZE = 10
LIMIT = 10
ENTRY_ZS = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
SIGNAL_TYPES = ["spread", "sum", "product", "ratio"]


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df = df[df["product"].isin(SYMBOLS)]
        frames.append(df[["day", "timestamp", "product", "bid_price_1", "ask_price_1", "mid_price"]])
    return pd.concat(frames, ignore_index=True)


def pivot_book(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {symbol: DataFrame indexed by (day, timestamp) with mid/best_bid/best_ask}."""
    out = {}
    for s in SYMBOLS:
        sub = df[df["product"] == s].copy()
        sub = sub.set_index(["day", "timestamp"]).sort_index()
        out[s] = sub[["mid_price", "bid_price_1", "ask_price_1"]].rename(
            columns={"bid_price_1": "bb", "ask_price_1": "ba", "mid_price": "mid"}
        )
    return out


def signal(ma: float, mb: float, kind: str) -> float:
    if kind == "spread":
        return ma - mb
    if kind == "sum":
        return ma + mb
    if kind == "product":
        return ma * mb / 1e4
    if kind == "ratio":
        return ma / mb if mb != 0 else 0.0
    raise ValueError(kind)


def simulate(books: dict[str, pd.DataFrame], a: str, b: str, kind: str, entry_z: float):
    """Run mean-reversion sim. Returns (pnl, n_trades, n_entries)."""
    da = books[a]
    db = books[b]
    common = da.index.intersection(db.index)
    da = da.loc[common]
    db = db.loc[common]

    n_obs = len(common)
    if n_obs < WARMUP + 100:
        return 0.0, 0, 0

    n_seen = 0
    mean = 0.0
    m2 = 0.0
    target = 0  # -1, 0, +1
    pos_a = 0
    pos_b = 0
    cash = 0.0
    n_entries = 0
    n_trades = 0

    mid_a_arr = da["mid"].values
    mid_b_arr = db["mid"].values
    bb_a = da["bb"].values
    ba_a = da["ba"].values
    bb_b = db["bb"].values
    ba_b = db["ba"].values

    for i in range(n_obs):
        ma = mid_a_arr[i]
        mb = mid_b_arr[i]
        sig = signal(ma, mb, kind)

        n_seen += 1
        delta = sig - mean
        mean += delta / n_seen
        m2 += delta * (sig - mean)

        if n_seen < WARMUP:
            continue
        var = m2 / (n_seen - 1) if n_seen > 1 else 0.0
        sd = var ** 0.5
        if sd <= 1e-9:
            continue
        z = (sig - mean) / sd

        new_target = target
        if target == 0:
            if z > entry_z:
                new_target = -1
            elif z < -entry_z:
                new_target = +1
        else:
            if abs(z) < EXIT_Z:
                new_target = 0

        if new_target != target:
            # Determine target leg positions for new_target
            ta = SIZE * new_target
            # For spread: tb = -ta; for sum: tb = +ta;
            # For product/ratio we treat like sum (both legs move together to be balanced).
            if kind == "spread":
                tb = -ta
            elif kind == "sum":
                tb = ta
            elif kind == "product":
                # Product mean reversion: if signal too high, both legs likely high -> short both
                tb = ta
            elif kind == "ratio":
                # Ratio mean reversion: signal high -> a too high vs b -> short a, long b
                tb = -ta
            else:
                tb = -ta

            # Clip to limit
            ta = max(-LIMIT, min(LIMIT, ta))
            tb = max(-LIMIT, min(LIMIT, tb))

            # Trade leg A
            d_a = ta - pos_a
            if d_a > 0:
                cash -= ba_a[i] * d_a  # buy at ask
                n_trades += 1
            elif d_a < 0:
                cash -= bb_a[i] * d_a  # sell at bid (d_a negative)
                n_trades += 1
            pos_a = ta

            # Trade leg B
            d_b = tb - pos_b
            if d_b > 0:
                cash -= ba_b[i] * d_b
                n_trades += 1
            elif d_b < 0:
                cash -= bb_b[i] * d_b
                n_trades += 1
            pos_b = tb

            if target == 0 and new_target != 0:
                n_entries += 1
            target = new_target

    # Mark-to-market at last mid
    final_pnl = cash + pos_a * mid_a_arr[-1] + pos_b * mid_b_arr[-1]
    return final_pnl, n_trades, n_entries


def main():
    print("Loading data...")
    df = load_prices()
    books = pivot_book(df)
    print(f"Loaded {len(df)} rows, {len(books)} symbols")

    pairs = list(itertools.combinations(SYMBOLS, 2))
    print(f"\nSweeping {len(pairs)} pairs × {len(SIGNAL_TYPES)} signal types × {len(ENTRY_ZS)} z thresholds = {len(pairs)*len(SIGNAL_TYPES)*len(ENTRY_ZS)} configs\n")

    rows = []
    for a, b in pairs:
        for kind in SIGNAL_TYPES:
            for ez in ENTRY_ZS:
                pnl, ntr, nen = simulate(books, a, b, kind, ez)
                rows.append({
                    "a": a.replace("MICROCHIP_", ""),
                    "b": b.replace("MICROCHIP_", ""),
                    "kind": kind,
                    "entry_z": ez,
                    "pnl": pnl,
                    "trades": ntr,
                    "entries": nen,
                })

    res = pd.DataFrame(rows).sort_values("pnl", ascending=False)

    print("=== TOP 30 BY PnL (all signal types) ===")
    print(res.head(30).to_string(index=False))

    print("\n=== BEST PER PAIR (max over signal_type × entry_z) ===")
    best_per_pair = res.loc[res.groupby(["a", "b"])["pnl"].idxmax()].sort_values("pnl", ascending=False)
    print(best_per_pair.to_string(index=False))

    print("\n=== BASELINE (current FINAL_GLAUCO config) ===")
    baseline_specs = [
        ("OVAL", "TRIANGLE", "spread", 1.2),
        ("CIRCLE", "TRIANGLE", "sum", 1.5),
        ("RECTANGLE", "TRIANGLE", "spread", 2.0),
    ]
    total = 0.0
    for a, b, k, z in baseline_specs:
        m = res[(res["a"] == a) & (res["b"] == b) & (res["kind"] == k) & (res["entry_z"] == z)]
        if not m.empty:
            r = m.iloc[0]
            print(f"  {a:10s} {k:7s} {b:10s} z={z}: pnl={r['pnl']:>10.0f}  trades={r['trades']}  entries={r['entries']}")
            total += r["pnl"]
    print(f"  TOTAL baseline: {total:.0f}")

    out_path = Path(__file__).parent / "microchip_pair_sweep_results.csv"
    res.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
