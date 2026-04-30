"""
SLEEP_POD pair scanner — find new mean-reverting combinations.

Tests for each unordered pair (a, b) of SLEEP_POD variants:
  - a + b           (sum)
  - a - b           (spread, unit beta)
  - a - beta*b      (OLS-scaled spread)
  - a / b           (ratio)
  - log(a) - log(b) (log-ratio, multiplicative)
  - a * b           (product, sanity check)

For each signal we compute:
  - half_life: AR(1)-implied mean-reversion half-life on z-score residuals
  - adf_p: ADF-style p-value via statsmodels if available, else None
  - cross_rate: zero-crossings of (sig - mean) per 1000 ticks
  - sharpe_sim: rough Sharpe of a z-score strategy:
        enter -1 at z>+2, +1 at z<-2, exit at |z|<0.3, hold position,
        PnL = -position_change * sig (proxy: trading at mid).
  - n_trades_sim, total_pnl_sim
  - corr_with_existing: correlation with our current 1 pair (POLYESTER - COTTON)

Output: eda5/sleep_pod/pair_scan_results.csv ranked by sharpe_sim.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent
SLEEP_POD = [
    "SLEEP_POD_COTTON",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_SUEDE",
]

# Z-score strategy parameters
ENTRY_Z = 1.8
EXIT_Z = 0.3
WARMUP = 500
# Rolling window for z-score (instead of running stats — captures regime drift)
ROLL_WIN = 2000


def load_mids() -> pd.DataFrame:
    frames = []
    for day in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{day}.csv", sep=";")
        df = df[df["product"].isin(SLEEP_POD)][["timestamp", "product", "mid_price"]]
        df["day"] = day
        df["t"] = df["day"] * 1_000_000 + df["timestamp"]
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index="t", columns="product", values="mid_price").sort_index()
    wide = wide.dropna()
    return wide


def half_life(series: pd.Series) -> float:
    """AR(1) half-life of mean-reversion. Returns inf if not mean-reverting."""
    s = series.dropna()
    if len(s) < 100:
        return float("inf")
    s_lag = s.shift(1).dropna()
    s_diff = (s - s.shift(1)).dropna()
    s_lag = s_lag.loc[s_diff.index]
    x = s_lag.values - s_lag.mean()
    y = s_diff.values
    denom = (x * x).sum()
    if denom <= 0:
        return float("inf")
    beta = (x * y).sum() / denom
    if beta >= 0:
        return float("inf")
    return -math.log(2) / math.log(1 + beta)


def cross_rate(series: pd.Series) -> float:
    s = series - series.mean()
    signs = np.sign(s.values)
    crosses = (signs[1:] * signs[:-1] < 0).sum()
    return crosses / max(1, len(s)) * 1000.0


def simulate_zscore_pnl(sig: pd.Series, entry_z: float = ENTRY_Z,
                        exit_z: float = EXIT_Z, roll_win: int = ROLL_WIN):
    """
    Simulate z-score strategy with ROLLING mean/std (window=roll_win).
    This better matches a regime-tracking strategy and gives more trades
    than running stats over a 30k-tick dataset.
    PnL proxy: sum over trades of (entry - exit) * direction.
    """
    n = len(sig)
    if n < roll_win + 200:
        return 0.0, 0, 0.0
    s = sig.rolling(roll_win)
    mean = s.mean()
    sd = s.std()
    z = (sig - mean) / sd.replace(0, np.nan)
    z = z.fillna(0).values
    vals = sig.values
    pos = 0
    last_entry_price = 0.0
    pnls = []
    for i in range(roll_win, n):
        zi = z[i]
        new_pos = pos
        if pos == 0:
            if zi > entry_z:
                new_pos = -1
            elif zi < -entry_z:
                new_pos = +1
        else:
            if abs(zi) < exit_z:
                new_pos = 0
        if new_pos != pos:
            if pos == 0:
                last_entry_price = vals[i]
            else:
                trade_pnl = pos * (last_entry_price - vals[i])
                pnls.append(trade_pnl)
            pos = new_pos
    if not pnls:
        return 0.0, 0, 0.0
    arr = np.array(pnls)
    pnl = arr.sum()
    sharpe = arr.mean() / arr.std() * math.sqrt(len(arr)) if arr.std() > 0 else 0.0
    return pnl, len(arr), sharpe


def ols_beta(y: pd.Series, x: pd.Series) -> float:
    yv = y.values - y.mean()
    xv = x.values - x.mean()
    denom = (xv * xv).sum()
    return (xv * yv).sum() / denom if denom > 0 else 0.0


def evaluate_signal(name: str, sig: pd.Series, baseline: pd.Series | None) -> dict:
    sig = sig.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sig) < 1000:
        return {}
    hl = half_life(sig)
    cr = cross_rate(sig)
    pnl, n_tr, sharpe = simulate_zscore_pnl(sig)
    corr_base = float(sig.corr(baseline.loc[sig.index])) if baseline is not None else 0.0
    return {
        "signal": name,
        "n": len(sig),
        "half_life": hl,
        "cross_rate_per_1k": cr,
        "n_trades_sim": n_tr,
        "total_pnl_sim": pnl,
        "sharpe_sim": sharpe,
        "corr_with_baseline": corr_base,
        "sd": float(sig.std()),
    }


def main():
    print("Loading mids...")
    mids = load_mids()
    print(f"Loaded {len(mids)} ticks across {len(mids.columns)} symbols")

    # Baseline = current production pair (POLYESTER - COTTON)
    baseline = mids["SLEEP_POD_POLYESTER"] - mids["SLEEP_POD_COTTON"]

    rows = []
    pairs = list(itertools.combinations(SLEEP_POD, 2))
    print(f"Scanning {len(pairs)} unordered pairs across 6 transforms each...")

    for a, b in pairs:
        sa, sb = mids[a], mids[b]
        # 1. sum
        rows.append({"a": a, "b": b, "form": "sum",
                     **evaluate_signal(f"{a}+{b}", sa + sb, baseline)})
        # 2. spread (unit beta)
        rows.append({"a": a, "b": b, "form": "spread",
                     **evaluate_signal(f"{a}-{b}", sa - sb, baseline)})
        # 3. OLS-scaled spread
        beta = ols_beta(sa, sb)
        rows.append({"a": a, "b": b, "form": f"ols_spread(beta={beta:.3f})",
                     **evaluate_signal(f"{a}-{beta:.3f}*{b}", sa - beta * sb, baseline)})
        # 4. ratio
        rows.append({"a": a, "b": b, "form": "ratio",
                     **evaluate_signal(f"{a}/{b}", sa / sb, baseline)})
        # 5. log-ratio
        rows.append({"a": a, "b": b, "form": "log_ratio",
                     **evaluate_signal(f"log({a})-log({b})",
                                       np.log(sa) - np.log(sb), baseline)})
        # 6. product (sanity)
        rows.append({"a": a, "b": b, "form": "product",
                     **evaluate_signal(f"{a}*{b}", sa * sb, baseline)})

    # Also a few triples — sums and weighted combos worth trying
    triples = [
        ("SLEEP_POD_COTTON", "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"),
        ("SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON", "SLEEP_POD_SUEDE"),
        ("SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON"),
    ]
    for a, b, c in triples:
        sa, sb, sc = mids[a], mids[b], mids[c]
        rows.append({"a": a, "b": f"{b}+{c}", "form": "triple_sum",
                     **evaluate_signal(f"{a}-({b}+{c})/2",
                                       sa - (sb + sc) / 2, baseline)})
        rows.append({"a": f"{a}+{b}", "b": c, "form": "triple_sum_alt",
                     **evaluate_signal(f"({a}+{b})/2-{c}",
                                       (sa + sb) / 2 - sc, baseline)})

    df = pd.DataFrame([r for r in rows if r])
    # Drop product form (scale artifact: PnL inflated by 10k * 10k)
    df = df[df["form"] != "product"].copy()

    out_path = OUT_DIR / "pair_scan_results.csv"
    df.sort_values("total_pnl_sim", ascending=False).to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

    cols = ["signal", "form", "half_life", "n_trades_sim",
            "total_pnl_sim", "sharpe_sim", "corr_with_baseline"]

    print("\n=== TOP 15 by total simulated PnL ===")
    print(df.sort_values("total_pnl_sim", ascending=False)
          .head(15)[cols].to_string(index=False))

    print("\n=== TOP 15 by Sharpe (min 10 trades) ===")
    df_s = df[df["n_trades_sim"] >= 10].sort_values("sharpe_sim", ascending=False)
    print(df_s.head(15)[cols].to_string(index=False))

    print("\n=== Baseline (POLYESTER-COTTON spread) for reference ===")
    base_row = df[(df["a"] == "SLEEP_POD_COTTON") &
                  (df["b"] == "SLEEP_POD_POLYESTER") &
                  (df["form"] == "spread")]
    print(base_row[cols].to_string(index=False))


if __name__ == "__main__":
    main()
