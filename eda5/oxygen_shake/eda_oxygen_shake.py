"""
OXYGEN_SHAKE descriptive EDA — does the family have any pair-trading edge?

Outputs (in this folder):
  - descriptives.csv          per-symbol mean / sd / spread / drift stats
  - sum_check.csv             does sum(OXY_*) hold an invariant like PEBBLES?
  - corr_levels.csv           pairwise Pearson on mid levels
  - corr_returns_1.csv        pairwise Pearson on 1-tick log returns
  - corr_returns_10.csv       pairwise Pearson on 10-tick log returns
  - spread_stats.csv          for each (a, b) pair: mean, sd, half-life of (a-b)
                              and (a+b) plus full-sample OLS beta
  - prices_norm.png           normalised mids (price / day-1 mean)
  - rolling_corr.png          rolling-1000-tick corr for top 3 abs-corr pairs
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent

OXY = [
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_MORNING_BREATH",
]


def load_mids() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for day in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{day}.csv", sep=";")
        df = df[df["product"].isin(OXY)][
            ["timestamp", "product", "mid_price", "bid_price_1", "ask_price_1"]
        ]
        df["t"] = day * 1_000_000 + df["timestamp"]
        df["day"] = day
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    mids = (
        long.pivot_table(index="t", columns="product", values="mid_price")
        .sort_index()
        .dropna()
    )
    spreads = long.assign(spread=long.ask_price_1 - long.bid_price_1)
    return mids, spreads


def half_life(x: pd.Series) -> float:
    """OLS estimate of the AR(1) half-life of x (mean reversion)."""
    x = x.dropna()
    if len(x) < 50:
        return float("nan")
    dx = x.diff().dropna()
    lag = x.shift(1).loc[dx.index]
    A = np.column_stack([np.ones(len(lag)), lag.values - lag.mean()])
    y = dx.values
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    phi = coef[1]
    if phi >= 0 or phi <= -1:
        return float("inf")
    return float(-math.log(2) / math.log(1 + phi))


def descriptives(mids: pd.DataFrame, spreads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym in OXY:
        s = mids[sym]
        bbo = spreads.loc[spreads["product"] == sym, "spread"].dropna()
        ret = s.diff().dropna()
        rows.append({
            "symbol": sym,
            "mid_mean": round(s.mean(), 1),
            "mid_sd": round(s.std(), 1),
            "mid_min": round(s.min(), 1),
            "mid_max": round(s.max(), 1),
            "abs_drift_3d": round(s.iloc[-1] - s.iloc[0], 1),
            "ret1_sd": round(ret.std(), 2),
            "ret1_abs_mean": round(ret.abs().mean(), 2),
            "best_spread_mean": round(bbo.mean(), 2),
            "best_spread_p50": round(bbo.median(), 2),
            "best_spread_p90": round(bbo.quantile(0.9), 2),
            "half_life_mid": round(half_life(s), 1),
        })
    return pd.DataFrame(rows)


def sum_check(mids: pd.DataFrame) -> pd.DataFrame:
    """Mirror of PEBBLES sum-invariant check."""
    s = mids.sum(axis=1)
    return pd.DataFrame([{
        "sum_mean": round(s.mean(), 1),
        "sum_sd": round(s.std(), 2),
        "sum_min": round(s.min(), 1),
        "sum_max": round(s.max(), 1),
        "sum_range_pct": round(100 * (s.max() - s.min()) / s.mean(), 4),
    }])


def pair_correlations(mids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    levels = mids.corr().round(4)
    ret1 = mids.diff().dropna().corr().round(4)
    ret10 = mids.diff(10).dropna().corr().round(4)
    return levels, ret1, ret10


def spread_stats(mids: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(OXY, 2):
        spread = mids[a] - mids[b]
        ssum = mids[a] + mids[b]
        # OLS beta of a on b
        bv = mids[b].values - mids[b].mean()
        av = mids[a].values - mids[a].mean()
        denom = (bv * bv).sum()
        beta = float((bv * av).sum() / denom) if denom > 0 else 0.0
        ols_resid = mids[a] - beta * mids[b]
        rows.append({
            "a": a,
            "b": b,
            "corr_levels": round(mids[a].corr(mids[b]), 4),
            "corr_ret1": round(mids[a].diff().corr(mids[b].diff()), 4),
            "corr_ret10": round(mids[a].diff(10).corr(mids[b].diff(10)), 4),
            "spread_mean": round(spread.mean(), 1),
            "spread_sd": round(spread.std(), 2),
            "spread_hl": round(half_life(spread), 1),
            "sum_mean": round(ssum.mean(), 1),
            "sum_sd": round(ssum.std(), 2),
            "sum_hl": round(half_life(ssum), 1),
            "ols_beta_a_on_b": round(beta, 3),
            "ols_resid_sd": round(ols_resid.std(), 2),
            "ols_resid_hl": round(half_life(ols_resid), 1),
        })
    return pd.DataFrame(rows).sort_values("ols_resid_hl")


def plot_normalised(mids: pd.DataFrame) -> None:
    norm = mids.div(mids.iloc[0])
    fig, ax = plt.subplots(figsize=(11, 5))
    for sym in OXY:
        ax.plot(np.arange(len(norm)), norm[sym], label=sym.replace("OXYGEN_SHAKE_", ""), lw=0.6)
    ax.set_title("OXYGEN_SHAKE — normalised mids (start = 1.0)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("tick")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prices_norm.png", dpi=110)
    plt.close(fig)


def plot_rolling_corr(mids: pd.DataFrame, top_pairs: list[tuple[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    rets = mids.diff()
    for a, b in top_pairs:
        rc = rets[a].rolling(1000).corr(rets[b])
        ax.plot(np.arange(len(rc)), rc.values,
                label=f"{a.split('_')[-1]} vs {b.split('_')[-1]}", lw=0.7)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_title("OXYGEN_SHAKE — rolling 1000-tick return correlation (top abs-corr pairs)")
    ax.legend(fontsize=8)
    ax.set_xlabel("tick")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rolling_corr.png", dpi=110)
    plt.close(fig)


def main() -> None:
    print("Loading mids...")
    mids, spreads = load_mids()
    print(f"  ticks: {len(mids):>6}   symbols: {list(mids.columns)}")

    desc = descriptives(mids, spreads)
    desc.to_csv(OUT_DIR / "descriptives.csv", index=False)
    print("\n=== DESCRIPTIVES ===")
    print(desc.to_string(index=False))

    s = sum_check(mids)
    s.to_csv(OUT_DIR / "sum_check.csv", index=False)
    print("\n=== SUM(all 5 OXYGEN_SHAKE) — PEBBLES-style invariant check ===")
    print(s.to_string(index=False))

    cl, c1, c10 = pair_correlations(mids)
    cl.to_csv(OUT_DIR / "corr_levels.csv")
    c1.to_csv(OUT_DIR / "corr_returns_1.csv")
    c10.to_csv(OUT_DIR / "corr_returns_10.csv")
    print("\n=== CORR (1-tick returns) ===")
    print(c1.to_string())
    print("\n=== CORR (10-tick returns) ===")
    print(c10.to_string())

    ss = spread_stats(mids)
    ss.to_csv(OUT_DIR / "spread_stats.csv", index=False)
    print("\n=== PAIR SPREAD / SUM / OLS-RESID STATS (sorted by OLS-resid half-life) ===")
    print(ss.to_string(index=False))

    plot_normalised(mids)
    flat = c1.where(~np.eye(len(c1), dtype=bool)).abs().stack().sort_values(ascending=False)
    seen = set()
    top_pairs = []
    for (a, b), _ in flat.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        top_pairs.append((a, b))
        if len(top_pairs) == 3:
            break
    plot_rolling_corr(mids, top_pairs)
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
