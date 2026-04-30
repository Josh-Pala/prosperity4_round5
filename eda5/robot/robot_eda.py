"""ROBOT family — targeted EDA.

Goals:
  1. Levels & spreads: mean / vol / quoted spread per symbol.
  2. Constant-sum invariant test: Σ mid_robot ≈ const? Best linear combo?
  3. Lead-lag: cross-correlation of mid returns at lags ±1..5 ticks.
  4. Cointegration / pair structure: residual std for all spread / sum pairs.
  5. Z-score regimes: how many crossings of |z|>{1.5,2.0,2.5} per pair (proxy
     of expected pair-trading opportunities).
  6. Compare current pair selection vs all C(5,2)*2 combos.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/robot")
OUT.mkdir(parents=True, exist_ok=True)

ROBOTS = ["ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY", "ROBOT_MOPPING", "ROBOT_VACUUMING"]
DAYS = [2, 3, 4]


def load_mids() -> pd.DataFrame:
    parts = []
    for d in DAYS:
        p = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
        p = p[p["product"].isin(ROBOTS)][["day", "timestamp", "product", "mid_price",
                                           "bid_price_1", "ask_price_1"]]
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    df["t_global"] = df["day"] * 1_000_000 + df["timestamp"]
    return df


def pivot_mid(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="t_global", columns="product", values="mid_price").sort_index()


def pivot_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df.pivot_table(index="t_global", columns="product", values="spread").sort_index()


def levels_summary(mid: pd.DataFrame, spr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in ROBOTS:
        m = mid[s].dropna()
        sp = spr[s].dropna()
        ret = m.pct_change().dropna()
        rows.append({
            "symbol": s,
            "mean_mid": m.mean(),
            "std_mid": m.std(),
            "min_mid": m.min(),
            "max_mid": m.max(),
            "mean_spread": sp.mean(),
            "median_spread": sp.median(),
            "ret_std_bps": ret.std() * 1e4,
        })
    return pd.DataFrame(rows)


def constant_sum_test(mid: pd.DataFrame) -> dict:
    s = mid[ROBOTS].dropna().sum(axis=1)
    return {
        "sum_mean": s.mean(),
        "sum_std": s.std(),
        "sum_min": s.min(),
        "sum_max": s.max(),
        "cv": s.std() / s.mean() if s.mean() else float("nan"),
    }


def best_linear_combo(mid: pd.DataFrame) -> pd.DataFrame:
    """Try equal-weight Σ and signed combos. Fit each leg as α + β·others; report R²."""
    M = mid[ROBOTS].dropna()
    rows = []
    # Equal-weight Σ
    s = M.sum(axis=1)
    rows.append({"combo": "+".join(ROBOTS), "std": s.std(), "cv": s.std()/s.mean()})
    # Each leg vs others (regress y = a + Σ β_i x_i)
    for target in ROBOTS:
        X = M.drop(columns=[target]).values
        y = M[target].values
        X_ = np.c_[np.ones(len(X)), X]
        beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
        yhat = X_ @ beta
        resid = y - yhat
        ss_res = (resid ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        rows.append({
            "combo": f"{target} ~ a + " + " + ".join(f"{b:.3f}*{p}" for b, p in zip(beta[1:], M.columns.drop(target))),
            "intercept": beta[0],
            "r2": r2,
            "resid_std": resid.std(),
        })
    return pd.DataFrame(rows)


def lead_lag(mid: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    R = mid[ROBOTS].dropna().pct_change().dropna()
    rows = []
    for a, b in combinations(ROBOTS, 2):
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                c = R[a].corr(R[b])
            elif lag > 0:
                c = R[a].iloc[lag:].reset_index(drop=True).corr(R[b].iloc[:-lag].reset_index(drop=True))
            else:
                c = R[a].iloc[:lag].reset_index(drop=True).corr(R[b].iloc[-lag:].reset_index(drop=True))
            rows.append({"pair": f"{a}|{b}", "lag": lag, "corr": c})
    return pd.DataFrame(rows)


def all_pairs_zscan(mid: pd.DataFrame) -> pd.DataFrame:
    """For every (a,b) and sign in {spread,sum}, compute residual std and
    fraction of |z|>{1.5,2.0,2.5} — proxy for trading opportunities."""
    M = mid[ROBOTS].dropna()
    rows = []
    for a, b in combinations(ROBOTS, 2):
        for sign in ("spread", "sum"):
            sig = M[a] - M[b] if sign == "spread" else M[a] + M[b]
            z = (sig - sig.mean()) / sig.std()
            rows.append({
                "pair": f"{a}|{b}|{sign}",
                "mean": sig.mean(),
                "std": sig.std(),
                "abs_z_gt_1.5": (z.abs() > 1.5).mean(),
                "abs_z_gt_2.0": (z.abs() > 2.0).mean(),
                "abs_z_gt_2.5": (z.abs() > 2.5).mean(),
                # Zero-crossings (how often z crosses 0 — proxy for exit frequency)
                "z_crossings_per_1k": (np.sign(z).diff().abs() > 0).sum() / (len(z) / 1000),
            })
    return pd.DataFrame(rows).sort_values("std")


def correlation_matrices(mid: pd.DataFrame):
    M = mid[ROBOTS].dropna()
    return M.corr(), M.pct_change().dropna().corr()


def main():
    df = load_mids()
    mid = pivot_mid(df)
    spr = pivot_spread(df)

    print("=" * 70)
    print("LEVELS & SPREADS")
    print("=" * 70)
    lv = levels_summary(mid, spr)
    print(lv.to_string(index=False))
    lv.to_csv(OUT / "levels.csv", index=False)

    print("\n" + "=" * 70)
    print("CONSTANT-SUM TEST  (Σ mid_robot)")
    print("=" * 70)
    cs = constant_sum_test(mid)
    print(cs)
    pd.DataFrame([cs]).to_csv(OUT / "constant_sum.csv", index=False)

    print("\n" + "=" * 70)
    print("LINEAR COMBOS  (each leg ~ others)")
    print("=" * 70)
    bc = best_linear_combo(mid)
    print(bc.to_string(index=False))
    bc.to_csv(OUT / "linear_combos.csv", index=False)

    print("\n" + "=" * 70)
    print("LEVEL CORRELATION")
    print("=" * 70)
    cl, cr = correlation_matrices(mid)
    print(cl.round(3))
    cl.to_csv(OUT / "corr_levels.csv")
    cr.to_csv(OUT / "corr_returns.csv")
    print("\nRETURN CORRELATION")
    print(cr.round(3))

    print("\n" + "=" * 70)
    print("LEAD-LAG (top 10 |corr| at lag != 0)")
    print("=" * 70)
    ll = lead_lag(mid)
    ll_nz = ll[ll["lag"] != 0].copy()
    ll_nz["abs_corr"] = ll_nz["corr"].abs()
    print(ll_nz.sort_values("abs_corr", ascending=False).head(10).to_string(index=False))
    ll.to_csv(OUT / "lead_lag.csv", index=False)

    print("\n" + "=" * 70)
    print("ALL PAIR CANDIDATES (sorted by smallest residual std)")
    print("=" * 70)
    pz = all_pairs_zscan(mid)
    print(pz.to_string(index=False))
    pz.to_csv(OUT / "pairs_zscan.csv", index=False)


if __name__ == "__main__":
    main()
