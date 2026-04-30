"""TRANSLATOR family — full EDA.

5 symbols: ASTRO_BLACK, ECLIPSE_CHARCOAL, GRAPHITE_MIST, SPACE_GRAY, VOID_BLUE.
Current strategy in v3.1:
  - Pair: SPACE_GRAY|GRAPHITE_MIST spread z=1.2
  - Pair: SPACE_GRAY|VOID_BLUE     sum    z=1.2
  - MM:   ECLIPSE_CHARCOAL, ASTRO_BLACK
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/translator")
OUT.mkdir(parents=True, exist_ok=True)

TRANSLATORS = ["TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
               "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE"]
DAYS = [2, 3, 4]


def load() -> pd.DataFrame:
    parts = []
    for d in DAYS:
        p = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
        p = p[p["product"].isin(TRANSLATORS)].copy()
        cols = ["day", "timestamp", "product", "mid_price",
                "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
                "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3"]
        parts.append(p[cols])
    df = pd.concat(parts, ignore_index=True)
    df["t_global"] = df["day"] * 1_000_000 + df["timestamp"]
    return df


def trades() -> pd.DataFrame:
    parts = []
    for d in DAYS:
        p = pd.read_csv(DATA / f"trades_round_5_day_{d}.csv", sep=";")
        p = p[p["symbol"].isin(TRANSLATORS)].copy()
        p["day"] = d
        p["t_global"] = p["day"] * 1_000_000 + p["timestamp"]
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def pivot_mid(df):
    return df.pivot_table(index="t_global", columns="product", values="mid_price").sort_index()


def levels_summary(df):
    rows = []
    for s in TRANSLATORS:
        sub = df[df["product"] == s]
        m = sub["mid_price"]
        spread = sub["ask_price_1"] - sub["bid_price_1"]
        ret = m.pct_change().dropna()
        # Total volume: bid + ask top-of-book
        vol_bid = sub["bid_volume_1"].fillna(0)
        vol_ask = sub["ask_volume_1"].fillna(0)
        rows.append({
            "symbol": s,
            "mean_mid": m.mean(),
            "std_mid": m.std(),
            "min": m.min(),
            "max": m.max(),
            "spread_mean": spread.mean(),
            "spread_med": spread.median(),
            "ret_std_bps": ret.std() * 1e4,
            "bid_vol_top_mean": vol_bid.mean(),
            "ask_vol_top_mean": vol_ask.mean(),
        })
    return pd.DataFrame(rows)


def constant_sum_test(mid):
    s = mid[TRANSLATORS].dropna().sum(axis=1)
    return {"mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(),
            "cv": s.std() / s.mean() if s.mean() else float("nan")}


def linear_combos(mid):
    M = mid[TRANSLATORS].dropna()
    rows = []
    for target in TRANSLATORS:
        X = M.drop(columns=[target]).values
        y = M[target].values
        X_ = np.c_[np.ones(len(X)), X]
        beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
        yhat = X_ @ beta
        resid = y - yhat
        ss_res = (resid**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = 1 - ss_res/ss_tot
        rows.append({
            "target": target,
            "intercept": beta[0],
            "betas": dict(zip([c for c in M.columns if c != target], beta[1:].round(4))),
            "r2": r2,
            "resid_std": resid.std(),
        })
    return pd.DataFrame(rows)


def all_pairs_zscan(mid):
    M = mid[TRANSLATORS].dropna()
    rows = []
    for a, b in combinations(TRANSLATORS, 2):
        for sign in ("spread", "sum"):
            sig = M[a] - M[b] if sign == "spread" else M[a] + M[b]
            z = (sig - sig.mean()) / sig.std()
            rows.append({
                "pair": f"{a.replace('TRANSLATOR_','')}|{b.replace('TRANSLATOR_','')}|{sign}",
                "mean": sig.mean(),
                "std": sig.std(),
                "abs_z_gt_1.5": (z.abs() > 1.5).mean(),
                "abs_z_gt_2.0": (z.abs() > 2.0).mean(),
                "abs_z_gt_2.5": (z.abs() > 2.5).mean(),
                "z_crossings_per_1k": (np.sign(z).diff().abs() > 0).sum() / (len(z) / 1000),
            })
    return pd.DataFrame(rows).sort_values("std")


def correlation_matrices(mid):
    M = mid[TRANSLATORS].dropna()
    return M.corr(), M.pct_change().dropna().corr()


def lead_lag(mid, max_lag=10):
    R = mid[TRANSLATORS].dropna().pct_change().dropna()
    rows = []
    for a, b in combinations(TRANSLATORS, 2):
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                c = R[a].corr(R[b])
            elif lag > 0:
                c = R[a].iloc[lag:].reset_index(drop=True).corr(R[b].iloc[:-lag].reset_index(drop=True))
            else:
                c = R[a].iloc[:lag].reset_index(drop=True).corr(R[b].iloc[-lag:].reset_index(drop=True))
            rows.append({"pair": f"{a.replace('TRANSLATOR_','')}|{b.replace('TRANSLATOR_','')}", "lag": lag, "corr": c})
    return pd.DataFrame(rows)


def trade_volume_summary(t):
    rows = []
    for s in TRANSLATORS:
        sub = t[t["symbol"] == s]
        rows.append({
            "symbol": s,
            "trades_count": len(sub),
            "vol_total": sub["quantity"].sum(),
            "vol_mean": sub["quantity"].mean(),
            "vol_p95": sub["quantity"].quantile(0.95),
            "px_std": sub["price"].std(),
        })
    return pd.DataFrame(rows)


def plot_norm(mid):
    M = mid[TRANSLATORS].dropna()
    fig, ax = plt.subplots(figsize=(14, 6))
    for c in M.columns:
        ax.plot(M.index, (M[c] - M[c].iloc[0]) / M[c].iloc[0] * 100, label=c.replace("TRANSLATOR_",""), linewidth=0.7)
    ax.legend(); ax.set_title("TRANSLATOR — normalized mid (% from start)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "p1_normalized.png", dpi=120); plt.close(fig)


def plot_corr_heatmap(corr_levels, corr_returns):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, corr, title in zip(axes, [corr_levels, corr_returns], ["Levels corr", "Returns corr"]):
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
        ax.set_xticklabels([c.replace("TRANSLATOR_","") for c in corr.columns], rotation=45, ha="right")
        ax.set_yticklabels([c.replace("TRANSLATOR_","") for c in corr.index])
        ax.set_title(title)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(OUT / "p2_corr.png", dpi=120); plt.close(fig)


def plot_top_pairs(mid, top_pairs):
    fig, axes = plt.subplots(len(top_pairs), 1, figsize=(14, 3*len(top_pairs)))
    if len(top_pairs) == 1: axes = [axes]
    for ax, (label, sig) in zip(axes, top_pairs):
        z = (sig - sig.mean()) / sig.std()
        ax.plot(sig.index, z, linewidth=0.6)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axhline(2, color='r', linewidth=0.5, linestyle='--')
        ax.axhline(-2, color='r', linewidth=0.5, linestyle='--')
        ax.set_title(f"z-score: {label} (std={sig.std():.1f})")
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "p3_top_pairs.png", dpi=120); plt.close(fig)


def main():
    df = load()
    print(f"Loaded {len(df)} rows ({df['day'].nunique()} days)")
    mid = pivot_mid(df)

    print("\n" + "="*70)
    print("LEVELS, SPREAD, VOLUME (top of book)")
    print("="*70)
    lv = levels_summary(df)
    print(lv.to_string(index=False))
    lv.to_csv(OUT / "levels.csv", index=False)

    print("\n" + "="*70)
    print("CONSTANT-SUM TEST")
    print("="*70)
    cs = constant_sum_test(mid)
    print(cs)

    print("\n" + "="*70)
    print("LINEAR COMBOS (each leg ~ others)")
    print("="*70)
    lc = linear_combos(mid)
    print(lc.to_string(index=False))
    lc.to_csv(OUT / "linear_combos.csv", index=False)

    print("\n" + "="*70)
    print("LEVEL CORRELATION")
    print("="*70)
    cl, cr = correlation_matrices(mid)
    print(cl.round(3))
    print("\nRETURN CORRELATION")
    print(cr.round(3))
    cl.to_csv(OUT / "corr_levels.csv"); cr.to_csv(OUT / "corr_returns.csv")

    print("\n" + "="*70)
    print("ALL PAIR CANDIDATES (sorted by smallest residual std)")
    print("="*70)
    pz = all_pairs_zscan(mid)
    print(pz.to_string(index=False))
    pz.to_csv(OUT / "pairs_zscan.csv", index=False)

    print("\n" + "="*70)
    print("LEAD-LAG (top 10 by |corr| at lag != 0)")
    print("="*70)
    ll = lead_lag(mid)
    ll_nz = ll[ll["lag"] != 0].copy()
    ll_nz["abs_corr"] = ll_nz["corr"].abs()
    print(ll_nz.sort_values("abs_corr", ascending=False).head(10).to_string(index=False))
    ll.to_csv(OUT / "lead_lag.csv", index=False)

    print("\n" + "="*70)
    print("TRADE VOLUME (executed market trades)")
    print("="*70)
    t = trades()
    tv = trade_volume_summary(t)
    print(tv.to_string(index=False))
    tv.to_csv(OUT / "trade_volume.csv", index=False)

    # Plots
    plot_norm(mid)
    plot_corr_heatmap(cl, cr)
    # Top 3 pairs by smallest std
    M = mid[TRANSLATORS].dropna()
    top = []
    for _, r in pz.head(3).iterrows():
        a, b, sign = r["pair"].split("|")
        a = "TRANSLATOR_" + a; b = "TRANSLATOR_" + b
        sig = M[a] - M[b] if sign == "spread" else M[a] + M[b]
        top.append((r["pair"], sig))
    plot_top_pairs(M, top)

    print("\nPlots saved to:", OUT)


if __name__ == "__main__":
    main()
