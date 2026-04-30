"""PANEL family EDA: geometric arbitrage, PCA, lead-lag, volume clustering.

Outputs go to eda5/panel/. Reads Data_ROUND_5/prices_round_5_day_{2,3,4}.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent
DAYS = [2, 3, 4]
PANELS: List[str] = ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"]
# nominal surface area (square units)
AREA = {
    "PANEL_1X2": 2,
    "PANEL_2X2": 4,
    "PANEL_1X4": 4,
    "PANEL_2X4": 8,
    "PANEL_4X4": 16,
}


def load_mid_panel() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df = df[df["product"].isin(PANELS)].copy()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    wide = df.pivot_table(
        index=["day", "timestamp"], columns="product", values="mid_price"
    ).sort_index()
    return wide.dropna()


def geometric_relations(mid: pd.DataFrame) -> pd.DataFrame:
    """Test stationary linear relations among panels (price-per-area, ratios)."""
    rows = []
    # price per unit area
    ppa = mid.copy()
    for p in PANELS:
        ppa[p] = mid[p] / AREA[p]
    rows.append(
        {
            "metric": "price_per_unit_area_mean",
            **{p: round(ppa[p].mean(), 2) for p in PANELS},
        }
    )
    rows.append(
        {
            "metric": "price_per_unit_area_std",
            **{p: round(ppa[p].std(), 2) for p in PANELS},
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "p1_price_per_area.csv", index=False)

    # candidate stationary spreads (geometric: same total area)
    cand = {
        # 2*1x2 = 2x2 (same area=4)
        "2*1X2 - 2X2": 2 * mid["PANEL_1X2"] - mid["PANEL_2X2"],
        # 2*1x2 = 1x4 (same area=4)
        "2*1X2 - 1X4": 2 * mid["PANEL_1X2"] - mid["PANEL_1X4"],
        # 1x4 vs 2x2 (same area=4)
        "1X4 - 2X2": mid["PANEL_1X4"] - mid["PANEL_2X2"],
        # 2*2x2 = 2x4 (area=8)
        "2*2X2 - 2X4": 2 * mid["PANEL_2X2"] - mid["PANEL_2X4"],
        # 2*1x4 = 2x4 (area=8)
        "2*1X4 - 2X4": 2 * mid["PANEL_1X4"] - mid["PANEL_2X4"],
        # 2x2 + 1x4 = 2x4 (area=8)
        "2X2+1X4 - 2X4": mid["PANEL_2X2"] + mid["PANEL_1X4"] - mid["PANEL_2X4"],
        # 4*1x2 = 2x4 (area=8)
        "4*1X2 - 2X4": 4 * mid["PANEL_1X2"] - mid["PANEL_2X4"],
        # 2*2x4 = 4x4 (area=16)
        "2*2X4 - 4X4": 2 * mid["PANEL_2X4"] - mid["PANEL_4X4"],
        # 4*2x2 = 4x4 (area=16)
        "4*2X2 - 4X4": 4 * mid["PANEL_2X2"] - mid["PANEL_4X4"],
        # 4*1x4 = 4x4
        "4*1X4 - 4X4": 4 * mid["PANEL_1X4"] - mid["PANEL_4X4"],
        # 8*1x2 = 4x4
        "8*1X2 - 4X4": 8 * mid["PANEL_1X2"] - mid["PANEL_4X4"],
        # 2x4 + 2*2x2 = 4x4 (area=16)
        "2X4+2*2X2 - 4X4": mid["PANEL_2X4"] + 2 * mid["PANEL_2X2"] - mid["PANEL_4X4"],
        # 2x4 + 2*1x4 = 4x4
        "2X4+2*1X4 - 4X4": mid["PANEL_2X4"] + 2 * mid["PANEL_1X4"] - mid["PANEL_4X4"],
    }
    cand_df = pd.DataFrame(cand)
    summary = pd.DataFrame(
        {
            "spread": list(cand.keys()),
            "mean": [cand_df[k].mean() for k in cand],
            "std": [cand_df[k].std() for k in cand],
            "abs_mean_over_std": [
                abs(cand_df[k].mean()) / (cand_df[k].std() + 1e-9) for k in cand
            ],
            "min": [cand_df[k].min() for k in cand],
            "max": [cand_df[k].max() for k in cand],
        }
    )
    summary["zscore_range"] = (summary["max"] - summary["min"]) / (summary["std"] + 1e-9)
    summary = summary.sort_values("std")
    summary.to_csv(OUT_DIR / "p2_geometric_spreads.csv", index=False)
    return summary


def pca_panel(mid: pd.DataFrame) -> pd.DataFrame:
    """PCA on panel mid prices (centred, scaled by std)."""
    X = mid[PANELS].values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    # sort descending
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    explained = vals / vals.sum()
    out = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(len(vals))],
            "eigenvalue": vals,
            "explained_var": explained,
            "cum_explained": np.cumsum(explained),
        }
    )
    loadings = pd.DataFrame(vecs, index=PANELS, columns=out["PC"])
    out.to_csv(OUT_DIR / "p3_pca_explained.csv", index=False)
    loadings.to_csv(OUT_DIR / "p3_pca_loadings.csv")

    # Project residual = price - reconstruction from PC1 only
    pc1_score = X @ vecs[:, 0]
    recon = np.outer(pc1_score, vecs[:, 0])
    resid = X - recon
    resid_df = pd.DataFrame(resid, index=mid.index, columns=PANELS)
    resid_summary = pd.DataFrame(
        {
            "panel": PANELS,
            "resid_mean": [resid_df[p].mean() for p in PANELS],
            "resid_std": [resid_df[p].std() for p in PANELS],
            "resid_abs_max": [resid_df[p].abs().max() for p in PANELS],
        }
    ).sort_values("resid_std", ascending=False)
    resid_summary.to_csv(OUT_DIR / "p3_pca_residuals.csv", index=False)
    return out


def lead_lag(mid: pd.DataFrame) -> pd.DataFrame:
    """Per-day lead-lag: cross-correlation of returns with lags ±5."""
    rets = mid.groupby(level="day").apply(lambda g: g[PANELS].diff()).reset_index(level=0, drop=True)
    rets = rets.dropna()
    lags = list(range(-5, 6))
    rows = []
    for a in PANELS:
        for b in PANELS:
            if a == b:
                continue
            best_lag = 0
            best_corr = 0.0
            for lag in lags:
                if lag >= 0:
                    c = rets[a].corr(rets[b].shift(lag))
                else:
                    c = rets[a].corr(rets[b].shift(lag))
                if c is not None and abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag
            rows.append(
                {"a": a, "b": b, "best_lag": best_lag, "best_corr": round(best_corr, 4)}
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "p4_leadlag.csv", index=False)
    # Aggregate: leader score = number of times each panel appears with positive lag
    # If a leads b, then rets[a] correlates with rets[b].shift(+lag) for lag>0,
    # i.e. b's CURRENT return is explained by a's PAST returns. We want lag>0 here.
    pos = df[df["best_lag"] > 0].groupby("a").size().reindex(PANELS, fill_value=0)
    neg = df[df["best_lag"] < 0].groupby("a").size().reindex(PANELS, fill_value=0)
    leader = pd.DataFrame({"panel": PANELS, "leads_count": pos.values, "lags_count": neg.values})
    leader["score"] = leader["leads_count"] - leader["lags_count"]
    leader = leader.sort_values("score", ascending=False)
    leader.to_csv(OUT_DIR / "p4_leader_score.csv", index=False)
    return df


def volume_clusters(mid: pd.DataFrame) -> pd.DataFrame:
    """Volume>20 events and subsequent spread widening across panels."""
    rows = []
    for d in DAYS:
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df = df[df["product"].isin(PANELS)]
        wide_bid = df.pivot_table(index="timestamp", columns="product", values="bid_price_1")
        wide_ask = df.pivot_table(index="timestamp", columns="product", values="ask_price_1")
        wide_bv = df.pivot_table(index="timestamp", columns="product", values="bid_volume_1")
        wide_av = df.pivot_table(index="timestamp", columns="product", values="ask_volume_1")
        spread = wide_ask - wide_bid
        for p in PANELS:
            v = wide_bv[p].fillna(0) + wide_av[p].fillna(0)
            big = v > 20
            if big.sum() == 0:
                continue
            # mean spread BEFORE and AFTER big events (1-step ahead)
            sp = spread[p]
            spread_before = sp.shift(1)[big].mean()
            spread_after = sp.shift(-1)[big].mean()
            rows.append(
                {
                    "day": d,
                    "panel": p,
                    "n_big_volume_events": int(big.sum()),
                    "mean_spread_before": round(spread_before, 3),
                    "mean_spread_after": round(spread_after, 3),
                    "delta": round(spread_after - spread_before, 3),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "p5_volume_clusters.csv", index=False)
    return df


def pair_zscore_screen(mid: pd.DataFrame) -> pd.DataFrame:
    """Screen all (a, b, sign) pairs in PANEL family for stationary z-trading
    (mirror approach used in other family EDAs)."""
    rows = []
    panels = PANELS
    for i, a in enumerate(panels):
        for b in panels[i + 1 :]:
            for sign in ("spread", "sum"):
                sig = mid[a] - mid[b] if sign == "spread" else mid[a] + mid[b]
                m, s = sig.mean(), sig.std()
                if s < 1e-9:
                    continue
                z = (sig - m) / s
                # simulated naive PnL: enter at |z|>1.5, exit at |z|<0.3, size=10
                pos = 0
                pnl = 0.0
                last_sig = None
                for v in sig.values:
                    if last_sig is None:
                        last_sig = v
                        continue
                    cur_z = (v - m) / s
                    if pos == 0:
                        if cur_z > 1.5:
                            pos = -1
                            entry = v
                        elif cur_z < -1.5:
                            pos = +1
                            entry = v
                    else:
                        if abs(cur_z) < 0.3:
                            pnl += pos * (entry - v) * 10
                            pos = 0
                rows.append(
                    {
                        "a": a,
                        "b": b,
                        "sign": sign,
                        "mean": round(m, 2),
                        "std": round(s, 2),
                        "abs_mean_over_std": round(abs(m) / (s + 1e-9), 2),
                        "naive_pnl_10": round(pnl, 0),
                    }
                )
    df = pd.DataFrame(rows).sort_values("naive_pnl_10", ascending=False)
    df.to_csv(OUT_DIR / "p6_pair_screen.csv", index=False)
    return df


def main() -> None:
    print("Loading panel mid prices...")
    mid = load_mid_panel()
    print(f"  {len(mid)} ticks across days {DAYS}")

    print("\n[1] Geometric relations...")
    geo = geometric_relations(mid)
    print(geo.head(10).to_string(index=False))

    print("\n[2] PCA...")
    pca = pca_panel(mid)
    print(pca.to_string(index=False))

    print("\n[3] Lead-lag (top 15 by |corr|)...")
    ll = lead_lag(mid)
    print(ll.reindex(ll["best_corr"].abs().sort_values(ascending=False).index).head(15).to_string(index=False))

    print("\n[4] Volume clustering...")
    vol = volume_clusters(mid)
    print(vol.to_string(index=False))

    print("\n[5] Pair screen (top 10 by naive PnL)...")
    ps = pair_zscore_screen(mid)
    print(ps.head(10).to_string(index=False))

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
