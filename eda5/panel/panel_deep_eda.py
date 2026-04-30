"""Deep PANEL EDA — looking for the 200k edge.

Sections:
  [A] Per-day microstructure: spread, depth, tick volatility, mid jump distrib.
  [B] Mid autocorrelation & mean-reversion timescale (variance ratio test).
  [C] Realistic MM edge: spread - 2*tick * fillrate proxy.
  [D] Best executed pairs (using bid/ask, not mid) — enter at ask, exit at bid.
  [E] PANEL_1X2 deep-dive: why does it bleed in day 3?
  [F] Per-day basket regime test: are coefs stable across days?
  [G] Joint-residual: trade ALL panels at once on a single shared z signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent
PANELS = ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"]
DAYS = [2, 3, 4]


def load_full(day: int) -> pd.DataFrame:
    """Full per-tick book for PANEL on a single day."""
    df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{day}.csv", sep=";")
    return df[df["product"].isin(PANELS)].copy()


def section_A_microstructure() -> pd.DataFrame:
    """Per-(day, panel) spread, top-of-book depth, tick volatility."""
    rows = []
    for day in DAYS:
        df = load_full(day)
        for p in PANELS:
            sub = df[df["product"] == p].sort_values("timestamp")
            spread = sub["ask_price_1"] - sub["bid_price_1"]
            mid = (sub["ask_price_1"] + sub["bid_price_1"]) / 2.0
            ret = mid.diff()
            rows.append(
                {
                    "day": day,
                    "panel": p,
                    "n_ticks": len(sub),
                    "mid_mean": round(mid.mean(), 1),
                    "mid_std": round(mid.std(), 1),
                    "spread_mean": round(spread.mean(), 2),
                    "spread_p50": round(spread.median(), 2),
                    "spread_p95": round(spread.quantile(0.95), 2),
                    "bid_vol_mean": round(sub["bid_volume_1"].mean(), 1),
                    "ask_vol_mean": round(sub["ask_volume_1"].mean(), 1),
                    "tick_vol_std": round(ret.std(), 2),
                    "tick_vol_p95": round(ret.abs().quantile(0.95), 2),
                    "abs_ret_mean": round(ret.abs().mean(), 2),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "A_microstructure.csv", index=False)
    return df


def section_B_mean_reversion(mid: pd.DataFrame) -> pd.DataFrame:
    """Variance ratio: VR(k) = Var(r_k) / (k * Var(r_1)).
    VR<1 → mean-reverting, VR>1 → trending, VR≈1 → random walk."""
    rows = []
    rets = mid.diff().dropna()
    for p in PANELS:
        var1 = rets[p].var()
        for k in (2, 5, 10, 20, 50, 100):
            r_k = mid[p].diff(k).dropna()
            vr = r_k.var() / (k * var1) if var1 > 0 else float("nan")
            rows.append({"panel": p, "k": k, "VR": round(vr, 3)})
    df = pd.DataFrame(rows).pivot(index="panel", columns="k", values="VR")
    df.to_csv(OUT_DIR / "B_variance_ratio.csv")
    # autocorrelation of returns at lag 1..5
    ac_rows = []
    for p in PANELS:
        for lag in range(1, 6):
            ac_rows.append({"panel": p, "lag": lag, "rho": round(rets[p].autocorr(lag), 4)})
    ac = pd.DataFrame(ac_rows).pivot(index="panel", columns="lag", values="rho")
    ac.to_csv(OUT_DIR / "B_return_autocorr.csv")
    return df


def section_C_mm_edge() -> pd.DataFrame:
    """For each panel and each day: estimate the MM edge if you quote +1/-1.
    Edge = spread - 1 (you give up 1 tick to be inside).
    Combined with fill probability proxy (book volume at top).
    Profit_per_round_trip ≈ (spread - 2) if you fill both sides at +1/-1.
    """
    rows = []
    for day in DAYS:
        df = load_full(day)
        for p in PANELS:
            sub = df[df["product"] == p].sort_values("timestamp").reset_index(drop=True)
            spread = sub["ask_price_1"] - sub["bid_price_1"]
            # quotable_pct: ticks where spread >= 3 (room for +1/-1 inside)
            quotable = (spread >= 3).mean()
            quotable_5 = (spread >= 5).mean()
            quotable_10 = (spread >= 10).mean()
            # raw mm round-trip if both sides fill at +1/-1
            mm_edge = (spread - 2).clip(lower=0)
            rows.append(
                {
                    "day": day,
                    "panel": p,
                    "spread_mean": round(spread.mean(), 2),
                    "pct_spread_ge_3": round(quotable * 100, 1),
                    "pct_spread_ge_5": round(quotable_5 * 100, 1),
                    "pct_spread_ge_10": round(quotable_10 * 100, 1),
                    "mm_edge_per_rt_mean": round(mm_edge.mean(), 2),
                    # crude proxy: edge * size 5 * 0.05 fillrate per tick * 10000 ticks
                    "mm_pnl_proxy": round(mm_edge.mean() * 5 * 0.05 * 10000, 0),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "C_mm_edge.csv", index=False)
    return df


def section_D_realistic_pair_pnl(mid: pd.DataFrame, day_books: dict) -> pd.DataFrame:
    """Simulate (a, b, sign) pair trades using REAL bid/ask for execution.
    Enter aggressive: pay ask on long leg, hit bid on short leg.
    Exit aggressive at threshold."""
    rows = []
    panels = PANELS
    for entry_z in (1.5, 2.0, 2.5, 3.0):
        for i, a in enumerate(panels):
            for b in panels[i + 1:]:
                for sign in ("spread", "sum"):
                    sig = mid[a] - mid[b] if sign == "spread" else mid[a] + mid[b]
                    m, s = sig.mean(), sig.std()
                    if s < 1e-9:
                        continue
                    pos = 0
                    pnl = 0.0
                    sig_vals = sig.values
                    keys = list(mid.index)
                    for idx, v in enumerate(sig_vals):
                        cur_z = (v - m) / s
                        if pos == 0:
                            if cur_z > entry_z:
                                pos = -1
                            elif cur_z < -entry_z:
                                pos = +1
                            if pos != 0:
                                day, ts = keys[idx]
                                book = day_books[day]
                                # entry: long a → pay ask_a; short b → hit bid_b (sign='sum'
                                # means same direction on both legs)
                                ask_a = book[a]["ask"].iloc[idx % len(book[a])]
                                bid_a = book[a]["bid"].iloc[idx % len(book[a])]
                                ask_b = book[b]["ask"].iloc[idx % len(book[b])]
                                bid_b = book[b]["bid"].iloc[idx % len(book[b])]
                                if pos == 1:  # long a, short_or_long b
                                    entry_a_px = ask_a
                                    entry_b_px = bid_b if sign == "spread" else ask_b
                                else:
                                    entry_a_px = bid_a
                                    entry_b_px = ask_b if sign == "spread" else bid_b
                        else:
                            if abs(cur_z) < 0.3:
                                day, ts = keys[idx]
                                book = day_books[day]
                                ask_a = book[a]["ask"].iloc[idx % len(book[a])]
                                bid_a = book[a]["bid"].iloc[idx % len(book[a])]
                                ask_b = book[b]["ask"].iloc[idx % len(book[b])]
                                bid_b = book[b]["bid"].iloc[idx % len(book[b])]
                                if pos == 1:
                                    exit_a_px = bid_a
                                    exit_b_px = ask_b if sign == "spread" else bid_b
                                else:
                                    exit_a_px = ask_a
                                    exit_b_px = bid_b if sign == "spread" else ask_b
                                pnl_a = pos * (exit_a_px - entry_a_px) * 10
                                if sign == "spread":
                                    pnl_b = -pos * (exit_b_px - entry_b_px) * 10
                                else:
                                    pnl_b = pos * (exit_b_px - entry_b_px) * 10
                                # Note: spread sign means b is short when a is long, so b's
                                # contribution flips
                                if sign == "spread":
                                    pnl += pnl_a + pnl_b
                                else:
                                    pnl += pnl_a + pnl_b  # both same direction = 'sum'
                                pos = 0
                    rows.append(
                        {
                            "entry_z": entry_z,
                            "a": a,
                            "b": b,
                            "sign": sign,
                            "pnl_taker": round(pnl, 0),
                        }
                    )
    df = pd.DataFrame(rows).sort_values("pnl_taker", ascending=False)
    df.to_csv(OUT_DIR / "D_realistic_pair_pnl.csv", index=False)
    return df


def section_E_panel_1x2_deepdive() -> pd.DataFrame:
    """Why does PANEL_1X2 bleed in day 3 (-9k)? Look at MM scenarios:
    - filled passive at +1/-1
    - hit by adverse selection (mid moves against quote)
    """
    rows = []
    for day in DAYS:
        df = load_full(day)
        sub = df[df["product"] == "PANEL_1X2"].sort_values("timestamp").reset_index(drop=True)
        bid = sub["bid_price_1"]
        ask = sub["ask_price_1"]
        mid = (bid + ask) / 2.0
        spread = ask - bid
        ret_1 = mid.diff().shift(-1)  # next-tick mid move
        # If we quote at bid+1 and get filled (bid_size matches inflow), we're long.
        # Adverse selection if mid drops next tick.
        adv_long = (ret_1 < 0).mean()
        adv_short = (ret_1 > 0).mean()
        # Net mid drift over the day
        drift = mid.iloc[-1] - mid.iloc[0]
        rows.append(
            {
                "day": day,
                "n_ticks": len(sub),
                "mid_start": round(mid.iloc[0], 1),
                "mid_end": round(mid.iloc[-1], 1),
                "drift_total": round(drift, 1),
                "spread_mean": round(spread.mean(), 2),
                "spread_p10": round(spread.quantile(0.1), 2),
                "pct_adverse_long": round(adv_long * 100, 1),
                "pct_adverse_short": round(adv_short * 100, 1),
                "abs_ret_mean": round(mid.diff().abs().mean(), 3),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "E_panel_1x2_deepdive.csv", index=False)
    return df


def section_F_basket_regime() -> pd.DataFrame:
    """Are basket OLS coefficients stable across days?"""
    rows = []
    for day in DAYS:
        df = load_full(day)
        wide = df.pivot_table(index="timestamp", columns="product", values="mid_price").dropna()
        for p in PANELS:
            others = [o for o in PANELS if o != p]
            X = np.column_stack([np.ones(len(wide)), wide[others].values])
            y = wide[p].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            row = {"day": day, "panel": p, "intercept": round(beta[0], 1),
                   "resid_std": round(r.std(), 1),
                   "R2": round(1 - r.var() / y.var(), 3)}
            for s, b in zip(others, beta[1:]):
                row[f"b_{s}"] = round(b, 4)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "F_basket_per_day.csv", index=False)
    return df


def section_G_pair_2x4_4x4_pnl_realistic() -> pd.DataFrame:
    """Replicate the live v1 pair (PANEL_2X4|PANEL_4X4 sum 2.0) with REAL bid/ask
    execution to see what's the achievable ceiling and where v1 leaves money."""
    rows = []
    for day in DAYS:
        df = load_full(day)
        wide = df.pivot_table(index="timestamp", columns="product",
                              values=["mid_price", "bid_price_1", "ask_price_1"]).dropna()
        mid_a = wide["mid_price"]["PANEL_2X4"].values
        mid_b = wide["mid_price"]["PANEL_4X4"].values
        bid_a = wide["bid_price_1"]["PANEL_2X4"].values
        ask_a = wide["ask_price_1"]["PANEL_2X4"].values
        bid_b = wide["bid_price_1"]["PANEL_4X4"].values
        ask_b = wide["ask_price_1"]["PANEL_4X4"].values
        sig = mid_a + mid_b
        m, s = sig.mean(), sig.std()
        for entry_z in (1.5, 2.0, 2.5, 3.0):
            for exit_z in (0.0, 0.3, 0.5):
                pos = 0
                pnl = 0.0
                ea = eb = 0.0
                for i, v in enumerate(sig):
                    z = (v - m) / s
                    if pos == 0:
                        if z > entry_z:
                            pos = -1
                            ea, eb = bid_a[i], bid_b[i]  # short both at bid
                        elif z < -entry_z:
                            pos = 1
                            ea, eb = ask_a[i], ask_b[i]  # long both at ask
                    else:
                        if abs(z) <= exit_z:
                            if pos == 1:
                                xa, xb = bid_a[i], bid_b[i]
                            else:
                                xa, xb = ask_a[i], ask_b[i]
                            pnl += pos * (xa - ea) * 10 + pos * (xb - eb) * 10
                            pos = 0
                rows.append({
                    "day": day, "entry_z": entry_z, "exit_z": exit_z,
                    "pnl_taker_realistic": round(pnl, 0),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "G_2x4_4x4_pair_grid.csv", index=False)
    return df


def main():
    print("[A] Microstructure...")
    A = section_A_microstructure()
    print(A.to_string(index=False))

    # load mid wide (3 days)
    frames = []
    for d in DAYS:
        df = load_full(d)
        w = df.pivot_table(index="timestamp", columns="product", values="mid_price").dropna()
        w["day"] = d
        w = w.set_index("day", append=True).reorder_levels(["day", "timestamp"])
        frames.append(w)
    mid = pd.concat(frames).sort_index()

    print("\n[B] Variance ratio (mean-reversion timescale)...")
    B = section_B_mean_reversion(mid)
    print(B.to_string())

    print("\n[C] MM edge realistic...")
    C = section_C_mm_edge()
    print(C.to_string(index=False))

    print("\n[E] PANEL_1X2 deepdive...")
    E = section_E_panel_1x2_deepdive()
    print(E.to_string(index=False))

    print("\n[F] Basket coefs per day (stability check)...")
    F = section_F_basket_regime()
    print(F.to_string(index=False))

    print("\n[G] PANEL_2X4|PANEL_4X4 sum pair grid (taker realistic)...")
    G = section_G_pair_2x4_4x4_pnl_realistic()
    pivot = G.pivot_table(index=["entry_z", "exit_z"], columns="day",
                         values="pnl_taker_realistic", aggfunc="sum")
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.sort_values("total", ascending=False).to_string())


if __name__ == "__main__":
    main()
