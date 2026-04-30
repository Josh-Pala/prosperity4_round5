"""
EVENING_BREATH deep dive — replicate MINT cross-family fair-value workflow.

Outputs:
  - eb_lagged_corr.csv          lead/lag corr with all other symbols
  - eb_basket_regression.txt    OLS of EB mid on best predictors (top8)
  - eb_autocorr.csv             ACF on returns
  - eb_walkforward.csv          walk-forward holdout sd for candidate baskets
  - eb_panel_only.txt           OLS coefs for PANEL-only basket (if relevant)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent
TARGET = "OXYGEN_SHAKE_EVENING_BREATH"


def load_all_mids() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df["t"] = d * 1_000_000 + df["timestamp"]
        df["day"] = d
        frames.append(df[["t", "day", "product", "mid_price"]])
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["t", "day"], columns="product", values="mid_price").sort_index()
    return wide.reset_index()


def lead_lag_correlations(mids: pd.DataFrame, target: str,
                          lags=(-50, -20, -10, -5, -1, 1, 5, 10, 20, 50)) -> pd.DataFrame:
    rets = mids.drop(columns=["t", "day"]).diff()
    target_ret = rets[target]
    rows = []
    for s in rets.columns:
        if s == target:
            continue
        row = {"symbol": s}
        for L in lags:
            shifted = rets[s].shift(-L)
            row[f"lag_{L:+d}"] = round(target_ret.corr(shifted), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xm = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    pred = Xm @ coef
    resid = y - pred
    return coef, resid


def basket_regression(mids: pd.DataFrame, target: str, predictors: list[str],
                      label: str) -> str:
    msg = [f"=== {label}: predictors={predictors} ==="]
    Xy = mids[predictors + [target]].dropna()
    X = Xy[predictors].values
    y = Xy[target].values
    coef, resid = fit_ols(X, y)
    msg.append("OLS coefficients:")
    msg.append(f"  intercept = {coef[0]:.4f}")
    for s, b in zip(predictors, coef[1:]):
        msg.append(f"  {s:35s} beta={b:+.4f}")
    msg.append(f"Residual mean={resid.mean():.2f}  sd={resid.std():.2f}  R^2={1 - resid.var()/y.var():.4f}")
    # Half-life
    dx = np.diff(resid)
    A = np.column_stack([np.ones(len(dx)), resid[:-1] - resid[:-1].mean()])
    cc, *_ = np.linalg.lstsq(A, dx, rcond=None)
    phi = cc[1]
    if -1 < phi < 0:
        hl = -np.log(2) / np.log(1 + phi)
        msg.append(f"Residual half-life: {hl:.0f} ticks")
    else:
        msg.append(f"Residual half-life: ∞ (phi={phi:.4f})")
    return "\n".join(msg), coef


def walk_forward(mids: pd.DataFrame, target: str, predictors: list[str], label: str) -> dict:
    """Train on 2 days, test sd on the 3rd."""
    out = {"basket": label, "n_predictors": len(predictors)}
    full = mids[["day"] + predictors + [target]].dropna()
    # In-sample fit on the full sample
    X_full = full[predictors].values
    y_full = full[target].values
    coef, resid_in = fit_ols(X_full, y_full)
    out["R2_in"] = round(1 - resid_in.var() / y_full.var(), 4)
    out["sd_in"] = round(resid_in.std(), 1)
    # Walk-forward: hold out each day
    days = sorted(full["day"].unique())
    for hold in days:
        train = full[full["day"] != hold]
        test = full[full["day"] == hold]
        Xt = train[predictors].values
        yt = train[target].values
        coef_t, _ = fit_ols(Xt, yt)
        Xh = test[predictors].values
        yh = test[target].values
        pred = coef_t[0] + Xh @ coef_t[1:]
        oos_resid = yh - pred
        out[f"sd_oos_d{hold}"] = round(oos_resid.std(), 1)
    return out


def autocorr_returns(mids: pd.DataFrame, target: str, lags=range(1, 21)) -> pd.DataFrame:
    r = mids[target].diff().dropna()
    rows = [{"lag": L, "acf": round(r.autocorr(lag=L), 4)} for L in lags]
    return pd.DataFrame(rows)


def main():
    wide = load_all_mids()
    only_mid = wide.drop(columns=["t", "day"])
    print(f"Loaded {len(only_mid)} ticks × {len(only_mid.columns)} symbols")

    # Top correlations (level)
    levels = only_mid.corrwith(only_mid[TARGET]).drop(TARGET).abs().sort_values(ascending=False)
    print("\n=== TOP 15 level correlations w/ EB ===")
    print(levels.head(15).to_string())
    levels.to_csv(OUT_DIR / "eb_level_corr.csv")

    # Lead-lag
    ll = lead_lag_correlations(wide, TARGET)
    ll["max_abs"] = ll.iloc[:, 1:].abs().max(axis=1)
    ll = ll.sort_values("max_abs", ascending=False)
    ll.to_csv(OUT_DIR / "eb_lagged_corr.csv", index=False)
    print("\n=== TOP 15 lead-lag correlations ===")
    print(ll.head(15).to_string(index=False))

    # Candidate baskets
    top8 = levels.head(8).index.tolist()
    top5 = levels.head(5).index.tolist()
    top4 = levels.head(4).index.tolist()
    top3 = levels.head(3).index.tolist()
    panels_in_top = [s for s in levels.index if s.startswith("PANEL_")][:3]
    oxys_in_top = [s for s in levels.index if s.startswith("OXYGEN_SHAKE_") and s != TARGET][:3]

    candidates = {
        "top8": top8,
        "top5": top5,
        "top4": top4,
        "top3": top3,
    }
    if len(panels_in_top) >= 3:
        candidates["panel_only_top3"] = panels_in_top
    if len(oxys_in_top) >= 3:
        candidates["oxy_only_top3"] = oxys_in_top
    # Also try a single-family basket reusing MINT's PANEL trio explicitly
    candidates["panel_2x2_1x4_4x4"] = ["PANEL_2X2", "PANEL_1X4", "PANEL_4X4"]

    # Top8 regression text
    msg_top8, coef_top8 = basket_regression(wide, TARGET, top8, "TOP8 BASKET")
    (OUT_DIR / "eb_basket_regression.txt").write_text(msg_top8)
    print("\n" + msg_top8)

    # Walk-forward across candidates
    rows = []
    for label, preds in candidates.items():
        rows.append(walk_forward(wide, TARGET, preds, label))
    wf = pd.DataFrame(rows)
    wf.to_csv(OUT_DIR / "eb_walkforward.csv", index=False)
    print("\n=== WALK-FORWARD HOLDOUT (sd_oos by day) ===")
    print(wf.to_string(index=False))

    # Save per-basket OLS coefficients (full-sample) for trader use
    coef_dump = []
    for label, preds in candidates.items():
        Xy = wide[preds + [TARGET]].dropna()
        coef, resid = fit_ols(Xy[preds].values, Xy[TARGET].values)
        line = {"basket": label, "intercept": round(coef[0], 4)}
        for s, b in zip(preds, coef[1:]):
            line[s] = round(b, 6)
        coef_dump.append(line)
    pd.DataFrame(coef_dump).to_csv(OUT_DIR / "eb_basket_coefs.csv", index=False)

    # ACF
    ac = autocorr_returns(only_mid, TARGET)
    ac.to_csv(OUT_DIR / "eb_autocorr.csv", index=False)
    print("\n=== ACF of EB returns (lags 1-20) ===")
    print(ac.to_string(index=False))


if __name__ == "__main__":
    main()
