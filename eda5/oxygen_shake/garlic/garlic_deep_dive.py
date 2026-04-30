"""
GARLIC deep dive — replicates mint_deep_dive.py for OXYGEN_SHAKE_GARLIC.

Hypotheses to test:
  H1. GARLIC mid mean-reverts to a basket / linear combo of others.
  H2. Top-K vs single-family baskets differ in OOS stability.
  H3. Lead-lag predictive returns.
  H4. ACF on returns.

Outputs:
  - garlic_lagged_corr.csv         lead/lag corr with all other symbols
  - garlic_basket_regression.txt   OLS of GARLIC mid on best predictors
  - garlic_autocorr.csv            ACF on returns
  - garlic_holdout_results.csv     walk-forward sd for candidate baskets
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent
TARGET = "OXYGEN_SHAKE_GARLIC"


def load_all_mids() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df["t"] = d * 1_000_000 + df["timestamp"]
        df["day"] = d
        frames.append(df[["t", "day", "product", "mid_price"]])
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["t", "day"], columns="product", values="mid_price").sort_index()
    wide = wide.reset_index().set_index("t")
    return wide


def lead_lag_correlations(mids: pd.DataFrame, target: str, lags=(-50, -20, -10, -5, -1, 1, 5, 10, 20, 50)) -> pd.DataFrame:
    rets = mids.drop(columns=["day"]).diff()
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
    return coef, pred


def basket_summary(mids: pd.DataFrame, target: str, basket: list[str], label: str) -> dict:
    sub = mids[basket + [target]].dropna()
    X = sub[basket].values
    y = sub[target].values
    coef, pred = fit_ols(X, y)
    resid = y - pred
    r2 = 1 - resid.var() / y.var()
    return {
        "basket": label,
        "symbols": ",".join(basket),
        "n": len(y),
        "intercept": float(coef[0]),
        "betas": dict(zip(basket, [float(b) for b in coef[1:]])),
        "resid_sd": float(resid.std()),
        "R2": float(r2),
    }


def walk_forward_holdout(mids: pd.DataFrame, target: str, basket: list[str]) -> dict:
    """Train on 2 days, test on 3rd. Report OOS resid sd per held-out day."""
    out = {}
    days = [2, 3, 4]
    for held in days:
        train_mask = mids["day"] != held
        test_mask = mids["day"] == held
        sub_tr = mids.loc[train_mask, basket + [target]].dropna()
        sub_te = mids.loc[test_mask, basket + [target]].dropna()
        if len(sub_tr) < 100 or len(sub_te) < 100:
            out[f"oos_d{held}"] = float("nan")
            continue
        coef, _ = fit_ols(sub_tr[basket].values, sub_tr[target].values)
        Xte = np.column_stack([np.ones(len(sub_te)), sub_te[basket].values])
        pred_te = Xte @ coef
        resid_te = sub_te[target].values - pred_te
        out[f"oos_d{held}"] = float(resid_te.std())
    return out


def basket_regression(mids: pd.DataFrame, target: str, top_k: int = 8) -> tuple[str, list[str]]:
    levels_corr = mids.drop(columns=["day"]).corrwith(mids[target]).drop(target).abs().sort_values(ascending=False)
    top = levels_corr.head(top_k).index.tolist()
    sub = mids[top + [target]].dropna()
    X = sub[top].values
    y = sub[target].values
    coef, pred = fit_ols(X, y)
    resid = y - pred
    msg = []
    msg.append(f"Top {top_k} level correlations with {target}:")
    for s, c in levels_corr.head(top_k).items():
        msg.append(f"  {s:35s} corr={c:.3f}")
    msg.append("")
    msg.append("OLS coefficients (top-8 full sample):")
    msg.append(f"  intercept = {coef[0]:.2f}")
    for s, b in zip(top, coef[1:]):
        msg.append(f"  {s:35s} beta={b:+.4f}")
    msg.append("")
    r2 = 1 - resid.var() / y.var()
    msg.append(f"Residual: mean={resid.mean():.2f}  sd={resid.std():.2f}  R2={r2:.4f}")
    dx = np.diff(resid)
    A = np.column_stack([np.ones(len(dx)), resid[:-1] - resid[:-1].mean()])
    cc, *_ = np.linalg.lstsq(A, dx, rcond=None)
    phi = cc[1]
    if phi < 0 and phi > -1:
        hl = -np.log(2) / np.log(1 + phi)
        msg.append(f"Residual half-life: {hl:.0f} ticks")
    else:
        msg.append(f"Residual half-life: inf (phi={phi:.4f})")
    return "\n".join(msg), top


def autocorr_returns(mids: pd.DataFrame, target: str, lags=range(1, 21)) -> pd.DataFrame:
    r = mids[target].diff().dropna()
    rows = []
    for L in lags:
        rows.append({"lag": L, "acf": round(r.autocorr(lag=L), 4)})
    return pd.DataFrame(rows)


def main():
    mids = load_all_mids()
    print(f"Loaded {len(mids)} ticks x {len(mids.columns)-1} symbols (excl day col)")

    # H3 — lead-lag
    ll = lead_lag_correlations(mids, TARGET)
    ll["max_abs"] = ll.iloc[:, 1:].abs().max(axis=1)
    ll = ll.sort_values("max_abs", ascending=False)
    ll.to_csv(OUT_DIR / "garlic_lagged_corr.csv", index=False)
    print("\n=== TOP 15 lead-lag correlations of GARLIC ===")
    print(ll.head(15).to_string(index=False))

    # H1 — basket regression top8
    msg, top8 = basket_regression(mids, TARGET, top_k=8)
    (OUT_DIR / "garlic_basket_regression.txt").write_text(msg)
    print("\n=== BASKET REGRESSION (GARLIC on top-8 corr symbols) ===")
    print(msg)

    # H2 — walk-forward across candidate baskets
    candidates: list[tuple[str, list[str]]] = []
    candidates.append(("top8", top8))
    candidates.append(("top5", top8[:5]))
    candidates.append(("top4", top8[:4]))
    candidates.append(("top3", top8[:3]))
    # group top-8 by family prefix and pick families with >=2 reps
    by_family: dict[str, list[str]] = {}
    for s in top8:
        fam = s.split("_")[0] if not s.startswith("OXYGEN") else "OXYGEN_SHAKE"
        # handle multi-token families
        if s.startswith("OXYGEN_SHAKE"):
            fam = "OXYGEN_SHAKE"
        elif s.startswith("SLEEP_POD"):
            fam = "SLEEP_POD"
        elif s.startswith("UV_VISOR"):
            fam = "UV_VISOR"
        else:
            fam = s.split("_")[0]
        by_family.setdefault(fam, []).append(s)
    for fam, syms in by_family.items():
        if len(syms) >= 2:
            candidates.append((f"{fam}-only", syms))

    rows = []
    print("\n=== WALK-FORWARD HOLDOUT (sd in / sd OOS per held-out day) ===")
    for label, basket in candidates:
        if any(s not in mids.columns for s in basket):
            continue
        in_summary = basket_summary(mids, TARGET, basket, label)
        oos = walk_forward_holdout(mids, TARGET, basket)
        row = {
            "basket": label,
            "symbols": in_summary["symbols"],
            "k": len(basket),
            "R2_in": round(in_summary["R2"], 4),
            "sd_in": round(in_summary["resid_sd"], 1),
            "sd_oos_d2": round(oos["oos_d2"], 1) if not np.isnan(oos["oos_d2"]) else None,
            "sd_oos_d3": round(oos["oos_d3"], 1) if not np.isnan(oos["oos_d3"]) else None,
            "sd_oos_d4": round(oos["oos_d4"], 1) if not np.isnan(oos["oos_d4"]) else None,
            "intercept": round(in_summary["intercept"], 2),
            "betas": " ".join(f"{k}={v:+.4f}" for k, v in in_summary["betas"].items()),
        }
        rows.append(row)
        print(
            f"  {label:20s} k={len(basket)} R2={row['R2_in']:.3f} "
            f"sd_in={row['sd_in']:.1f} oos d2/d3/d4={row['sd_oos_d2']}/{row['sd_oos_d3']}/{row['sd_oos_d4']}"
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "garlic_holdout_results.csv", index=False)

    # H4 — autocorrelation
    ac = autocorr_returns(mids, TARGET)
    ac.to_csv(OUT_DIR / "garlic_autocorr.csv", index=False)
    print("\n=== ACF of GARLIC 1-tick returns (lags 1-20) ===")
    print(ac.to_string(index=False))


if __name__ == "__main__":
    main()
