"""
MORNING_BREATH deep dive — replicate MINT workflow on OXYGEN_SHAKE_MORNING_BREATH.

Outputs:
  - mb_lagged_corr.csv         lead/lag corr with all other symbols
  - mb_basket_regression.txt   OLS of MB mid on best predictors (top8) + variants
  - mb_autocorr.csv            ACF on returns
  - mb_holdout.txt             walk-forward (train 2 days, test 3rd) per basket
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent
TARGET = "OXYGEN_SHAKE_MORNING_BREATH"


def load_all_mids() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df["t"] = d * 1_000_000 + df["timestamp"]
        df["day"] = d
        frames.append(df[["t", "day", "product", "mid_price"]])
    long = pd.concat(frames, ignore_index=True)
    pivot = long.pivot_table(index=["t", "day"], columns="product", values="mid_price").sort_index()
    pivot = pivot.reset_index().set_index("t")
    return pivot


def lead_lag_correlations(mids: pd.DataFrame, target: str,
                          lags=(-50, -20, -10, -5, -1, 1, 5, 10, 20, 50)) -> pd.DataFrame:
    rets = mids.diff()
    target_ret = rets[target]
    rows = []
    for s in mids.columns:
        if s == target or s == "day":
            continue
        row = {"symbol": s}
        for L in lags:
            shifted = rets[s].shift(-L)
            row[f"lag_{L:+d}"] = round(target_ret.corr(shifted), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xm = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return coef


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xm = np.column_stack([np.ones(len(X)), X])
    return Xm @ coef


def basket_regression(mids: pd.DataFrame, target: str, top_k: int = 8) -> tuple[str, list[str]]:
    candidate_cols = [c for c in mids.columns if c not in (target, "day")]
    levels_corr = mids[candidate_cols].corrwith(mids[target]).abs().sort_values(ascending=False)
    top = levels_corr.head(top_k).index.tolist()
    X = mids[top].dropna()
    y = mids[target].loc[X.index]
    coef = fit_ols(X.values, y.values)
    pred = predict(coef, X.values)
    resid = y.values - pred
    msg = []
    msg.append(f"Top {top_k} level correlations with {target}:")
    for s, c in levels_corr.head(top_k).items():
        msg.append(f"  {s:35s} corr={c:.3f}")
    msg.append("")
    msg.append("OLS coefficients (top8 full sample):")
    msg.append(f"  intercept = {coef[0]:.2f}")
    for s, b in zip(top, coef[1:]):
        msg.append(f"  {s:35s} beta={b:+.4f}")
    msg.append("")
    msg.append(f"Residual: mean={resid.mean():.2f}  sd={resid.std():.2f}  R²={1 - resid.var()/y.var():.4f}")
    dx = np.diff(resid)
    A = np.column_stack([np.ones(len(dx)), resid[:-1] - resid[:-1].mean()])
    cc, *_ = np.linalg.lstsq(A, dx, rcond=None)
    phi = cc[1]
    if -1 < phi < 0:
        hl = -np.log(2) / np.log(1 + phi)
        msg.append(f"Residual half-life: {hl:.0f} ticks")
    else:
        msg.append(f"Residual half-life: ∞ (phi={phi:.4f})")
    return "\n".join(msg), top


def holdout_eval(mids: pd.DataFrame, target: str, basket: list[str], label: str) -> str:
    """Train on 2 days, test on the 3rd. Return text block."""
    days = [2, 3, 4]
    out = [f"\n--- Basket: {label}  ({', '.join(basket)}) ---"]
    full_X = mids[basket].dropna()
    full_y = mids[target].loc[full_X.index]
    coef_full = fit_ols(full_X.values, full_y.values)
    resid_full = full_y.values - predict(coef_full, full_X.values)
    r2_full = 1 - resid_full.var() / full_y.values.var()
    out.append(f"  Full-sample: R²={r2_full:.4f}  sd_in={resid_full.std():.1f}")
    out.append(f"  Coefs: intercept={coef_full[0]:.2f}  " +
               "  ".join(f"{s}={b:+.4f}" for s, b in zip(basket, coef_full[1:])))

    for hold in days:
        train_idx = mids["day"].isin([d for d in days if d != hold])
        test_idx = mids["day"] == hold
        Xtr = mids.loc[train_idx, basket].dropna()
        ytr = mids.loc[Xtr.index, target]
        Xte = mids.loc[test_idx, basket].dropna()
        yte = mids.loc[Xte.index, target]
        coef = fit_ols(Xtr.values, ytr.values)
        # in-sample on train
        resid_in = ytr.values - predict(coef, Xtr.values)
        # OOS on hold
        resid_oos = yte.values - predict(coef, Xte.values)
        out.append(
            f"  hold=d{hold}: sd_in={resid_in.std():.1f}  sd_oos={resid_oos.std():.1f}  "
            f"mean_oos={resid_oos.mean():+.1f}"
        )
    return "\n".join(out)


def autocorr_returns(mids: pd.DataFrame, target: str, lags=range(1, 21)) -> pd.DataFrame:
    r = mids[target].diff().dropna()
    rows = [{"lag": L, "acf": round(r.autocorr(lag=L), 4)} for L in lags]
    return pd.DataFrame(rows)


def main():
    mids = load_all_mids()
    print(f"Loaded {len(mids)} ticks × {len(mids.columns)-1} symbols")

    # H1 — lead-lag
    ll = lead_lag_correlations(mids, TARGET)
    ll["max_abs"] = ll.iloc[:, 1:].abs().max(axis=1)
    ll = ll.sort_values("max_abs", ascending=False)
    ll.to_csv(OUT_DIR / "mb_lagged_corr.csv", index=False)
    print("\n=== TOP 15 lead-lag correlations of MORNING_BREATH ===")
    print(ll.head(15).to_string(index=False))

    # H2 — basket regression
    msg, top8 = basket_regression(mids, TARGET, top_k=8)
    (OUT_DIR / "mb_basket_regression.txt").write_text(msg)
    print("\n=== BASKET REGRESSION (MB on top-8 corr symbols) ===")
    print(msg)

    # Walk-forward holdout: top8, top5, top4, top3 + family-only baskets
    candidate_cols = [c for c in mids.columns if c not in (TARGET, "day")]
    levels_corr = mids[candidate_cols].corrwith(mids[TARGET]).abs().sort_values(ascending=False)
    top_full = levels_corr.head(8).index.tolist()

    baskets: list[tuple[str, list[str]]] = [
        ("top8", top_full[:8]),
        ("top5", top_full[:5]),
        ("top4", top_full[:4]),
        ("top3", top_full[:3]),
    ]
    # Family baskets — top3 within each family in top8
    by_family: dict[str, list[str]] = {}
    for s in top_full:
        fam = s.split("_")[0] if not s.startswith(("OXYGEN_SHAKE", "SLEEP_POD", "UV_VISOR", "GALAXY_SOUNDS")) else "_".join(s.split("_")[:-1])
        # Robust family extraction
        for prefix in ("OXYGEN_SHAKE", "SLEEP_POD", "UV_VISOR", "GALAXY_SOUNDS",
                       "MICROCHIP", "PANEL", "PEBBLES", "ROBOT", "SNACKPACK", "TRANSLATOR"):
            if s.startswith(prefix + "_") or s == prefix:
                fam = prefix
                break
        by_family.setdefault(fam, []).append(s)
    # also scan a wider candidate list for any family with >=2 in top16
    top16 = levels_corr.head(16).index.tolist()
    by_family_wide: dict[str, list[str]] = {}
    for s in top16:
        for prefix in ("OXYGEN_SHAKE", "SLEEP_POD", "UV_VISOR", "GALAXY_SOUNDS",
                       "MICROCHIP", "PANEL", "PEBBLES", "ROBOT", "SNACKPACK", "TRANSLATOR"):
            if s.startswith(prefix + "_"):
                by_family_wide.setdefault(prefix, []).append(s)
                break

    print("\nFamily groupings within top16:")
    for fam, syms in by_family_wide.items():
        print(f"  {fam}: {syms}")
        if len(syms) >= 2:
            baskets.append((f"{fam}-only({len(syms)})", syms))

    holdout_msg = [f"Walk-forward holdout (train 2 days, test 3rd) — target = {TARGET}"]
    for label, basket in baskets:
        holdout_msg.append(holdout_eval(mids, TARGET, basket, label))
    holdout_text = "\n".join(holdout_msg)
    (OUT_DIR / "mb_holdout.txt").write_text(holdout_text)
    print("\n=== HOLDOUT EVAL ===")
    print(holdout_text)

    # H3 — autocorrelation
    ac = autocorr_returns(mids, TARGET)
    ac.to_csv(OUT_DIR / "mb_autocorr.csv", index=False)
    print("\n=== ACF of MORNING_BREATH 1-tick returns (lags 1-20) ===")
    print(ac.to_string(index=False))


if __name__ == "__main__":
    main()
