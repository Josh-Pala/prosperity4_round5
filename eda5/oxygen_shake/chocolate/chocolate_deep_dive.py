"""
CHOCOLATE deep dive — replica del workflow MINT per OXYGEN_SHAKE_CHOCOLATE.

Hypotheses:
  H1. CHOCOLATE correla con returns laggati di altri simboli (lead-lag).
  H2. CHOCOLATE mid mean-reverte verso un basket lineare cross-family.
  H3. ACF significativa sui returns 1-tick.

Outputs:
  - chocolate_lagged_corr.csv         lead/lag corr con tutti gli altri simboli
  - chocolate_basket_regression.txt   OLS top-N + walk-forward holdout
  - chocolate_autocorr.csv            ACF 1-20
  - chocolate_holdout_table.csv       sd OOS per ogni basket candidato
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent
TARGET = "OXYGEN_SHAKE_CHOCOLATE"


def load_all_mids() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df["t"] = d * 1_000_000 + df["timestamp"]
        df["day"] = d
        frames.append(df[["t", "day", "product", "mid_price"]])
    long = pd.concat(frames, ignore_index=True)
    return long.pivot_table(index=["t", "day"], columns="product", values="mid_price").sort_index()


def lead_lag_correlations(mids: pd.DataFrame, target: str,
                          lags=(-50, -20, -10, -5, -1, 1, 5, 10, 20, 50)) -> pd.DataFrame:
    rets = mids.diff()
    target_ret = rets[target]
    rows = []
    for s in mids.columns:
        if s == target:
            continue
        row = {"symbol": s}
        for L in lags:
            shifted = rets[s].shift(-L)
            row[f"lag_{L:+d}"] = round(target_ret.corr(shifted), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_ols(X: pd.DataFrame, y: pd.Series):
    """Return (coef, resid, R2, sd)."""
    Xm = np.column_stack([np.ones(len(X)), X.values])
    coef, *_ = np.linalg.lstsq(Xm, y.values, rcond=None)
    pred = Xm @ coef
    resid = y.values - pred
    r2 = 1 - resid.var() / y.var()
    return coef, resid, r2, resid.std()


def basket_regression(mids: pd.DataFrame, target: str, top_k: int = 8) -> tuple[str, list[str], np.ndarray]:
    levels_corr = mids.corrwith(mids[target]).drop(target).abs().sort_values(ascending=False)
    top = levels_corr.head(top_k).index.tolist()
    X = mids[top].dropna()
    y = mids[target].loc[X.index]
    coef, resid, r2, sd = fit_ols(X, y)
    msg = []
    msg.append(f"Top {top_k} level correlations with {target}:")
    for s, c in levels_corr.head(top_k).items():
        msg.append(f"  {s:35s} corr={c:.3f}")
    msg.append("")
    msg.append("OLS coefficients (full sample):")
    msg.append(f"  intercept = {coef[0]:.2f}")
    for s, b in zip(top, coef[1:]):
        msg.append(f"  {s:35s} beta={b:+.4f}")
    msg.append("")
    msg.append(f"Residual: mean={resid.mean():.2f}  sd={sd:.2f}  R2={r2:.4f}")
    dx = np.diff(resid)
    A = np.column_stack([np.ones(len(dx)), resid[:-1] - resid[:-1].mean()])
    cc, *_ = np.linalg.lstsq(A, dx, rcond=None)
    phi = cc[1]
    if phi < 0 and phi > -1:
        hl = -np.log(2) / np.log(1 + phi)
        msg.append(f"Residual half-life: {hl:.0f} ticks")
    else:
        msg.append(f"Residual half-life: inf (phi={phi:.4f})")
    return "\n".join(msg), top, levels_corr


def walk_forward_holdout(mids_with_day: pd.DataFrame, target: str, basket: list[str], label: str) -> dict:
    """Train on 2 days, test on the 3rd. Return per-day OOS sd."""
    days = (2, 3, 4)
    out = {"basket": label, "k": len(basket)}
    # In-sample full-fit diagnostics
    X_full = mids_with_day[basket].dropna()
    y_full = mids_with_day[target].loc[X_full.index]
    coef_full, resid_full, r2_full, sd_full = fit_ols(X_full, y_full)
    out["R2_in"] = round(r2_full, 4)
    out["sd_in"] = round(sd_full, 2)
    for hold in days:
        train_mask = mids_with_day.index.get_level_values("day") != hold
        test_mask = ~train_mask
        Xtr = mids_with_day.loc[train_mask, basket].dropna()
        ytr = mids_with_day.loc[Xtr.index, target]
        coef, _, _, _ = fit_ols(Xtr, ytr)
        Xte = mids_with_day.loc[test_mask, basket].dropna()
        yte = mids_with_day.loc[Xte.index, target]
        Xte_m = np.column_stack([np.ones(len(Xte)), Xte.values])
        pred = Xte_m @ coef
        resid_oos = yte.values - pred
        out[f"sd_oos_d{hold}"] = round(resid_oos.std(), 2)
    return out


def autocorr_returns(mids: pd.DataFrame, target: str, lags=range(1, 21)) -> pd.DataFrame:
    r = mids[target].diff().dropna()
    rows = []
    for L in lags:
        rows.append({"lag": L, "acf": round(r.autocorr(lag=L), 4)})
    return pd.DataFrame(rows)


def main():
    mids = load_all_mids()
    # mids has MultiIndex (t, day). For correlation drop day level.
    mids_plain = mids.copy()
    mids_plain.index = mids_plain.index.get_level_values("t")
    print(f"Loaded {len(mids_plain)} ticks x {len(mids_plain.columns)} symbols")

    # --- H1: lead-lag
    ll = lead_lag_correlations(mids_plain, TARGET)
    ll["max_abs"] = ll.iloc[:, 1:].abs().max(axis=1)
    ll = ll.sort_values("max_abs", ascending=False)
    ll.to_csv(OUT_DIR / "chocolate_lagged_corr.csv", index=False)
    print("\n=== TOP 15 lead-lag correlations of CHOCOLATE vs other symbols ===")
    print(ll.head(15).to_string(index=False))

    # --- H2: basket regression top-8
    msg, top8, levels_corr = basket_regression(mids_plain, TARGET, top_k=8)
    (OUT_DIR / "chocolate_basket_regression.txt").write_text(msg)
    print("\n=== BASKET REGRESSION (CHOCOLATE on top-8 corr symbols) ===")
    print(msg)

    # --- Candidate baskets for walk-forward
    top_all = levels_corr.head(8).index.tolist()
    candidates: dict[str, list[str]] = {
        "top8": top_all[:8],
        "top5": top_all[:5],
        "top4": top_all[:4],
        "top3": top_all[:3],
    }
    # Family-only baskets: see if a single family dominates top-8
    fam_groups: dict[str, list[str]] = {}
    for s in top_all:
        fam = s.rsplit("_", 1)[0] if not s.startswith("UV_VISOR") else "UV_VISOR"
        # generic family extraction: split on first underscore family-prefix
    # Simpler: group by known family prefix
    FAMILIES = ["GALAXY_SOUNDS", "MICROCHIP", "OXYGEN_SHAKE", "PANEL", "PEBBLES",
                "ROBOT", "SLEEP_POD", "SNACKPACK", "TRANSLATOR", "UV_VISOR"]
    for s in top_all:
        for f in FAMILIES:
            if s.startswith(f + "_"):
                fam_groups.setdefault(f, []).append(s)
                break
    for f, syms in fam_groups.items():
        if len(syms) >= 2:
            candidates[f"{f}_only"] = syms

    # Also try top-N from each family across full universe (not just top-8)
    for f in FAMILIES:
        fam_corrs = levels_corr[levels_corr.index.str.startswith(f + "_")]
        if len(fam_corrs) >= 2:
            # all variants of that family that are not the target
            fam_syms = [s for s in fam_corrs.index if s != TARGET][:5]
            key = f"{f}_full"
            if key not in candidates and len(fam_syms) >= 2:
                candidates[key] = fam_syms

    # Walk-forward holdout on each candidate
    rows = []
    for label, basket in candidates.items():
        try:
            row = walk_forward_holdout(mids, TARGET, basket, label)
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"Failed {label}: {e}")
    ho = pd.DataFrame(rows)
    ho.to_csv(OUT_DIR / "chocolate_holdout_table.csv", index=False)
    print("\n=== WALK-FORWARD HOLDOUT (train 2 days, test 3rd) ===")
    print(ho.to_string(index=False))

    # --- H3: ACF
    ac = autocorr_returns(mids_plain, TARGET)
    ac.to_csv(OUT_DIR / "chocolate_autocorr.csv", index=False)
    print("\n=== ACF of CHOCOLATE 1-tick returns (lags 1-20) ===")
    print(ac.to_string(index=False))


if __name__ == "__main__":
    main()
