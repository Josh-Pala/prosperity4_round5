"""
MINT deep dive — find any predictive relationship to lift it from break-even.

Hypotheses to test:
  H1. MINT correlates with some other symbol's lagged returns (lead-lag).
  H2. MINT mid mean-reverts to a basket / linear combo of others (cointegration).
  H3. MINT has predictable autocorrelation patterns at specific lags.
  H4. MINT's order-flow imbalance predicts its own next-tick mid (already
      tested as ~0.05 — see if a lag or non-linearity helps).

Outputs:
  - mint_lagged_corr.csv         lead/lag corr with all other symbols
  - mint_basket_regression.txt   OLS of MINT mid on best predictors
  - mint_autocorr.csv            ACF on returns
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent
TARGET = "OXYGEN_SHAKE_MINT"


def load_all_mids() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df["t"] = d * 1_000_000 + df["timestamp"]
        frames.append(df[["t", "product", "mid_price"]])
    long = pd.concat(frames, ignore_index=True)
    return long.pivot_table(index="t", columns="product", values="mid_price").sort_index()


def lead_lag_correlations(mids: pd.DataFrame, target: str, lags=(-50, -20, -10, -5, -1, 1, 5, 10, 20, 50)) -> pd.DataFrame:
    """For each other symbol s, compute corr(Δmid_target_t, Δmid_s_{t+lag})."""
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


def basket_regression(mids: pd.DataFrame, target: str, top_k: int = 8) -> tuple[str, np.ndarray]:
    """Fit OLS of MINT mid on the top_k symbols by abs(level corr).
    Then check if the residual mean-reverts (good for pair trading)."""
    levels_corr = mids.corrwith(mids[target]).drop(target).abs().sort_values(ascending=False)
    top = levels_corr.head(top_k).index.tolist()
    X = mids[top].dropna()
    y = mids[target].loc[X.index]
    Xm = np.column_stack([np.ones(len(X)), X.values])
    coef, *_ = np.linalg.lstsq(Xm, y.values, rcond=None)
    pred = Xm @ coef
    resid = y.values - pred
    msg = []
    msg.append(f"Top {top_k} level correlations with {target}:")
    for s, c in levels_corr.head(top_k).items():
        msg.append(f"  {s:35s} corr={c:.3f}")
    msg.append("")
    msg.append("OLS coefficients:")
    msg.append(f"  intercept = {coef[0]:.2f}")
    for s, b in zip(top, coef[1:]):
        msg.append(f"  {s:35s} beta={b:+.4f}")
    msg.append("")
    msg.append(f"Residual: mean={resid.mean():.2f}  sd={resid.std():.2f}  R²={1 - resid.var()/y.var():.4f}")
    # Half-life of residual
    dx = np.diff(resid)
    A = np.column_stack([np.ones(len(dx)), resid[:-1] - resid[:-1].mean()])
    cc, *_ = np.linalg.lstsq(A, dx, rcond=None)
    phi = cc[1]
    if phi < 0 and phi > -1:
        hl = -np.log(2) / np.log(1 + phi)
        msg.append(f"Residual half-life: {hl:.0f} ticks")
    else:
        msg.append(f"Residual half-life: ∞ (phi={phi:.4f})")
    return "\n".join(msg), resid


def autocorr_returns(mids: pd.DataFrame, target: str, lags=range(1, 21)) -> pd.DataFrame:
    r = mids[target].diff().dropna()
    rows = []
    for L in lags:
        rows.append({"lag": L, "acf": round(r.autocorr(lag=L), 4)})
    return pd.DataFrame(rows)


def main():
    mids = load_all_mids()
    print(f"Loaded {len(mids)} ticks × {len(mids.columns)} symbols")

    # H1 — lead-lag
    ll = lead_lag_correlations(mids, TARGET)
    ll["max_abs"] = ll.iloc[:, 1:].abs().max(axis=1)
    ll = ll.sort_values("max_abs", ascending=False)
    ll.to_csv(OUT_DIR / "mint_lagged_corr.csv", index=False)
    print("\n=== TOP 15 lead-lag correlations of MINT vs other symbols ===")
    print(ll.head(15).to_string(index=False))

    # H2 — basket regression
    msg, resid = basket_regression(mids, TARGET, top_k=8)
    (OUT_DIR / "mint_basket_regression.txt").write_text(msg)
    print("\n=== BASKET REGRESSION (MINT on top-8 corr symbols) ===")
    print(msg)

    # H3 — autocorrelation
    ac = autocorr_returns(mids, TARGET)
    ac.to_csv(OUT_DIR / "mint_autocorr.csv", index=False)
    print("\n=== ACF of MINT 1-tick returns (lags 1-20) ===")
    print(ac.to_string(index=False))


if __name__ == "__main__":
    main()
