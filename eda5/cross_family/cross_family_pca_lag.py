"""
Two follow-ups to cross_family_eda.py:

(1) PCA on 10-tick returns of all 50 symbols.
    If two products from different families load on the same factor,
    a hedged spread on that factor could be tradable even when pairwise
    corr is low.

(2) Lead-lag: for each cross-family pair with |corr_ret10| in the top 5%,
    compute corr(ret10_a(t), ret10_b(t+lag)) for lag in -20..+20 ticks.
    A peak away from lag=0 = predictive signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT_DIR = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/cross_family")

FAMILIES = (
    "GALAXY_SOUNDS", "MICROCHIP", "OXYGEN_SHAKE", "PANEL", "PEBBLES",
    "ROBOT", "SLEEP_POD", "SNACKPACK", "TRANSLATOR", "UV_VISOR",
)
DAYS = (2, 3, 4)


def family_of(sym: str) -> str:
    for f in FAMILIES:
        if sym.startswith(f + "_"):
            return f
    return ""


def load_mids() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        sub = df[["timestamp", "product", "mid_price"]].copy()
        sub["timestamp"] = sub["timestamp"] + (d - DAYS[0]) * 1_000_000_0
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(
        index="timestamp", columns="product", values="mid_price", aggfunc="last"
    )
    return wide.sort_index().ffill().bfill()


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    sd = df.std(ddof=1).replace(0, np.nan)
    return (df - df.mean()) / sd


def pca_top(returns: pd.DataFrame, k: int = 5):
    z = standardize(returns).dropna(how="any")
    X = z.values
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    var_explained = vals / vals.sum()
    loadings = pd.DataFrame(
        vecs[:, :k],
        index=z.columns,
        columns=[f"PC{i+1}" for i in range(k)],
    )
    return loadings, var_explained[:k]


def lead_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 20) -> pd.Series:
    """corr(a_t, b_{t+lag}) for lag in [-max_lag, max_lag]."""
    a = pd.Series(a)
    b = pd.Series(b)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = a[: len(a) - lag] if lag > 0 else a
            y = b[lag:].reset_index(drop=True)
        else:
            x = a[-lag:].reset_index(drop=True)
            y = b[: len(b) + lag]
        n = min(len(x), len(y))
        if n < 200:
            out[lag] = np.nan
            continue
        out[lag] = float(np.corrcoef(x[:n], y[:n])[0, 1])
    return pd.Series(out)


def main():
    mids = load_mids()
    ret10 = mids.diff(10).iloc[10:]

    # ---- PCA on 10-tick returns ----
    loadings, var_exp = pca_top(ret10, k=6)
    loadings["family"] = [family_of(s) for s in loadings.index]
    loadings.to_csv(OUT_DIR / "cross_family_pca_loadings.csv")
    pd.Series(var_exp, name="var_explained",
              index=[f"PC{i+1}" for i in range(len(var_exp))]).to_csv(
        OUT_DIR / "cross_family_pca_variance.csv"
    )

    # For each PC, list top symbols by |loading| and show how many families they span
    pc_summary = []
    for pc in [c for c in loadings.columns if c.startswith("PC")]:
        s = loadings[[pc, "family"]].copy()
        s["abs"] = s[pc].abs()
        top = s.sort_values("abs", ascending=False).head(10)
        fams = sorted(set(top["family"]))
        pc_summary.append({
            "pc": pc,
            "var_explained": float(var_exp[int(pc[2:]) - 1]),
            "n_families_in_top10": len(fams),
            "families_in_top10": ",".join(fams),
            "top_symbols": ",".join(top.index.tolist()),
        })
    pd.DataFrame(pc_summary).to_csv(OUT_DIR / "cross_family_pca_summary.csv", index=False)

    # ---- Lead-lag on top correlated cross-family pairs ----
    pairs_df = pd.read_csv(OUT_DIR / "cross_family_corr_returns_10.csv")
    pairs_df = pairs_df.head(40)  # top ~3.5%
    rows = []
    for _, r in pairs_df.iterrows():
        a, b = r["a"], r["b"]
        ll = lead_lag_corr(ret10[a].values, ret10[b].values, max_lag=20)
        # find lag with strongest abs corr
        best_lag = int(ll.abs().idxmax())
        best_val = float(ll.loc[best_lag])
        zero_val = float(ll.loc[0])
        rows.append({
            "a": a, "b": b,
            "family_a": r["family_a"], "family_b": r["family_b"],
            "corr_lag0": zero_val,
            "best_lag": best_lag,
            "corr_at_best_lag": best_val,
            "improvement_vs_lag0": abs(best_val) - abs(zero_val),
        })
    df_ll = pd.DataFrame(rows).sort_values("improvement_vs_lag0", ascending=False)
    df_ll.to_csv(OUT_DIR / "cross_family_leadlag.csv", index=False)

    # Markdown digest
    lines = []
    lines.append("# PCA + lead-lag follow-up")
    lines.append("")
    lines.append("## PCA on 10-tick returns")
    lines.append("")
    lines.append(pd.DataFrame(pc_summary).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Lead-lag: top 40 cross-family pairs")
    lines.append("Negative `best_lag` = `a` leads `b`.  Positive = `b` leads `a`.")
    lines.append("")
    lines.append(df_ll.head(20).to_markdown(index=False, floatfmt=".4f"))

    (OUT_DIR / "cross_family_pca_lag.md").write_text("\n".join(lines))
    print("[done]", OUT_DIR)


if __name__ == "__main__":
    main()
