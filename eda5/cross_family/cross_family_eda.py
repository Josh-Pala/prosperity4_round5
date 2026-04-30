"""
Cross-family EDA for IMC Prosperity 4 Round 5.

Goal: find statistical relationships between products that belong to DIFFERENT
families (FINAL_GLAUCO already exploits intra-family pairs heavily).

Pipeline:
  1. Load mids for all 50 symbols across days 2,3,4 (concat).
  2. Compute correlations on:
       - levels                  -> trivial, dominated by trend
       - 1-tick returns          -> tick-to-tick co-movement
       - 10-tick log returns     -> noise-filtered co-movement
  3. For every cross-family pair rank by |corr(returns_1)| AND |corr(returns_10)|.
  4. For the top candidates, fit y = a + b*x on mids and check the residual:
       - sd(residual)
       - half-life of AR(1) on residual  (mean-reversion speed)
       - "edge per tick" proxy = sd(d residual) / sd(residual)
  5. Persist:
       - cross_family_corr_returns_1.csv
       - cross_family_corr_returns_10.csv
       - cross_family_top_pairs.csv
       - cross_family_summary.md  (human readable digest)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT_DIR = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/cross_family")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS = (2, 3, 4)

FAMILIES = (
    "GALAXY_SOUNDS", "MICROCHIP", "OXYGEN_SHAKE", "PANEL", "PEBBLES",
    "ROBOT", "SLEEP_POD", "SNACKPACK", "TRANSLATOR", "UV_VISOR",
)


def family_of(sym: str) -> str:
    for f in FAMILIES:
        if sym.startswith(f + "_"):
            return f
    return ""


def load_mids() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        path = DATA_DIR / f"prices_round_5_day_{d}.csv"
        df = pd.read_csv(path, sep=";")
        # Keep timestamp+product+mid_price; pivot to wide
        sub = df[["timestamp", "product", "mid_price"]].copy()
        # offset timestamp so days don't overlap
        sub["timestamp"] = sub["timestamp"] + (d - DAYS[0]) * 1_000_000_0
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(
        index="timestamp", columns="product", values="mid_price", aggfunc="last"
    )
    wide = wide.sort_index()
    # Forward-fill short gaps then drop any column still partially missing
    wide = wide.ffill().bfill()
    return wide


def half_life_ar1(x: np.ndarray) -> float:
    """Half-life of AR(1) mean reversion: x_{t+1} - mu = phi * (x_t - mu) + e.
    Returns half-life in ticks. nan if not mean-reverting."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 100:
        return float("nan")
    mu = x.mean()
    y = x[1:] - mu
    z = x[:-1] - mu
    denom = float((z * z).sum())
    if denom <= 0:
        return float("nan")
    phi = float((y * z).sum() / denom)
    if phi <= 0 or phi >= 1:
        return float("nan")
    return -math.log(2.0) / math.log(phi)


def fit_residual(y: np.ndarray, x: np.ndarray) -> dict:
    """Fit y = a + b*x by OLS, return slope, intercept, residual sd, half-life."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(x)
    y = y[mask]
    x = x[mask]
    if y.size < 200:
        return {}
    x_mean = x.mean()
    y_mean = y.mean()
    cov_xy = float(((x - x_mean) * (y - y_mean)).mean())
    var_x = float(((x - x_mean) ** 2).mean())
    if var_x <= 0:
        return {}
    b = cov_xy / var_x
    a = y_mean - b * x_mean
    resid = y - (a + b * x)
    return {
        "intercept": a,
        "beta": b,
        "resid_sd": float(resid.std(ddof=1)),
        "resid_mean": float(resid.mean()),
        "half_life": half_life_ar1(resid),
        "n": int(y.size),
    }


def cross_family_pairs(symbols):
    out = []
    for i, a in enumerate(symbols):
        fa = family_of(a)
        for b in symbols[i + 1:]:
            fb = family_of(b)
            if fa and fb and fa != fb:
                out.append((a, b))
    return out


def main():
    print("[load] reading mids ...")
    mids = load_mids()
    print(f"[load] shape={mids.shape}, symbols={mids.shape[1]}")

    # Drop columns that are constant (would explode corr)
    nunique = mids.nunique()
    keep = nunique[nunique > 1].index.tolist()
    mids = mids[keep]
    print(f"[load] non-constant symbols={len(keep)}")

    # Returns
    ret1 = mids.diff().iloc[1:]
    ret10 = mids.diff(10).iloc[10:]

    # Correlation matrices on returns
    corr_r1 = ret1.corr()
    corr_r10 = ret10.corr()

    # Cross-family pairs
    pairs = cross_family_pairs(list(mids.columns))
    print(f"[pairs] cross-family candidates: {len(pairs)}")

    rows = []
    for a, b in pairs:
        r1 = corr_r1.at[a, b] if a in corr_r1.index and b in corr_r1.columns else np.nan
        r10 = corr_r10.at[a, b] if a in corr_r10.index and b in corr_r10.columns else np.nan
        rows.append((a, b, family_of(a), family_of(b), r1, r10))

    df_pairs = pd.DataFrame(
        rows, columns=["a", "b", "family_a", "family_b", "corr_ret1", "corr_ret10"]
    )
    df_pairs["abs_r1"] = df_pairs["corr_ret1"].abs()
    df_pairs["abs_r10"] = df_pairs["corr_ret10"].abs()

    df_pairs.sort_values("abs_r10", ascending=False).to_csv(
        OUT_DIR / "cross_family_corr_returns_10.csv", index=False
    )
    df_pairs.sort_values("abs_r1", ascending=False).to_csv(
        OUT_DIR / "cross_family_corr_returns_1.csv", index=False
    )

    # Pick top by 10-tick returns (smoother, less microstructure noise)
    TOP_N = 50
    top = df_pairs.sort_values("abs_r10", ascending=False).head(TOP_N).copy()

    # Fit OLS residual diagnostics on the level
    diag = []
    for _, row in top.iterrows():
        a, b = row["a"], row["b"]
        fit_ab = fit_residual(mids[a].values, mids[b].values)
        if not fit_ab:
            continue
        # also fit the inverse (b on a) for sanity
        fit_ba = fit_residual(mids[b].values, mids[a].values)
        diag.append({
            "a": a, "b": b,
            "family_a": row["family_a"], "family_b": row["family_b"],
            "corr_ret1": row["corr_ret1"], "corr_ret10": row["corr_ret10"],
            "beta_a_on_b": fit_ab.get("beta"),
            "intercept_a_on_b": fit_ab.get("intercept"),
            "resid_sd_a_on_b": fit_ab.get("resid_sd"),
            "half_life_a_on_b": fit_ab.get("half_life"),
            "beta_b_on_a": fit_ba.get("beta"),
            "resid_sd_b_on_a": fit_ba.get("resid_sd"),
            "half_life_b_on_a": fit_ba.get("half_life"),
            "n": fit_ab.get("n"),
        })

    df_top = pd.DataFrame(diag)
    df_top.to_csv(OUT_DIR / "cross_family_top_pairs.csv", index=False)

    # Family vs family aggregate (mean |corr| of returns_10) — useful map
    df_pairs["pair_family"] = df_pairs[["family_a", "family_b"]].apply(
        lambda r: " | ".join(sorted([r["family_a"], r["family_b"]])), axis=1
    )
    fam_map = (
        df_pairs.groupby("pair_family")
        .agg(mean_abs_r10=("abs_r10", "mean"),
             max_abs_r10=("abs_r10", "max"),
             n=("abs_r10", "size"))
        .sort_values("mean_abs_r10", ascending=False)
    )
    fam_map.to_csv(OUT_DIR / "cross_family_familymap.csv")

    # Markdown summary
    lines = []
    lines.append("# Cross-family EDA — Round 5")
    lines.append("")
    lines.append(f"Symbols loaded (non-constant): {len(keep)}")
    lines.append(f"Cross-family pair count: {len(pairs)}")
    lines.append("")
    lines.append("## Top 25 cross-family pairs by |corr(10-tick returns)|")
    lines.append("")
    show = df_top.sort_values("corr_ret10", key=lambda s: s.abs(), ascending=False).head(25)
    cols = ["a", "b", "family_a", "family_b", "corr_ret10", "corr_ret1",
            "beta_a_on_b", "resid_sd_a_on_b", "half_life_a_on_b"]
    lines.append(show[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Family-vs-family map (mean |corr| of 10-tick returns)")
    lines.append("")
    lines.append(fam_map.head(30).to_markdown(floatfmt=".4f"))

    (OUT_DIR / "cross_family_summary.md").write_text("\n".join(lines))
    print("[done] outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
