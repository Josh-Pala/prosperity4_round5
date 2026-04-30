"""Extract OLS coefs (full sample) for the 2 candidate baskets we'll ship."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "Data_ROUND_5"
TARGET = "OXYGEN_SHAKE_CHOCOLATE"


def load_all_mids() -> pd.DataFrame:
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df["t"] = d * 1_000_000 + df["timestamp"]
        frames.append(df[["t", "product", "mid_price"]])
    long = pd.concat(frames, ignore_index=True)
    return long.pivot_table(index="t", columns="product", values="mid_price").sort_index()


def fit(mids, basket, label):
    X = mids[basket].dropna()
    y = mids[TARGET].loc[X.index]
    Xm = np.column_stack([np.ones(len(X)), X.values])
    coef, *_ = np.linalg.lstsq(Xm, y.values, rcond=None)
    pred = Xm @ coef
    resid = y.values - pred
    print(f"--- {label} ---")
    print(f"INTERCEPT = {coef[0]:.4f}")
    for s, b in zip(basket, coef[1:]):
        print(f'    "{s}": {b:+.4f},')
    print(f"# R2={1 - resid.var()/y.var():.4f}  resid_sd={resid.std():.2f}")
    print()


def main():
    mids = load_all_mids()
    levels_corr = mids.corrwith(mids[TARGET]).drop(TARGET).abs().sort_values(ascending=False)

    # MICROCHIP_full: tutti i 5 microchip
    microchip = [s for s in mids.columns if s.startswith("MICROCHIP_")]
    fit(mids, microchip, "MICROCHIP_full")

    # top3 cross-family
    top3 = levels_corr.head(3).index.tolist()
    fit(mids, top3, "top3")

    # top4 cross-family
    top4 = levels_corr.head(4).index.tolist()
    fit(mids, top4, "top4")


if __name__ == "__main__":
    main()
