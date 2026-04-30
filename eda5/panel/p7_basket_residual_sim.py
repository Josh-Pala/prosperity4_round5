"""Simulate Multi-Leg PANEL basket residual strategy.

For each PANEL p, compute fair_p = a0 + sum_i b_i * mid_other_i (OLS on training day).
Trade the residual = mid_p - fair_p, z-score normalized over rolling window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data_ROUND_5"
PANELS = ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"]


def load_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{day}.csv", sep=";")
    df = df[df["product"].isin(PANELS)]
    return df.pivot_table(index="timestamp", columns="product", values="mid_price").dropna()


def fit_basket(mid: pd.DataFrame):
    """Per-panel OLS: mid_p = a + B * mid_others. Return dict[panel] -> (a, b_vec, others)."""
    coefs = {}
    for p in PANELS:
        others = [o for o in PANELS if o != p]
        X = np.column_stack([np.ones(len(mid)), mid[others].values])
        y = mid[p].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = beta[0]
        b = beta[1:]
        coefs[p] = (a, b, others)
    return coefs


def simulate(day_train: int, day_test: int, entry_z: float = 2.0, exit_z: float = 0.3) -> dict:
    train = load_day(day_train)
    coefs = fit_basket(train)
    # Compute residual mean/std from train
    resid_stats = {}
    for p in PANELS:
        a, b, others = coefs[p]
        fair = a + train[others].values @ b
        r = train[p].values - fair
        resid_stats[p] = (r.mean(), r.std())

    test = load_day(day_test)
    pnls = {}
    for p in PANELS:
        a, b, others = coefs[p]
        fair = a + test[others].values @ b
        resid = test[p].values - fair
        m, s = resid_stats[p]
        if s < 1e-9:
            pnls[p] = 0.0
            continue
        z = (resid - m) / s
        # naive trade: enter at |z|>entry_z, exit at |z|<exit_z, size=10 on this leg only
        pos = 0
        entry_px = 0.0
        pnl = 0.0
        for i, zi in enumerate(z):
            px = test[p].values[i]
            if pos == 0:
                if zi > entry_z:
                    pos = -1
                    entry_px = px
                elif zi < -entry_z:
                    pos = +1
                    entry_px = px
            else:
                if abs(zi) < exit_z:
                    pnl += pos * (px - entry_px) * 10
                    pos = 0
        pnls[p] = round(pnl, 0)
    return pnls


def main():
    print("Train day 2 -> test day 3:")
    print(simulate(2, 3))
    print("\nTrain day 2 -> test day 4:")
    print(simulate(2, 4))
    print("\nTrain day 3 -> test day 4:")
    print(simulate(3, 4))
    print("\nTrain day 3 -> test day 2:")
    print(simulate(3, 2))
    # In-sample sanity
    print("\nIn-sample (train=test=day 2):")
    print(simulate(2, 2))


if __name__ == "__main__":
    main()
