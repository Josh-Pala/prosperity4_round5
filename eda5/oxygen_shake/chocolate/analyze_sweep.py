"""Aggregate the sweep results into the threshold sweep table."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
BASELINE_TOTAL = 851472
BASELINE_CHOC = 25083  # current CHOCOLATE PnL post-MINT integration


def main():
    df = pd.read_csv(HERE / "sweep_results.csv")
    df = df.dropna()
    # de-dup any duplicates from overlapping runs
    df = df.drop_duplicates(subset=["variant", "day"]).reset_index(drop=True)
    df["choc_pnl"] = df["choc_pnl"].astype(int)
    df["total"] = df["total"].astype(int)
    # parse variant -> basket, threshold
    df["basket"] = df["variant"].str.extract(r"choc_([a-z0-9]+)_t")
    df["thr"] = df["variant"].str.extract(r"_t(\d+)").astype(int)
    pivot_choc = df.pivot_table(index=["basket", "thr"], columns="day", values="choc_pnl").reset_index()
    pivot_total = df.pivot_table(index=["basket", "thr"], columns="day", values="total").reset_index()
    pivot_choc["choc_3day"] = pivot_choc[["5-2", "5-3", "5-4"]].sum(axis=1)
    pivot_total["total_3day"] = pivot_total[["5-2", "5-3", "5-4"]].sum(axis=1)
    pivot_total["delta_vs_base"] = pivot_total["total_3day"] - BASELINE_TOTAL

    summary = pivot_choc.merge(
        pivot_total[["basket", "thr", "total_3day", "delta_vs_base"]],
        on=["basket", "thr"],
    )
    summary = summary.sort_values(["basket", "thr"]).reset_index(drop=True)
    summary.columns = [str(c) for c in summary.columns]
    summary.to_csv(HERE / "sweep_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nBaseline CHOCOLATE: {BASELINE_CHOC:,}  Baseline total: {BASELINE_TOTAL:,}")
    best = summary.loc[summary["delta_vs_base"].idxmax()]
    print(f"\nBEST: basket={best['basket']} thr={best['thr']} "
          f"choc_3day={int(best['choc_3day']):,} total={int(best['total_3day']):,} "
          f"delta=+{int(best['delta_vs_base']):,}")


if __name__ == "__main__":
    main()
