"""Aggregate MB sweep results from /tmp/mb_sweep into a CSV/markdown table."""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict

SWEEP_DIR = Path("/tmp/mb_sweep")
OUT_DIR = Path(__file__).resolve().parent
BASELINE_TOTAL_3D = 851_472
BASELINE_MB_3D = 15_221


def parse(text: str) -> tuple[int | None, int | None]:
    mb, tot = None, None
    for line in text.splitlines():
        if "MORNING_BREATH" in line:
            m = re.search(r"(-?[\d,]+)$", line.strip())
            if m:
                mb = int(m.group(1).replace(",", ""))
        elif line.startswith("Total profit:"):
            m = re.search(r"(-?[\d,]+)$", line.strip())
            if m:
                tot = int(m.group(1).replace(",", ""))
    return mb, tot


# variant -> {day: (mb, total)}
results: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)

for p in sorted(SWEEP_DIR.glob("mb_*.txt")):
    name = p.stem  # e.g. mb_top3_t200_5-2
    m = re.match(r"(mb_(?:top3|top4|pebbles)_t\d+)_(5-\d)$", name)
    if not m:
        continue
    variant, day = m.group(1), m.group(2)
    mb, tot = parse(p.read_text())
    if mb is not None and tot is not None:
        results[variant][day] = (mb, tot)

rows = []
for variant, by_day in sorted(results.items()):
    if len(by_day) != 3:
        continue
    mb_3d = sum(v[0] for v in by_day.values())
    tot_3d = sum(v[1] for v in by_day.values())
    parts = variant.split("_")
    basket = parts[1]
    threshold = int(parts[2][1:])
    rows.append({
        "basket": basket,
        "threshold": threshold,
        "MB_d2": by_day["5-2"][0],
        "MB_d3": by_day["5-3"][0],
        "MB_d4": by_day["5-4"][0],
        "MB_3d": mb_3d,
        "Total_3d": tot_3d,
        "delta_vs_base": tot_3d - BASELINE_TOTAL_3D,
        "MB_delta": mb_3d - BASELINE_MB_3D,
    })

rows.sort(key=lambda r: (r["basket"], r["threshold"]))

csv_lines = ["basket,threshold,MB_d2,MB_d3,MB_d4,MB_3d,Total_3d,delta_vs_base,MB_delta"]
for r in rows:
    csv_lines.append(",".join(str(r[k]) for k in
                              ["basket", "threshold", "MB_d2", "MB_d3", "MB_d4",
                               "MB_3d", "Total_3d", "delta_vs_base", "MB_delta"]))
(OUT_DIR / "mb_sweep_results.csv").write_text("\n".join(csv_lines) + "\n")

# Pretty markdown
md = ["| basket | thr | MB_d2 | MB_d3 | MB_d4 | MB_3d | Total_3d | Δ vs base | Δ MB |",
      "|--------|----:|------:|------:|------:|------:|---------:|----------:|-----:|"]
for r in rows:
    md.append(
        f"| {r['basket']} | {r['threshold']} | {r['MB_d2']:,} | {r['MB_d3']:,} | "
        f"{r['MB_d4']:,} | {r['MB_3d']:,} | {r['Total_3d']:,} | "
        f"{r['delta_vs_base']:+,} | {r['MB_delta']:+,} |"
    )
(OUT_DIR / "mb_sweep_table.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nBaseline: Total_3d = {BASELINE_TOTAL_3D:,}  MB_3d = {BASELINE_MB_3D:,}")
print(f"\n{len(rows)} variants aggregated. Top 5 by Total_3d:")
for r in sorted(rows, key=lambda x: -x["Total_3d"])[:5]:
    print(f"  {r['basket']:8s} t={r['threshold']:<4} Total={r['Total_3d']:,}  Δ={r['delta_vs_base']:+,}  MB={r['MB_3d']:,}")
