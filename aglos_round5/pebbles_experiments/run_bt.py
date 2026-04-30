"""Helper: run backtest of a strategy file on Round 5 days 2,3,4 and return PnL per day + per leg.

Usage:
    python3 run_bt.py v01_pair_xs_s_only.py
"""
import subprocess
import sys
import re
from pathlib import Path

BTX = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_windteam/.venv/bin/prosperity4btx"
DATA = "/tmp/p4data"
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]


def run(strategy_file: str, days=(2, 3, 4)):
    path = Path(strategy_file).resolve()
    rows = []
    for d in days:
        r = subprocess.run(
            [BTX, str(path), f"5-{d}", "--no-out", "--data", DATA, "--no-progress"],
            capture_output=True, text=True,
        )
        legs = {}
        total = None
        for line in r.stdout.splitlines():
            line = line.strip()
            for sym in PEBBLES:
                if line.startswith(f"{sym}:"):
                    legs[sym] = int(line.split(":")[1].replace(",", "").strip())
            if line.startswith("Total profit:"):
                total = int(line.split(":")[1].replace(",", "").strip())
        rows.append((d, legs, total))
    return rows


def fmt(rows, name):
    print(f"\n=== {name} ===")
    print(f"{'day':<4} {'XS':>8} {'S':>8} {'M':>8} {'L':>8} {'XL':>8} {'PEB_sum':>10} {'TOTAL':>10}")
    grand = 0
    g_legs = {p: 0 for p in PEBBLES}
    for d, legs, tot in rows:
        peb_sum = sum(legs.get(p, 0) for p in PEBBLES)
        for p in PEBBLES:
            g_legs[p] += legs.get(p, 0)
        grand += tot or 0
        print(f"{d:<4} {legs.get('PEBBLES_XS',0):>8} {legs.get('PEBBLES_S',0):>8} "
              f"{legs.get('PEBBLES_M',0):>8} {legs.get('PEBBLES_L',0):>8} "
              f"{legs.get('PEBBLES_XL',0):>8} {peb_sum:>10} {tot:>10}")
    peb_grand = sum(g_legs.values())
    print(f"{'tot':<4} {g_legs['PEBBLES_XS']:>8} {g_legs['PEBBLES_S']:>8} "
          f"{g_legs['PEBBLES_M']:>8} {g_legs['PEBBLES_L']:>8} "
          f"{g_legs['PEBBLES_XL']:>8} {peb_grand:>10} {grand:>10}")
    return grand, peb_grand


if __name__ == "__main__":
    f = sys.argv[1]
    rows = run(f)
    fmt(rows, f)
