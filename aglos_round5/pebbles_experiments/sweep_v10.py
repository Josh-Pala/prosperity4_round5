"""Sweep entry_z combinations on v10 (two pairs + MM)."""
import subprocess
import re
from pathlib import Path

BTX = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_windteam/.venv/bin/prosperity4btx"
DATA = "/tmp/p4data"
SRC = Path("v10_two_pairs_tuned.py")
TPL = SRC.read_text()
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]


def patch(z1, z2, half=10):
    s = TPL
    s = re.sub(r"ENTRY_Z_P1 = [\d.]+", f"ENTRY_Z_P1 = {z1}", s)
    s = re.sub(r"ENTRY_Z_P2 = [\d.]+", f"ENTRY_Z_P2 = {z2}", s)
    s = re.sub(r"HALF_QUOTE = \d+", f"HALF_QUOTE = {half}", s)
    return s


def run_once(z1, z2, half=10):
    SRC.write_text(patch(z1, z2, half))
    daily = []
    peb_legs = {p: 0 for p in PEBBLES}
    for d in [2, 3, 4]:
        r = subprocess.run([BTX, str(SRC.resolve()), f"5-{d}", "--no-out", "--data", DATA, "--no-progress"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("Total profit:"):
                daily.append(int(line.split(":")[1].replace(",", "").strip()))
            for p in PEBBLES:
                if line.startswith(f"{p}:"):
                    peb_legs[p] += int(line.split(":")[1].replace(",", "").strip())
    return daily, peb_legs


configs = [
    (1.5, 1.5), (1.5, 2.0), (1.5, 2.5),
    (2.0, 1.5), (2.0, 2.0), (2.0, 2.5),
    (2.5, 1.5), (2.5, 2.0), (2.5, 2.5),
    (1.0, 2.0), (1.0, 1.5), (1.8, 1.8), (1.8, 2.2),
]

print(f"{'z1':>5} {'z2':>5} {'d2':>8} {'d3':>8} {'d4':>8} {'total':>10}  legs(XS,S,M,L,XL)")
results = []
for z1, z2 in configs:
    daily, legs = run_once(z1, z2)
    tot = sum(daily)
    leg_str = ",".join(f"{legs[p]:>6d}" for p in PEBBLES)
    print(f"{z1:>5} {z2:>5} {daily[0]:>8} {daily[1]:>8} {daily[2]:>8} {tot:>10}  ({leg_str})")
    results.append((z1, z2, tot, daily))

print(f"\nBest: {max(results, key=lambda x: x[2])}")
SRC.write_text(TPL)  # restore
