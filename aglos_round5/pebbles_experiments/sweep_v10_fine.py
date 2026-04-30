"""Fine-grained sweep around the v10 peak."""
import subprocess
import re
from pathlib import Path

BTX = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_windteam/.venv/bin/prosperity4btx"
DATA = "/tmp/p4data"
SRC = Path("v10_two_pairs_tuned.py")
TPL = SRC.read_text()
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]


def patch(z1, z2, half=10, edge=2, qsize=5):
    s = TPL
    s = re.sub(r"ENTRY_Z_P1 = [\d.]+", f"ENTRY_Z_P1 = {z1}", s)
    s = re.sub(r"ENTRY_Z_P2 = [\d.]+", f"ENTRY_Z_P2 = {z2}", s)
    s = re.sub(r"HALF_QUOTE = \d+", f"HALF_QUOTE = {half}", s)
    s = re.sub(r"EDGE_TAKE = \d+", f"EDGE_TAKE = {edge}", s)
    s = re.sub(r"QUOTE_SIZE = \d+", f"QUOTE_SIZE = {qsize}", s)
    return s


def run_once(z1, z2, half=10, edge=2, qsize=5):
    SRC.write_text(patch(z1, z2, half, edge, qsize))
    daily = []
    for d in [2, 3, 4]:
        r = subprocess.run([BTX, str(SRC.resolve()), f"5-{d}", "--no-out", "--data", DATA, "--no-progress"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("Total profit:"):
                daily.append(int(line.split(":")[1].replace(",", "").strip()))
                break
    return daily


# Fine sweep around (1.8, 2.2) and (2.0, 2.5)
configs = [
    (1.6, 2.2), (1.7, 2.2), (1.8, 2.2), (1.9, 2.2), (2.0, 2.2),
    (1.8, 2.0), (1.8, 2.4), (1.8, 2.6), (1.8, 2.8),
    (2.0, 2.4), (2.0, 2.5), (2.0, 2.6), (2.0, 2.8), (2.0, 3.0),
    (1.7, 2.5), (1.9, 2.5),
]

print(f"{'z1':>5} {'z2':>5} {'d2':>8} {'d3':>8} {'d4':>8} {'total':>10} {'min':>8}")
results = []
for z1, z2 in configs:
    daily = run_once(z1, z2)
    tot = sum(daily)
    print(f"{z1:>5} {z2:>5} {daily[0]:>8} {daily[1]:>8} {daily[2]:>8} {tot:>10} {min(daily):>8}")
    results.append((z1, z2, tot, min(daily), daily))

print(f"\nBest by total: {max(results, key=lambda x: x[2])}")
print(f"Best by min-day: {max(results, key=lambda x: x[3])}")
SRC.write_text(TPL)
