"""
EDA Phase 2 — UV_VISOR — realistic execution backtests.

Findings from Phase 1:
- ALL UV_VISORs have spread >= 4 ticks 100% of the time, >= 5 ticks 99% — MM goldmine.
- MAGENTA|ORANGE (spread, z=1.2) shows 35,820 mid-to-mid pnl in toy backtest.
- ORANGE|YELLOW, MAGENTA|YELLOW also strong.
- No tight constant-sum invariant (best CV is 1.45%, vs PEBBLES 0.001%).
- Returns essentially uncorrelated → independent moves → MM works per-symbol.
- Lead-lag <2% — no exploitable predictive lag.

Phase 2:
1. Per-day breakdown of best pairs (avoid regime overfit).
2. Realistic execution: enter at far ask / cover at near bid (taker), MM at bb+1/ba-1.
3. MM-only backtest: how much can we earn just quoting both sides aggressively?
4. Combined MM + pair overlay simulation.
"""
import os, sys, time, warnings, itertools
warnings.filterwarnings("ignore")

class _Unbuf:
    def __init__(self, s): self.s = s
    def write(self, x): self.s.write(x); self.s.flush()
    def flush(self): self.s.flush()
sys.stdout = _Unbuf(sys.stdout)

T0 = time.time()
def tlog(msg): print(f"[t+{time.time()-T0:6.1f}s] {msg}")

import numpy as np
import pandas as pd

ROOT = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5"
DATA_DIR = f"{ROOT}/Data_ROUND_5"
OUT_DIR = f"{ROOT}/eda5/uv_visor"

SYMBOLS = ["UV_VISOR_AMBER","UV_VISOR_MAGENTA","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_YELLOW"]
SHORT = {s: s.replace("UV_VISOR_","") for s in SYMBOLS}

LIMIT = 10
SIZE = 10
WARMUP = 500
EXIT_Z = 0.3

# ---------- Load per-day order books ----------
tlog("loading per-day...")
days = {}
for d in (2,3,4):
    df = pd.read_csv(f"{DATA_DIR}/prices_round_5_day_{d}.csv", sep=";")
    df = df[df["product"].isin(SYMBOLS)].copy()
    bb = df.pivot_table(index="timestamp", columns="product", values="bid_price_1").sort_index()[SYMBOLS]
    ba = df.pivot_table(index="timestamp", columns="product", values="ask_price_1").sort_index()[SYMBOLS]
    bv = df.pivot_table(index="timestamp", columns="product", values="bid_volume_1").sort_index()[SYMBOLS]
    av = df.pivot_table(index="timestamp", columns="product", values="ask_volume_1").sort_index()[SYMBOLS]
    mid = (bb + ba) / 2.0
    days[d] = dict(bb=bb, ba=ba, bv=bv, av=av, mid=mid)

# ---------- Realistic pair backtest ----------
def pair_realistic(day_data, a, b, sign, entry_z, exit_z=EXIT_Z, size=SIZE, mode="taker"):
    """
    mode='taker': enter aggressively at far ask / cover at near bid.
    mode='passive': enter at bb+1/ba-1 with prob = vol_at_best/quote_size_5.
    Toy approximations to rank pairs.
    """
    bb_a, ba_a = day_data["bb"][a], day_data["ba"][a]
    bb_b, ba_b = day_data["bb"][b], day_data["ba"][b]
    mid_a = (bb_a + ba_a)/2.0; mid_b = (bb_b + ba_b)/2.0
    sig = (mid_a - mid_b) if sign == "spread" else (mid_a + mid_b)
    sig = sig.dropna()
    n = 0; mu = 0.0; M2 = 0.0
    pos = 0
    pnl = 0.0
    cycles = 0
    entry_a = entry_b = 0.0
    for t, s in sig.items():
        n += 1
        delta = s - mu; mu += delta/n; M2 += delta*(s-mu)
        if n < WARMUP: continue
        sd = (M2/(n-1))**0.5 if n>1 else 0.0
        if sd <= 1e-9: continue
        z = (s - mu)/sd
        if pos == 0:
            if z > entry_z:   # short signal
                # short A, long/short B per sign
                if mode == "taker":
                    pa = bb_a.loc[t]  # sell A at bid
                    if sign == "spread":
                        pb = ba_b.loc[t]  # buy B at ask
                    else:
                        pb = bb_b.loc[t]  # short B at bid
                else:
                    pa = ba_a.loc[t] - 1
                    pb = bb_b.loc[t] + 1 if sign == "spread" else ba_b.loc[t] - 1
                entry_a, entry_b = pa, pb
                pos = -1
            elif z < -entry_z:
                if mode == "taker":
                    pa = ba_a.loc[t]
                    pb = bb_b.loc[t] if sign == "spread" else ba_b.loc[t]
                else:
                    pa = bb_a.loc[t] + 1
                    pb = ba_b.loc[t] - 1 if sign == "spread" else bb_b.loc[t] + 1
                entry_a, entry_b = pa, pb
                pos = +1
        else:
            if abs(z) < exit_z:
                # close
                if pos == +1:  # we are long A, B-direction depends on sign
                    if mode == "taker":
                        pa_x = bb_a.loc[t]; pb_x = ba_b.loc[t] if sign=="spread" else bb_b.loc[t]
                    else:
                        pa_x = ba_a.loc[t] - 1
                        pb_x = bb_b.loc[t] + 1 if sign=="spread" else ba_b.loc[t] - 1
                    leg_a_pnl = (pa_x - entry_a) * size
                    leg_b_pnl = -(pb_x - entry_b) * size if sign=="spread" else (pb_x - entry_b) * size
                else:  # short A, B-direction depends on sign
                    if mode == "taker":
                        pa_x = ba_a.loc[t]; pb_x = bb_b.loc[t] if sign=="spread" else ba_b.loc[t]
                    else:
                        pa_x = bb_a.loc[t] + 1
                        pb_x = ba_b.loc[t] - 1 if sign=="spread" else bb_b.loc[t] + 1
                    leg_a_pnl = -(pa_x - entry_a) * size
                    leg_b_pnl = (pb_x - entry_b) * size if sign=="spread" else -(pb_x - entry_b) * size
                pnl += leg_a_pnl + leg_b_pnl
                cycles += 1
                pos = 0
    return pnl, cycles

# ---------- MM-only backtest ----------
def mm_only(day_data, sym, quote_size=5):
    """
    Aggressive MM at bb+1 / ba-1 every tick.
    Approximation: assume fill probability = min(1, vol_at_best / quote_size) / 2 each side
                   (split ~50/50 for queue + adverse selection).
    PnL per filled round-trip ≈ (ask_quote - bid_quote) per unit.
    """
    bb, ba, bv, av = day_data["bb"][sym], day_data["ba"][sym], day_data["bv"][sym], day_data["av"][sym]
    spread = ba - bb
    qbid = bb + 1
    qask = ba - 1
    quote_spread = qask - qbid  # = spread - 2
    # naive fill prob proxy
    fp_buy = (bv / quote_size).clip(upper=1.0) * 0.4
    fp_sell = (av / quote_size).clip(upper=1.0) * 0.4
    fills = pd.concat([fp_buy, fp_sell], axis=1).min(axis=1) * quote_size
    pnl = (quote_spread.clip(lower=0) * fills).sum()
    return float(pnl)

# ---------- Per-day pair backtest grid ----------
tlog("running pair backtests per day (taker)...")
rows = []
for a, b in itertools.combinations(SYMBOLS, 2):
    for sign in ("spread", "sum"):
        for ez in (1.0, 1.2, 1.5, 1.8, 2.0):
            row = {"a": SHORT[a], "b": SHORT[b], "sign": sign, "entry_z": ez}
            tot = 0
            for d in (2,3,4):
                pnl, cyc = pair_realistic(days[d], a, b, sign, ez, mode="taker")
                row[f"d{d}_pnl"] = round(pnl, 0)
                row[f"d{d}_cyc"] = cyc
                tot += pnl
            row["tot_pnl"] = round(tot, 0)
            rows.append(row)
df = pd.DataFrame(rows).sort_values("tot_pnl", ascending=False)
df.to_csv(f"{OUT_DIR}/p10_pair_taker_per_day.csv", index=False)
print("\n=== TAKER pair backtest, top 25 by total PnL across 3 days ===")
print(df.head(25).to_string(index=False))

# ---------- MM-only per symbol ----------
tlog("MM-only backtest per symbol per day...")
mm_rows = []
for s in SYMBOLS:
    row = {"sym": SHORT[s]}
    tot = 0
    for d in (2,3,4):
        v = mm_only(days[d], s)
        row[f"d{d}"] = round(v, 0)
        tot += v
    row["tot"] = round(tot, 0)
    mm_rows.append(row)
mm_df = pd.DataFrame(mm_rows)
mm_df.to_csv(f"{OUT_DIR}/p11_mm_per_day.csv", index=False)
print("\n=== MM-only PnL per symbol per day (aggressive bb+1/ba-1, qsize=5) ===")
print(mm_df.to_string(index=False))
print(f"Total MM-only across 5 symbols 3 days: {mm_df['tot'].sum():,.0f}")

# ---------- Best 3 pairs combined (additive, ignoring leg overlap) ----------
tlog("combined top pairs simulation...")
top = df.head(10).copy()
print("\nTop 10 pair config across 3 days:")
print(top[["a","b","sign","entry_z","tot_pnl","d2_pnl","d3_pnl","d4_pnl"]].to_string(index=False))

tlog("DONE")
