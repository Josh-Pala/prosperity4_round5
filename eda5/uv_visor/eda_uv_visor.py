"""
EDA — UV_VISOR family — Round 5
Goal: identify the dominant structure (cointegration, constant-sum invariant,
mean-reverting pairs, MM edge) and build a strategy aiming +200k on 3 days.

Outputs:
  - p1_descriptives.csv         per-symbol stats
  - p1_prices_norm.png          normalised price overlay
  - p2_corr_levels.csv          pearson corr on levels
  - p2_corr_returns.csv         pearson corr on log returns
  - p3_pair_zscan.csv           pair signals (spread/sum, half-life, z range)
  - p4_linear_combos.csv        OLS on cross-symbol linear combinations
  - p5_leadlag.csv              cross-correlation lead/lag
  - p6_invariant_check.csv      check for sum=K invariants per subset
  - p7_mm_microstructure.csv    spread/depth stats per symbol
  - p8_strategy_pnl.csv         expected PnL per candidate rule
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

class _Unbuf:
    def __init__(self, s): self.s = s
    def write(self, x): self.s.write(x); self.s.flush()
    def flush(self): self.s.flush()
sys.stdout = _Unbuf(sys.stdout)

T0 = time.time()
def tlog(msg): print(f"[t+{time.time()-T0:6.1f}s] {msg}")

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

ROOT = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5"
DATA_DIR = f"{ROOT}/Data_ROUND_5"
OUT_DIR = f"{ROOT}/eda5/uv_visor"
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "UV_VISOR_AMBER",
    "UV_VISOR_MAGENTA",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
]
SHORT = {s: s.replace("UV_VISOR_", "") for s in SYMBOLS}

# ---------- Load ----------
tlog("loading data...")
frames = []
for d in (2, 3, 4):
    df = pd.read_csv(f"{DATA_DIR}/prices_round_5_day_{d}.csv", sep=";")
    df = df[df["product"].isin(SYMBOLS)].copy()
    frames.append(df)
prices_long = pd.concat(frames, ignore_index=True)
prices_long["t"] = prices_long["day"].astype(int) * 1_000_000 + prices_long["timestamp"].astype(int)

tframes = []
for d in (2, 3, 4):
    df = pd.read_csv(f"{DATA_DIR}/trades_round_5_day_{d}.csv", sep=";")
    df = df[df["symbol"].isin(SYMBOLS)].copy()
    df["day"] = d
    df["t"] = df["day"].astype(int) * 1_000_000 + df["timestamp"].astype(int)
    tframes.append(df)
trades = pd.concat(tframes, ignore_index=True)

mid = prices_long.pivot_table(index="t", columns="product", values="mid_price").sort_index()[SYMBOLS]
ret = np.log(mid).diff().dropna()
vol = trades.pivot_table(index="t", columns="symbol", values="quantity", aggfunc="sum").reindex(mid.index).fillna(0)
vol = vol.reindex(columns=SYMBOLS, fill_value=0)
tlog(f"mid {mid.shape}, ret {ret.shape}, vol {vol.shape}, trades {trades.shape}")

# Build best bid / best ask series for microstructure
bbid = prices_long.pivot_table(index="t", columns="product", values="bid_price_1").sort_index()[SYMBOLS]
bask = prices_long.pivot_table(index="t", columns="product", values="ask_price_1").sort_index()[SYMBOLS]
spread = (bask - bbid)

# ---------- p1 descriptives ----------
tlog("p1 descriptives...")
desc = pd.DataFrame({
    "n": mid.count(),
    "mean_mid": mid.mean(),
    "std_mid": mid.std(),
    "min": mid.min(),
    "max": mid.max(),
    "range": mid.max() - mid.min(),
    "ret_std": ret.std(),
    "ret_skew": ret.skew(),
    "ret_kurt": ret.kurt(),
    "spread_mean": spread.mean(),
    "spread_p50": spread.quantile(0.5),
    "spread_p90": spread.quantile(0.9),
    "trade_vol_total": vol.sum(),
})
desc.to_csv(f"{OUT_DIR}/p1_descriptives.csv")
print(desc.round(3))

# Normalised plot
norm = (mid - mid.mean()) / mid.std()
plt.figure(figsize=(13, 5))
for s in SYMBOLS:
    plt.plot(norm.index, norm[s], lw=0.6, label=SHORT[s], alpha=0.85)
plt.title("UV_VISOR — normalised mids (z-score)")
plt.legend(loc="best", ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/p1_prices_norm.png", dpi=110)
plt.close()

# ---------- p2 correlations ----------
tlog("p2 correlations...")
cl = mid.corr(method="pearson")
cr = ret.corr(method="pearson")
cl.to_csv(f"{OUT_DIR}/p2_corr_levels.csv")
cr.to_csv(f"{OUT_DIR}/p2_corr_returns.csv")
print("\nCorr levels:\n", cl.round(3))
print("\nCorr returns:\n", cr.round(3))

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, mat, t in zip(axs, [cl, cr], ["levels", "returns"]):
    im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([SHORT[s] for s in SYMBOLS], rotation=45)
    ax.set_yticklabels([SHORT[s] for s in SYMBOLS])
    ax.set_title(f"corr {t}")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{mat.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(mat.values[i,j]) > 0.5 else "black")
fig.colorbar(im, ax=axs, fraction=0.04, pad=0.04)
plt.savefig(f"{OUT_DIR}/p2_corr.png", dpi=110)
plt.close()

# ---------- p3 pair z-scan ----------
tlog("p3 pair z-scan (spread + sum)...")
def half_life(s):
    s = pd.Series(s).dropna()
    if len(s) < 100 or s.std() < 1e-12:
        return np.nan
    s_lag = s.shift(1).dropna()
    s_d = s.diff().dropna()
    s_lag = s_lag.loc[s_d.index]
    X = sm.add_constant(s_lag.values)
    try:
        beta = OLS(s_d.values, X).fit().params[1]
        if beta >= 0:
            return np.nan
        return -np.log(2) / beta
    except Exception:
        return np.nan

rows = []
for a, b in itertools.combinations(SYMBOLS, 2):
    ma = mid[a]; mb = mid[b]
    for sign in ("spread", "sum"):
        sig = ma - mb if sign == "spread" else ma + mb
        sig = sig.dropna()
        if len(sig) < 200:
            continue
        z = (sig - sig.mean()) / sig.std()
        try:
            adf_p = adfuller(sig.values, regression="c")[1]
        except Exception:
            adf_p = np.nan
        try:
            coint_p = coint(ma, mb)[1] if sign == "spread" else np.nan
        except Exception:
            coint_p = np.nan
        rows.append({
            "a": SHORT[a], "b": SHORT[b], "sign": sign,
            "mean": float(sig.mean()),
            "std": float(sig.std()),
            "z_min": float(z.min()), "z_max": float(z.max()),
            "z_p1": float(z.quantile(0.01)), "z_p99": float(z.quantile(0.99)),
            "adf_p": adf_p,
            "coint_p": coint_p,
            "half_life": half_life(sig),
        })
ps = pd.DataFrame(rows).sort_values(["adf_p", "std"]).reset_index(drop=True)
ps.to_csv(f"{OUT_DIR}/p3_pair_zscan.csv", index=False)
print("\nTop pairs by ADF (most mean-reverting):")
print(ps.head(10).round(3))

# ---------- p4 linear combinations (5-symbol invariant?) ----------
tlog("p4 linear combos / invariants...")
# Full sum
sum_all = mid.sum(axis=1)
print("\nFull sum stats:", sum_all.describe().round(2).to_dict())
# OLS each leg vs others (look for tight residual)
combo_rows = []
for tgt in SYMBOLS:
    others = [s for s in SYMBOLS if s != tgt]
    X = sm.add_constant(mid[others].values)
    y = mid[tgt].values
    res = OLS(y, X).fit()
    pred = res.predict(X)
    resid = y - pred
    combo_rows.append({
        "target": SHORT[tgt],
        "intercept": float(res.params[0]),
        **{f"b_{SHORT[o]}": float(p) for o, p in zip(others, res.params[1:])},
        "r2": float(res.rsquared),
        "resid_std": float(np.std(resid)),
        "resid_p99": float(np.quantile(np.abs(resid), 0.99)),
    })
combo_df = pd.DataFrame(combo_rows)
combo_df.to_csv(f"{OUT_DIR}/p4_linear_combos.csv", index=False)
print(combo_df.round(3))

# Try simple sum/diff of subsets of 2/3/4
sub_rows = []
for k in (2, 3, 4):
    for combo in itertools.combinations(SYMBOLS, k):
        s = sum(mid[c] for c in combo)
        sub_rows.append({
            "k": k, "set": "+".join(SHORT[c] for c in combo),
            "mean": float(s.mean()), "std": float(s.std()),
            "cv": float(s.std() / s.mean()) if s.mean() != 0 else np.nan,
        })
sub_df = pd.DataFrame(sub_rows).sort_values("cv")
sub_df.to_csv(f"{OUT_DIR}/p4_subset_sums.csv", index=False)
print("\nTightest subset sums (lowest CV):")
print(sub_df.head(10).round(4))

# ---------- p5 lead-lag ----------
tlog("p5 lead-lag...")
def crosscorr(x, y, lag):
    if lag == 0:
        return x.corr(y)
    if lag > 0:
        return x.iloc[:-lag].corr(y.iloc[lag:].reset_index(drop=True))
    return x.iloc[-lag:].reset_index(drop=True).corr(y.iloc[:lag])

ll_rows = []
for a, b in itertools.combinations(SYMBOLS, 2):
    xa = ret[a].reset_index(drop=True)
    xb = ret[b].reset_index(drop=True)
    best_lag, best_c = 0, 0.0
    for L in range(-20, 21):
        c = crosscorr(xa, xb, L)
        if c is not None and not np.isnan(c) and abs(c) > abs(best_c):
            best_lag, best_c = L, c
    ll_rows.append({"a": SHORT[a], "b": SHORT[b],
                    "best_lag": best_lag, "max_abs_corr": float(best_c)})
ll_df = pd.DataFrame(ll_rows).sort_values("max_abs_corr", ascending=False)
ll_df.to_csv(f"{OUT_DIR}/p5_leadlag.csv", index=False)
print("\nLead-lag (returns), top 10:")
print(ll_df.head(10).round(3))

# ---------- p7 microstructure (spread, depth, MM edge) ----------
tlog("p7 microstructure...")
ms_rows = []
for s in SYMBOLS:
    sp = spread[s].dropna()
    bb = bbid[s].dropna()
    aa = bask[s].dropna()
    ms_rows.append({
        "sym": SHORT[s],
        "spread_mean": float(sp.mean()),
        "spread_p50": float(sp.quantile(0.5)),
        "spread_p90": float(sp.quantile(0.9)),
        "spread_p99": float(sp.quantile(0.99)),
        "frac_spread_eq_1": float((sp == 1).mean()),
        "frac_spread_ge_2": float((sp >= 2).mean()),
        "frac_spread_ge_3": float((sp >= 3).mean()),
        "frac_spread_ge_4": float((sp >= 4).mean()),
        "frac_spread_ge_5": float((sp >= 5).mean()),
    })
ms_df = pd.DataFrame(ms_rows)
ms_df.to_csv(f"{OUT_DIR}/p7_microstructure.csv", index=False)
print(ms_df.round(3))

# ---------- p8 strategy candidates simulated ----------
tlog("p8 strategy backtests (toy, mid-to-mid)...")
# 8a: pair mean-reversion at z=±entry, exit at z=±exit, hold mid-to-mid
def pair_bt(sig, entry_z, exit_z, hold_max=2000):
    z = (sig - sig.expanding(min_periods=500).mean()) / sig.expanding(min_periods=500).std()
    z = z.dropna()
    pos = 0
    pnl = 0.0
    entry_idx = None
    entry_val = 0.0
    cycles = 0
    for i, (t, zi) in enumerate(z.items()):
        s = sig.loc[t]
        if pos == 0:
            if zi > entry_z:
                pos = -1; entry_idx = i; entry_val = s
            elif zi < -entry_z:
                pos = +1; entry_idx = i; entry_val = s
        else:
            held = i - entry_idx
            if abs(zi) < exit_z or held >= hold_max:
                pnl += pos * (s - entry_val) * 10  # SIZE=10 leg
                cycles += 1
                pos = 0
    return pnl, cycles

bt_rows = []
for a, b in itertools.combinations(SYMBOLS, 2):
    for sign in ("spread", "sum"):
        sig = mid[a] - mid[b] if sign == "spread" else mid[a] + mid[b]
        for ez in (1.0, 1.2, 1.5, 1.8, 2.0):
            pnl, cyc = pair_bt(sig.dropna(), ez, 0.3)
            bt_rows.append({"a": SHORT[a], "b": SHORT[b], "sign": sign,
                            "entry_z": ez, "pnl_mid": pnl, "cycles": cyc})
bt_df = pd.DataFrame(bt_rows).sort_values("pnl_mid", ascending=False)
bt_df.to_csv(f"{OUT_DIR}/p8_pair_bt.csv", index=False)
print("\nTop 15 pair backtests (mid-to-mid, SIZE=10):")
print(bt_df.head(15).round(0).to_string(index=False))

# ---------- p9 MM edge proxy: how much can we earn at quote_at_best±1 ----------
tlog("p9 MM edge proxy...")
# A trivial approximation: every tick where spread>=2, capture (spread-1)/2 per side per quote_size,
# discounted by fill probability ~ 0.5*(volume/quote_size_cap).
# Rough but useful for ranking symbols.
mm_rows = []
QSIZE = 5
for s in SYMBOLS:
    sp = spread[s].dropna()
    v = vol[s].reindex(sp.index).fillna(0)
    edge_per_fill = ((sp - 1).clip(lower=0) / 2.0)
    fill_prob = (v / QSIZE).clip(upper=1.0)
    expected = (edge_per_fill * fill_prob * QSIZE).sum() * 2  # both sides
    mm_rows.append({"sym": SHORT[s], "mm_proxy_pnl": float(expected)})
mm_df = pd.DataFrame(mm_rows)
mm_df.to_csv(f"{OUT_DIR}/p9_mm_proxy.csv", index=False)
print(mm_df.round(0))

tlog("DONE")
