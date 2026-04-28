"""
EDA — Vertical Sleeping Pods (SLEEP_POD family) — Round 5
Optimised version: subsamples large series for slow tests; full series for stats/backtests.
"""
import os, sys, time
import warnings
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests, coint
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from arch import arch_model

ROOT = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5"
DATA_DIR = f"{ROOT}/Data_ROUND_5"
OUT_DIR = f"{ROOT}/eda5/sleep_pod"
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "SLEEP_POD_SUEDE",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_COTTON",
]
SHORT = {s: s.replace("SLEEP_POD_", "") for s in SYMBOLS}

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

# Subsampled series for SLOW tests only (KPSS, Granger, Hurst, GARCH)
SUB_STEP = 5
mid_sub = mid.iloc[::SUB_STEP]
ret_sub = ret.iloc[::SUB_STEP]
tlog(f"subsampled (step={SUB_STEP}) mid_sub {mid_sub.shape}")

# ---------- Helpers ----------
def safe_adf(x):
    x = pd.Series(x).dropna()
    if x.std() < 1e-12 or len(x) < 30:
        return np.nan
    try:
        return adfuller(x, regression="c", autolag="AIC")[1]
    except Exception:
        return np.nan

def safe_kpss(x):
    x = pd.Series(x).dropna()
    if x.std() < 1e-12 or len(x) < 30:
        return np.nan
    try:
        return kpss(x, regression="c", nlags=20)[1]   # fixed lag — much faster
    except Exception:
        return np.nan

def hurst_fast(ts):
    ts = np.asarray(pd.Series(ts).dropna())
    n = len(ts)
    if n < 200:
        return np.nan
    # use 8 lags only
    lags = np.unique(np.logspace(0.7, np.log10(min(200, n//4)), 8).astype(int))
    tau = []
    used = []
    for lag in lags:
        if lag < 2: continue
        diff = ts[lag:] - ts[:-lag]
        if diff.std() == 0: continue
        tau.append(np.sqrt(diff.var())); used.append(lag)
    if len(tau) < 4:
        return np.nan
    return np.polyfit(np.log(used), np.log(tau), 1)[0]

def half_life(spread):
    s = pd.Series(spread).dropna()
    delta = (s - s.shift(1)).dropna()
    s_lag = s.shift(1).loc[delta.index]
    if len(delta) < 30:
        return np.nan
    X = sm.add_constant(s_lag)
    try:
        b = OLS(delta, X).fit().params.iloc[1]
        return -np.log(2) / b if b < 0 else np.nan
    except Exception:
        return np.nan

# ---------- PART 1.1 Descriptives ----------
print("\n" + "=" * 70); print("PART 1 — RAW DATA HEALTH CHECK"); print("=" * 70)
print("\n[1.1] Shape / missing")
print(f"  prices_long  : {prices_long.shape}")
print(f"  mid wide     : {mid.shape}")
print(f"  trades       : {trades.shape}")
print(f"  duplicate t  : {mid.index.duplicated().sum()}")
print(f"  missing in mid:")
print(mid.isna().sum().to_string())

desc = pd.DataFrame({
    "mean_p": mid.mean(), "std_p": mid.std(), "min_p": mid.min(), "max_p": mid.max(),
    "skew_p": mid.skew(), "kurt_p": mid.kurt(),
    "mean_r": ret.mean(), "std_r": ret.std(),
    "skew_r": ret.skew(), "kurt_r": ret.kurt(),
})
print("\n[1.1] Per-symbol descriptives"); print(desc.round(6).to_string())
tlog("done 1.1")

# ---------- PART 1.2 Visual inspection ----------
fig, ax = plt.subplots(figsize=(11, 5))
norm = mid / mid.iloc[0] * 100
for s in SYMBOLS: ax.plot(norm.index, norm[s], label=SHORT[s], lw=0.7)
ax.set_title("SLEEP_POD prices normalised to 100"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_prices_norm.png", dpi=110); plt.close()

fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
for i, s in enumerate(SYMBOLS):
    axes[i].plot(ret.index, ret[s], lw=0.4); axes[i].set_ylabel(SHORT[s], fontsize=8)
    axes[i].axhline(0, color="k", lw=0.3)
axes[0].set_title("Log returns per symbol")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/02_logret_stack.png", dpi=110); plt.close()

fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
for i, s in enumerate(SYMBOLS):
    axes[i].plot(vol.index, vol[s], lw=0.5); axes[i].set_ylabel(SHORT[s], fontsize=8)
axes[0].set_title("Volume per tick")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/03_volume.png", dpi=110); plt.close()
tlog("done 1.2 plots")

# ---------- PART 1.3 Stationarity ----------
rows = []
for s in SYMBOLS:
    rows.append({
        "symbol": SHORT[s],
        "ADF_price_p": safe_adf(mid_sub[s]),
        "KPSS_price_p": safe_kpss(mid_sub[s]),
        "ADF_ret_p":   safe_adf(ret_sub[s]),
        "KPSS_ret_p":   safe_kpss(ret_sub[s]),
    })
print("\n[1.3] Stationarity — ADF & KPSS (subsampled, step=5)")
print(pd.DataFrame(rows).set_index("symbol").round(4).to_string())
tlog("done 1.3")

# ---------- PART 2.1 Autocorrelation ----------
print("\n" + "=" * 70); print("PART 2 — SINGLE-INSTRUMENT STRUCTURE"); print("=" * 70)
print("\n[2.1] Ljung-Box on log-returns (lags 5,10,20)")
lb_rows = []
for s in SYMBOLS:
    r = ret[s].dropna()
    lb = acorr_ljungbox(r, lags=[5, 10, 20], return_df=True)
    lb_rows.append({
        "symbol": SHORT[s],
        "LB5_p":  lb["lb_pvalue"].iloc[0],
        "LB10_p": lb["lb_pvalue"].iloc[1],
        "LB20_p": lb["lb_pvalue"].iloc[2],
        "ACF_lag1": acf(r, nlags=1, fft=True)[1],
    })
print(pd.DataFrame(lb_rows).set_index("symbol").round(4).to_string())

fig, axes = plt.subplots(5, 2, figsize=(11, 11))
for i, s in enumerate(SYMBOLS):
    r = ret[s].dropna()
    a = acf(r, nlags=40, fft=True)
    p = pacf(r, nlags=40, method="ywm")
    n = len(r); thr = 1.96/np.sqrt(n)
    axes[i, 0].bar(range(len(a)), a); axes[i, 0].set_ylabel(SHORT[s], fontsize=8)
    axes[i, 0].axhline(thr, color="r", lw=0.5, ls="--"); axes[i, 0].axhline(-thr, color="r", lw=0.5, ls="--")
    axes[i, 1].bar(range(len(p)), p)
    axes[i, 1].axhline(thr, color="r", lw=0.5, ls="--"); axes[i, 1].axhline(-thr, color="r", lw=0.5, ls="--")
axes[0, 0].set_title("ACF log-returns"); axes[0, 1].set_title("PACF log-returns")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_acf_pacf.png", dpi=110); plt.close()
tlog("done 2.1")

# ---------- PART 2.2 Hurst ----------
print("\n[2.2] Hurst exponent (variance scaling)")
hr = pd.DataFrame([{"symbol": SHORT[s], "Hurst": hurst_fast(mid[s].values)} for s in SYMBOLS]).set_index("symbol")
print(hr.round(4).to_string())

fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
for i, s in enumerate(SYMBOLS):
    axes[i].plot(mid.index, mid[s], lw=0.4, label="price")
    for w, c in [(20, "C1"), (50, "C2"), (100, "C3")]:
        axes[i].plot(mid.index, mid[s].rolling(w).mean(), lw=0.6, label=f"MA{w}", color=c)
    axes[i].set_ylabel(SHORT[s], fontsize=8); axes[i].legend(fontsize=6)
axes[0].set_title("Price + rolling means")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/05_rolling_means.png", dpi=110); plt.close()
tlog("done 2.2")

# ---------- PART 2.3 Volatility / GARCH ----------
print("\n[2.3] ARCH effects + GARCH(1,1) on subsampled returns")
v_rows = []
for s in SYMBOLS:
    r_full = ret[s].dropna()
    r = ret_sub[s].dropna()
    lb_sq = acorr_ljungbox(r_full**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    try:
        am = arch_model(r * 100, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)
        a = float(res.params.get("alpha[1]", np.nan))
        b = float(res.params.get("beta[1]", np.nan))
    except Exception:
        a, b = np.nan, np.nan
    v_rows.append({"symbol": SHORT[s], "LB_sq10_p": lb_sq, "alpha": a, "beta": b, "alpha+beta": (a+b) if pd.notna(a) and pd.notna(b) else np.nan})
print(pd.DataFrame(v_rows).set_index("symbol").round(4).to_string())

fig, ax = plt.subplots(figsize=(11, 5))
for s in SYMBOLS: ax.plot(ret.index, ret[s].rolling(20).std(), lw=0.6, label=SHORT[s])
ax.set_title("Rolling 20-tick std of log-returns"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/06_rolling_vol.png", dpi=110); plt.close()
tlog("done 2.3")

# ---------- PART 2.4 Distribution ----------
print("\n[2.4] Return distribution: skew, ex-kurt, JB p")
d_rows = []
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, s in enumerate(SYMBOLS):
    r = ret[s].dropna()
    jb = stats.jarque_bera(r)
    d_rows.append({"symbol": SHORT[s], "skew": float(stats.skew(r)),
                   "ex_kurt": float(stats.kurtosis(r)), "JB_p": float(jb.pvalue)})
    axes[0, i].hist(r, bins=80, density=True, alpha=0.7)
    xx = np.linspace(r.min(), r.max(), 200)
    axes[0, i].plot(xx, stats.norm.pdf(xx, r.mean(), r.std()), "r-", lw=0.8)
    axes[0, i].set_title(SHORT[s], fontsize=8)
    stats.probplot(r, dist="norm", plot=axes[1, i]); axes[1, i].set_title("")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/07_dist_qq.png", dpi=110); plt.close()
print(pd.DataFrame(d_rows).set_index("symbol").round(4).to_string())
tlog("done 2.4")

# ---------- PART 2.5 Volume ----------
print("\n[2.5] Volume diagnostics")
v2 = []
from sklearn.linear_model import LogisticRegression
for s in SYMBOLS:
    v = vol[s]; r = ret[s]
    common = v.index.intersection(r.index)
    v0 = v.loc[common]; r0 = r.loc[common]
    nz = (v0 > 0)
    if nz.sum() > 30:
        try:
            lb_v = acorr_ljungbox(v0, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
        except Exception:
            lb_v = np.nan
        corr_abs = float(np.corrcoef(np.abs(r0), v0)[0, 1])
        mask = (v0 > 0) & (r0.shift(-1).abs() > 0)
        X = np.log(v0[mask].values + 1).reshape(-1, 1)
        y = (r0.shift(-1)[mask].values > 0).astype(int)
        if len(np.unique(y)) > 1 and len(X) > 50:
            lr = LogisticRegression().fit(X, y)
            coef = float(lr.coef_[0, 0]); acc = float(lr.score(X, y))
        else:
            coef, acc = np.nan, np.nan
        amihud = float((np.abs(r0[nz]) / v0[nz]).mean())
        v2.append({"symbol": SHORT[s], "vol_LB10_p": lb_v, "corr|r|,v": corr_abs,
                   "logit_coef": coef, "logit_acc": acc, "Amihud": amihud, "n_trade_ticks": int(nz.sum())})
    else:
        v2.append({"symbol": SHORT[s], "vol_LB10_p": np.nan, "corr|r|,v": np.nan,
                   "logit_coef": np.nan, "logit_acc": np.nan, "Amihud": np.nan, "n_trade_ticks": int(nz.sum())})
print(pd.DataFrame(v2).set_index("symbol").round(6).to_string())
tlog("done 2.5")

# ---------- PART 3 — CROSS ----------
print("\n" + "=" * 70); print("PART 3 — CROSS-INSTRUMENT STRUCTURE"); print("=" * 70)

pearson = ret.corr(method="pearson")
spearman = ret.corr(method="spearman")
print("\n[3.1] Pearson"); print(pearson.round(3).to_string())
print("\n[3.1] Spearman"); print(spearman.round(3).to_string())

fig, ax = plt.subplots(figsize=(11, 5))
roll_corrs = {}
for i in range(len(SYMBOLS)):
    for j in range(i+1, len(SYMBOLS)):
        a, b = SYMBOLS[i], SYMBOLS[j]
        rc = ret[a].rolling(50).corr(ret[b])
        roll_corrs[(a, b)] = rc
        ax.plot(rc.index, rc, lw=0.3, label=f"{SHORT[a]}-{SHORT[b]}")
ax.set_title("Rolling 50-tick correlation"); ax.legend(fontsize=6, ncol=3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/08_rolling_corr.png", dpi=110); plt.close()

print("\n[3.1] Rolling-corr stability (mean & std)")
roll_summary = [{"pair": f"{SHORT[a]}-{SHORT[b]}", "mean_rc": rc.mean(), "std_rc": rc.std()}
                for (a, b), rc in roll_corrs.items()]
print(pd.DataFrame(roll_summary).set_index("pair").round(3).to_string())
tlog("done 3.1")

# 3.2 Engle-Granger (subsampled for speed)
print("\n[3.2] Engle-Granger pairwise cointegration (subsampled, step=5)")
coint_rows = []
for i in range(len(SYMBOLS)):
    for j in range(i+1, len(SYMBOLS)):
        a, b = SYMBOLS[i], SYMBOLS[j]
        x = np.log(mid_sub[a]).dropna(); y = np.log(mid_sub[b]).dropna()
        common = x.index.intersection(y.index)
        x = x.loc[common]; y = y.loc[common]
        try:
            _, pval, _ = coint(x, y)
        except Exception:
            pval = np.nan
        beta = OLS(x, sm.add_constant(y)).fit().params.iloc[1]
        spread = x - beta * y
        coint_rows.append({
            "pair": f"{SHORT[a]}-{SHORT[b]}",
            "EG_p": pval, "hedge_beta": beta,
            "spread_ADF_p": safe_adf(spread),
            "half_life": half_life(spread),
            "spread_std": spread.std(),
        })
coint_df = pd.DataFrame(coint_rows).set_index("pair").round(4)
print(coint_df.to_string())

print("\n[3.2] Johansen test (full basket, log prices, subsampled)")
try:
    log_p = np.log(mid_sub).dropna()
    j = coint_johansen(log_p, det_order=0, k_ar_diff=1)
    print("  trace stats : ", np.round(j.lr1, 3))
    print("  crit 95%    : ", np.round(j.cvt[:, 1], 3))
    rank = int((j.lr1 > j.cvt[:, 1]).sum())
    print(f"  cointegration rank @95%: {rank}")
except Exception as e:
    print(f"  Johansen failed: {e}")

best_pair = coint_df["EG_p"].idxmin()
print(f"\n[3.2] Best cointegrated pair: {best_pair}")
a_short, b_short = best_pair.split("-")
A = "SLEEP_POD_" + a_short; B = "SLEEP_POD_" + b_short
x = np.log(mid[A]); y = np.log(mid[B])
beta = OLS(x.dropna(), sm.add_constant(y.dropna())).fit().params.iloc[1]
spread = x - beta * y
mu, sd = spread.mean(), spread.std()
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(spread.index, spread, lw=0.4)
for k, c in [(1, "orange"), (2, "red")]:
    ax.axhline(mu + k*sd, color=c, lw=0.5, ls="--"); ax.axhline(mu - k*sd, color=c, lw=0.5, ls="--")
ax.axhline(mu, color="k", lw=0.5)
ax.set_title(f"Spread {best_pair}  β={beta:.3f}  μ={mu:.4f} σ={sd:.4f}")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/09_best_spread.png", dpi=110); plt.close()
tlog("done 3.2")

# 3.3 Granger (subsampled, max lag=5)
print("\n[3.3] Granger causality (best lag p-value, max lag=5, step=5 subsample)")
gc_rows = []
for i in range(len(SYMBOLS)):
    for j in range(len(SYMBOLS)):
        if i == j: continue
        a, b = SYMBOLS[i], SYMBOLS[j]
        sub = ret_sub[[b, a]].dropna()
        if len(sub) < 200: continue
        try:
            res = grangercausalitytests(sub, maxlag=5, verbose=False)
            best_p = min(res[lag][0]["ssr_ftest"][1] for lag in res)
        except Exception:
            best_p = np.nan
        gc_rows.append({"leader": SHORT[a], "follower": SHORT[b], "best_p": best_p})
gc_df = pd.DataFrame(gc_rows).sort_values("best_p")
print(gc_df.head(15).round(5).to_string(index=False))
top = gc_df.iloc[0]
print(f"\n[3.3] Strongest Granger pair: {top['leader']} → {top['follower']}, p={top['best_p']:.4g}")

A = "SLEEP_POD_" + top["leader"]; B = "SLEEP_POD_" + top["follower"]
xc = []
maxlag = 20
for L in range(-maxlag, maxlag+1):
    if L < 0:
        a_arr = ret[A].shift(-L); b_arr = ret[B]
    elif L > 0:
        a_arr = ret[A]; b_arr = ret[B].shift(L)
    else:
        a_arr = ret[A]; b_arr = ret[B]
    df = pd.concat([a_arr, b_arr], axis=1).dropna()
    xc.append(df.iloc[:, 0].corr(df.iloc[:, 1]))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(-maxlag, maxlag+1), xc)
ax.set_title(f"CCF {top['leader']} vs {top['follower']}")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/10_ccf_top_pair.png", dpi=110); plt.close()
tlog("done 3.3")

# 3.4 Order flow
print("\n[3.4] Kyle lambda (price change ~ signed volume)")
ki = []
for s in SYMBOLS:
    r = ret[s]; v = vol[s].reindex(r.index).fillna(0)
    sign = np.sign(r)
    sv = sign * v
    mask = sv != 0
    if mask.sum() > 100:
        slope, _, rval, _, _ = stats.linregress(sv[mask], r[mask])
        ki.append({"symbol": SHORT[s], "lambda": slope, "r2": rval**2, "n": int(mask.sum())})
    else:
        ki.append({"symbol": SHORT[s], "lambda": np.nan, "r2": np.nan, "n": int(mask.sum())})
print(pd.DataFrame(ki).set_index("symbol").to_string())

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=True)
for ax_, s in zip(axes, SYMBOLS):
    v = vol[s]; p = mid[s]
    nz = v[v > 0]
    if len(nz) < 30: ax_.set_title(f"{SHORT[s]} (no vol)"); continue
    thr = nz.quantile(0.95)
    spikes = v[v >= thr].index
    paths = []
    for t in spikes:
        try: iloc = p.index.get_loc(t)
        except KeyError: continue
        if iloc + 30 >= len(p): continue
        seg = np.log(p.iloc[iloc: iloc+31].values / p.iloc[iloc])
        paths.append(seg)
    if paths:
        avg = np.nanmean(np.array(paths), axis=0)
        ax_.plot(avg); ax_.axhline(0, color="k", lw=0.4)
    ax_.set_title(SHORT[s], fontsize=8)
plt.suptitle("Avg log-price path 30 ticks after volume spike (top 5%)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/11_impact_decay.png", dpi=110); plt.close()
tlog("done 3.4")

# ---------- PART 4 — STRATEGY BACKTESTS ----------
print("\n" + "=" * 70); print("PART 4 — STRATEGY SIGNAL BACKTESTS"); print("=" * 70)
POS_LIM = 10

def metrics(pnl, trade_pnls):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.std() == 0 or len(pnl) == 0: sharpe = 0.0
    else: sharpe = pnl.mean() / pnl.std() * np.sqrt(len(pnl))
    cum = np.cumsum(pnl)
    max_dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0
    if len(trade_pnls):
        wr = float((np.array(trade_pnls) > 0).mean()); mp = float(np.mean(trade_pnls))
    else: wr, mp = np.nan, np.nan
    return sharpe, wr, mp, max_dd, len(trade_pnls)

def run_signal(price_series, signal):
    """Position target = sign(signal)*POS_LIM. Returns metrics."""
    p = price_series.values.astype(float)
    sig = pd.Series(signal).reindex(price_series.index).fillna(0).values
    pos = 0; entry_p = None; entry_pos = 0
    pnl = []; trades = []
    for i in range(len(p) - 1):
        target = int(np.sign(sig[i])) * POS_LIM
        if target != pos:
            if pos != 0 and entry_p is not None:
                trades.append((p[i] - entry_p) * entry_pos)
            pos = target; entry_p = p[i]; entry_pos = pos
        pnl.append(pos * (p[i+1] - p[i]))
    if pos != 0 and entry_p is not None:
        trades.append((p[-1] - entry_p) * entry_pos)
    return metrics(pnl, trades)

# 4.1 Mean reversion
print("\n[4.1] Mean reversion (z-score window=50, in ±1.5, out |z|<0.3)")
mr_rows = []
for s in SYMBOLS:
    p = mid[s]
    m = p.rolling(50).mean(); sd = p.rolling(50).std()
    z = (p - m) / sd
    state = 0; sig = np.zeros(len(z))
    zv = z.values
    for i in range(len(zv)):
        if np.isnan(zv[i]):
            sig[i] = state; continue
        if state == 0:
            if zv[i] < -1.5: state = 1
            elif zv[i] > 1.5: state = -1
        else:
            if abs(zv[i]) < 0.3: state = 0
        sig[i] = state
    sh, wr, mp, dd, nt = run_signal(p, pd.Series(sig, index=p.index))
    mr_rows.append({"symbol": SHORT[s], "Sharpe": sh, "WinRate": wr, "MeanPnL": mp, "MaxDD": dd, "Trades": nt})
mr_df = pd.DataFrame(mr_rows).set_index("symbol")
print(mr_df.round(3).to_string())
tlog("done 4.1")

# 4.2 Momentum grid
print("\n[4.2] Momentum grid — best (N,M) per symbol")
mom_rows = []
for s in SYMBOLS:
    p = mid[s]
    best = None
    for N in (5, 10, 20):
        past_ret = np.log(p).diff(N).values
        for M in (5, 10, 20):
            sig_arr = np.zeros(len(p))
            i = 0
            while i < len(p) - 1:
                if not np.isnan(past_ret[i]) and past_ret[i] != 0:
                    direction = 1 if past_ret[i] > 0 else -1
                    end = min(i + M, len(p) - 1)
                    sig_arr[i:end] = direction
                    i = end
                else: i += 1
            sh, wr, mp, dd, nt = run_signal(p, pd.Series(sig_arr, index=p.index))
            if best is None or sh > best["Sharpe"]:
                best = {"N": N, "M": M, "Sharpe": sh, "WinRate": wr, "MeanPnL": mp, "MaxDD": dd, "Trades": nt}
    best["symbol"] = SHORT[s]; mom_rows.append(best)
mom_df = pd.DataFrame(mom_rows).set_index("symbol")
print(mom_df.round(3).to_string())
tlog("done 4.2")

# 4.3 Volume-triggered momentum
print("\n[4.3] Volume-triggered momentum (vol > 2× MA20, hold 5)")
vt_rows = []
for s in SYMBOLS:
    p = mid[s]; v = vol[s]; r = ret[s]
    vma = v.rolling(20).mean()
    sig_arr = np.zeros(len(p))
    pi = 0; vi = v.values; vmai = vma.values; ri = r.reindex(p.index).values
    while pi < len(p) - 1:
        if pd.notna(vmai[pi]) and vi[pi] > 2 * vmai[pi] and vi[pi] > 0 and pd.notna(ri[pi]) and ri[pi] != 0:
            direction = 1 if ri[pi] > 0 else -1
            end = min(pi + 5, len(p) - 1)
            sig_arr[pi:end] = direction
            pi = end
        else:
            pi += 1
    sh, wr, mp, dd, nt = run_signal(p, pd.Series(sig_arr, index=p.index))
    vt_rows.append({"symbol": SHORT[s], "Sharpe": sh, "WinRate": wr, "MeanPnL": mp, "MaxDD": dd, "Trades": nt})
vt_df = pd.DataFrame(vt_rows).set_index("symbol")
print(vt_df.round(3).to_string())
tlog("done 4.3")

# 4.4 Lead-lag
print("\n[4.4] Lead-lag cross signal — top Granger pair")
top_l = "SLEEP_POD_" + top["leader"]; top_f = "SLEEP_POD_" + top["follower"]
p = mid[top_f]
lr_lag = ret[top_l].shift(1).reindex(p.index).fillna(0)
sig = np.sign(lr_lag.values)
sh, wr, mp, dd, nt = run_signal(p, pd.Series(sig, index=p.index))
print(f"  {top['leader']} → {top['follower']}: Sharpe={sh:.3f} WR={wr if not np.isnan(wr) else 0:.3f} MeanPnL={mp:.3f} DD={dd:.1f} N={nt}")
ll_row = {"symbol": top['follower'], "leader": top['leader'], "Sharpe": sh, "WinRate": wr, "MeanPnL": mp, "MaxDD": dd, "Trades": nt}
tlog("done 4.4")

# 4.5 Pairs / stat-arb
print("\n[4.5] Pairs / stat-arb on best cointegrated pair")
A = "SLEEP_POD_" + best_pair.split("-")[0]; B = "SLEEP_POD_" + best_pair.split("-")[1]
x = np.log(mid[A]); y = np.log(mid[B])
common = x.dropna().index.intersection(y.dropna().index)
x = x.loc[common]; y = y.loc[common]
beta_h = OLS(x, sm.add_constant(y)).fit().params.iloc[1]
spread = x - beta_h * y
mu, sd = spread.mean(), spread.std()
z = (spread - mu) / sd

state = 0; pos_a = 0; pos_b = 0
pnl = []; trade_pnls = []; open_trade = None
mid_a = mid[A].loc[common].values; mid_b = mid[B].loc[common].values
zv = z.values
for i in range(len(z) - 1):
    if np.isnan(zv[i]):
        pnl.append(0); continue
    if state == 0:
        if zv[i] < -1.5:
            state = 1; pos_a = POS_LIM; pos_b = -int(round(beta_h * POS_LIM))
            open_trade = (mid_a[i], mid_b[i], pos_a, pos_b)
        elif zv[i] > 1.5:
            state = -1; pos_a = -POS_LIM; pos_b = int(round(beta_h * POS_LIM))
            open_trade = (mid_a[i], mid_b[i], pos_a, pos_b)
    else:
        if abs(zv[i]) < 0.3:
            if open_trade:
                pa, pb, pa0, pb0 = open_trade
                trade_pnls.append(pa0 * (mid_a[i] - pa) + pb0 * (mid_b[i] - pb))
            state = 0; pos_a = 0; pos_b = 0; open_trade = None
    pnl.append(pos_a * (mid_a[i+1] - mid_a[i]) + pos_b * (mid_b[i+1] - mid_b[i]))

sh, wr, mp, dd, nt = metrics(pnl, trade_pnls)
print(f"  Pair {best_pair}  β={beta_h:.3f}  Sharpe={sh:.3f}  WR={wr if not np.isnan(wr) else 0:.3f}  MeanPnL={mp:.3f}  DD={dd:.1f}  Trades={nt}")
pair_row = {"pair": best_pair, "Sharpe": sh, "WinRate": wr, "MeanPnL": mp, "MaxDD": dd, "Trades": nt}
tlog("done 4.5")

# ---------- VERDICT TABLE ----------
print("\n" + "=" * 70); print("PART 5 — STRATEGY VERDICT TABLE"); print("=" * 70)

def verdict(sharpe, ntrades):
    if ntrades < 10: return "NO EDGE"
    if sharpe >= 2.0: return "STRONG EDGE"
    if sharpe >= 1.0: return "WEAK EDGE"
    if sharpe >= 0.0: return "NO EDGE"
    return "AVOID"

rows = []
for s in SYMBOLS:
    sh = mr_df.loc[SHORT[s], "Sharpe"]; wr = mr_df.loc[SHORT[s], "WinRate"]; mp = mr_df.loc[SHORT[s], "MeanPnL"]
    dd = mr_df.loc[SHORT[s], "MaxDD"]; nt = mr_df.loc[SHORT[s], "Trades"]
    rows.append([SHORT[s], "MeanRev z-score (50)", sh, wr, mp, dd, verdict(sh, nt)])
for s in SYMBOLS:
    sh = mom_df.loc[SHORT[s], "Sharpe"]; wr = mom_df.loc[SHORT[s], "WinRate"]; mp = mom_df.loc[SHORT[s], "MeanPnL"]
    dd = mom_df.loc[SHORT[s], "MaxDD"]; nt = mom_df.loc[SHORT[s], "Trades"]
    rows.append([SHORT[s], f"Momentum N={int(mom_df.loc[SHORT[s],'N'])} M={int(mom_df.loc[SHORT[s],'M'])}", sh, wr, mp, dd, verdict(sh, nt)])
for s in SYMBOLS:
    sh = vt_df.loc[SHORT[s], "Sharpe"]; wr = vt_df.loc[SHORT[s], "WinRate"]; mp = vt_df.loc[SHORT[s], "MeanPnL"]
    dd = vt_df.loc[SHORT[s], "MaxDD"]; nt = vt_df.loc[SHORT[s], "Trades"]
    rows.append([SHORT[s], "Volume-trig momentum", sh, wr, mp, dd, verdict(sh, nt)])
rows.append([ll_row["symbol"], f"Lead-lag from {ll_row['leader']}", ll_row["Sharpe"], ll_row["WinRate"], ll_row["MeanPnL"], ll_row["MaxDD"], verdict(ll_row["Sharpe"], ll_row["Trades"])])
rows.append([pair_row["pair"], "Pairs spread z-score", pair_row["Sharpe"], pair_row["WinRate"], pair_row["MeanPnL"], pair_row["MaxDD"], verdict(pair_row["Sharpe"], pair_row["Trades"])])

verdict_df = pd.DataFrame(rows, columns=["Symbol", "Strategy", "Sharpe", "WinRate", "MeanPnL", "MaxDD", "Verdict"])
print(verdict_df.round(3).to_string(index=False))
verdict_df.to_csv(f"{OUT_DIR}/verdict.csv", index=False)
tlog("done verdict")

print("\nDONE — Plots written to:", OUT_DIR)
