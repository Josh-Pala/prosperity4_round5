"""SNACKPACK EDA — Part 3: Cross-instrument structure."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm

OUT_DIR = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/snackpack"
SYMBOLS = ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
           "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"]

mid = pd.read_parquet(f"{OUT_DIR}/_mid.parquet")
ret = pd.read_parquet(f"{OUT_DIR}/_ret.parquet")
vol = pd.read_parquet(f"{OUT_DIR}/_vol.parquet")
logp = np.log(mid)

# 3.1 correlation
print("=== 3.1 Correlation matrices ===")
pearson = ret.corr(method="pearson")
spearman = ret.corr(method="spearman")
print("Pearson:\n", pearson.round(3))
print("\nSpearman:\n", spearman.round(3))
pearson.to_csv(f"{OUT_DIR}/p3_pearson.csv")
spearman.to_csv(f"{OUT_DIR}/p3_spearman.csv")

# rolling correlation
roll_corr_stats = []
fig, axes = plt.subplots(5,2, figsize=(14,12))
axes = axes.flatten()
pairs = list(combinations(SYMBOLS, 2))
for i,(a,b) in enumerate(pairs):
    rc = ret[a].rolling(50).corr(ret[b])
    axes[i].plot(rc.index, rc, lw=0.4); axes[i].set_title(f"{a[10:]}-{b[10:]} rolling50 corr", fontsize=7)
    axes[i].axhline(rc.mean(), color="r", lw=0.5)
    roll_corr_stats.append({"pair":f"{a}|{b}", "mean":round(rc.mean(),4),
                            "std":round(rc.std(),4), "min":round(rc.min(),4), "max":round(rc.max(),4)})
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/p3_rolling_corr.png", dpi=110); plt.close()
rc_df = pd.DataFrame(roll_corr_stats); print("\nRolling correlation summary:\n", rc_df.to_string(index=False))
rc_df.to_csv(f"{OUT_DIR}/p3_rolling_corr.csv", index=False)

# 3.2 Cointegration — Engle-Granger pairwise
print("\n=== 3.2 Engle-Granger cointegration ===")
coint_rows = []
for a,b in pairs:
    pa, pb = logp[a].dropna(), logp[b].dropna()
    common = pa.index.intersection(pb.index)
    pa, pb = pa.loc[common], pb.loc[common]
    # subsample to speed up
    step = max(1, len(pa)//8000)
    pa_s, pb_s = pa.iloc[::step], pb.iloc[::step]
    score, pval, _ = coint(pa_s, pb_s)
    # OLS hedge
    X = sm.add_constant(pb_s); ols = sm.OLS(pa_s, X).fit()
    beta = ols.params.iloc[1]
    spread_full = pa - beta*pb
    # ADF on spread
    adf_p = adfuller(spread_full.iloc[::step], autolag="AIC")[1]
    # half-life via OLS on lagged spread change
    s_lag = spread_full.shift(1).dropna()
    s_diff = spread_full.diff().dropna()
    common2 = s_lag.index.intersection(s_diff.index)
    Xh = sm.add_constant(s_lag.loc[common2])
    rh = sm.OLS(s_diff.loc[common2], Xh).fit()
    rho = rh.params.iloc[1]
    half_life = -np.log(2)/np.log(1+rho) if (rho<0 and rho>-1) else np.nan
    coint_rows.append({"pair":f"{a}|{b}", "EG_p":round(pval,4),
                       "hedge":round(beta,4), "ADF_spread_p":round(adf_p,4),
                       "half_life":round(half_life,2) if not np.isnan(half_life) else None,
                       "spread_std":round(spread_full.std(),5)})
coint_df = pd.DataFrame(coint_rows).sort_values("EG_p")
print(coint_df.to_string(index=False))
coint_df.to_csv(f"{OUT_DIR}/p3_cointegration.csv", index=False)

# best pair plot
best = coint_df.iloc[0]
a,b = best["pair"].split("|")
beta = best["hedge"]
sp = logp[a] - beta*logp[b]
mu, sd = sp.mean(), sp.std()
plt.figure(figsize=(12,4))
plt.plot(sp.index, sp, lw=0.4, label="spread")
plt.axhline(mu, color="k", lw=0.4)
for k in (1,2):
    plt.axhline(mu+k*sd, color="r", lw=0.3, ls="--")
    plt.axhline(mu-k*sd, color="r", lw=0.3, ls="--")
plt.title(f"Best coint pair: {a} vs {b}, beta={beta:.3f}, EG p={best['EG_p']}")
plt.legend(); plt.savefig(f"{OUT_DIR}/p3_best_spread.png", dpi=110, bbox_inches="tight"); plt.close()

# Johansen on full basket
print("\n=== Johansen test (full basket) ===")
lp_aligned = logp.dropna()
step = max(1, len(lp_aligned)//5000)
lp_s = lp_aligned.iloc[::step]
joh = coint_johansen(lp_s.values, det_order=0, k_ar_diff=1)
trace = joh.lr1
cv95 = joh.cvt[:,1]
joh_df = pd.DataFrame({"r<=":[f"r<={i}" for i in range(len(trace))],
                       "trace":np.round(trace,3), "cv_95":np.round(cv95,3),
                       "reject":trace>cv95})
print(joh_df.to_string(index=False))
joh_df.to_csv(f"{OUT_DIR}/p3_johansen.csv", index=False)

# 3.3 Lead-lag — cross-correlation + Granger
print("\n=== 3.3 Lead-lag (cross-correlation) ===")
def ccf(a, b, lags):
    a = (a-a.mean())/a.std(); b = (b-b.mean())/b.std()
    n = len(a); out=[]
    for k in lags:
        if k>=0:
            out.append(np.mean(a[k:]*b[:n-k]) if n>k else np.nan)
        else:
            out.append(np.mean(a[:n+k]*b[-k:]) if n>-k else np.nan)
    return np.array(out)

lags = list(range(-20,21))
fig, axes = plt.subplots(5,2, figsize=(14,12)); axes=axes.flatten()
ccf_rows = []
for i,(a,b) in enumerate(pairs):
    ra = ret[a].dropna().values; rb = ret[b].dropna().values
    n = min(len(ra), len(rb))
    c = ccf(ra[:n], rb[:n], lags)
    axes[i].bar(lags, c); axes[i].set_title(f"{a[10:]} ↔ {b[10:]}", fontsize=7)
    axes[i].axhline(0, color="k", lw=0.3)
    # find max abs cross-corr at non-zero lag
    nonzero = [(l,v) for l,v in zip(lags,c) if l!=0]
    lmax, vmax = max(nonzero, key=lambda x: abs(x[1]))
    ccf_rows.append({"pair":f"{a}|{b}", "lag0":round(c[lags.index(0)],4),
                     "best_lag":lmax, "best_ccf":round(vmax,4)})
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/p3_ccf.png", dpi=110); plt.close()
ccf_df = pd.DataFrame(ccf_rows); print(ccf_df.to_string(index=False))
ccf_df.to_csv(f"{OUT_DIR}/p3_ccf.csv", index=False)

# Granger causality (subsample to keep it fast)
print("\n=== 3.3 Granger causality (max lag 5, subsample 5k obs) ===")
gr_rows = []
ret_s = ret.dropna().iloc[::max(1,len(ret)//5000)]
for a,b in pairs:
    for direction, (x,y) in [(f"{a}->{b}", (a,b)), (f"{b}->{a}", (b,a))]:
        try:
            data = ret_s[[y,x]].dropna()
            res = grangercausalitytests(data, maxlag=5, verbose=False)
            best_p = min(res[k][0]["ssr_ftest"][1] for k in res)
            best_lag = min(res, key=lambda k: res[k][0]["ssr_ftest"][1])
            gr_rows.append({"causality":direction, "min_p":round(best_p,4), "best_lag":best_lag})
        except Exception as e:
            gr_rows.append({"causality":direction, "min_p":np.nan, "best_lag":None})
gr_df = pd.DataFrame(gr_rows).sort_values("min_p")
print(gr_df.to_string(index=False))
gr_df.to_csv(f"{OUT_DIR}/p3_granger.csv", index=False)

# 3.4 Kyle's lambda — proxy: signed volume = (askv - bidv)*sign?  We approximate signed volume from order book imbalance
print("\n=== 3.4 Kyle's lambda (price impact via OBI) ===")
# We don't have signed trade flow, but order book imbalance per tick is a usable proxy.
# Signed volume proxy: bid_volume_1 - ask_volume_1 (sign of pressure), magnitude = total volume.
# Reload raw
DATA = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5"
frames = []
for d in (2,3,4):
    df = pd.read_csv(f"{DATA}/prices_round_5_day_{d}.csv", sep=";")
    df = df[df["product"].isin(SYMBOLS)].copy()
    df["t"] = d*1_000_000 + df["timestamp"]
    df["bidv"] = df[["bid_volume_1","bid_volume_2","bid_volume_3"]].fillna(0).sum(axis=1)
    df["askv"] = df[["ask_volume_1","ask_volume_2","ask_volume_3"]].fillna(0).sum(axis=1)
    df["signed_v"] = df["bidv"] - df["askv"]
    frames.append(df[["product","t","mid_price","signed_v"]])
raw = pd.concat(frames, ignore_index=True).sort_values(["product","t"])

kyle_rows = []
for s in SYMBOLS:
    sub = raw[raw["product"]==s].copy()
    sub["dp"] = sub["mid_price"].diff()
    sub = sub.dropna()
    X = sm.add_constant(sub["signed_v"])
    res = sm.OLS(sub["dp"], X).fit()
    lam = res.params.iloc[1]; r2 = res.rsquared
    # impact decay after large signed_v spike (top 5% absolute)
    thr = sub["signed_v"].abs().quantile(0.95)
    spikes = sub.index[sub["signed_v"].abs() >= thr]
    horizons = [1,5,10,20,30]
    decay = {}
    sub_reset = sub.reset_index(drop=True)
    pos_arr = np.where(sub_reset["signed_v"].abs() >= thr)[0]
    for h in horizons:
        deltas = []
        for p in pos_arr:
            if p+h < len(sub_reset):
                sgn = np.sign(sub_reset.loc[p,"signed_v"])
                deltas.append(sgn * (sub_reset.loc[p+h,"mid_price"] - sub_reset.loc[p,"mid_price"]))
        decay[f"d{h}"] = round(np.mean(deltas),3) if deltas else np.nan
    # asymmetry
    buy = sub[sub["signed_v"]>0]; sell = sub[sub["signed_v"]<0]
    Xb = sm.add_constant(buy["signed_v"]); Xs = sm.add_constant(sell["signed_v"])
    lam_b = sm.OLS(buy["dp"], Xb).fit().params.iloc[1] if len(buy)>10 else np.nan
    lam_s = sm.OLS(sell["dp"], Xs).fit().params.iloc[1] if len(sell)>10 else np.nan
    kyle_rows.append({"symbol":s, "lambda":round(lam,5), "R2":round(r2,4),
                      "lam_buy":round(lam_b,5), "lam_sell":round(lam_s,5), **decay})
k_df = pd.DataFrame(kyle_rows); print(k_df.to_string(index=False))
k_df.to_csv(f"{OUT_DIR}/p3_kyle.csv", index=False)

# save
print("\nPart 3 complete.")
