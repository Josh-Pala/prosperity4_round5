"""Out-of-sample sanity check for the winning strategies."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm

OUT = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/snackpack"
SYMBOLS = ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO",
           "SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"]
LIMIT = 10
mid = pd.read_parquet(f"{OUT}/_mid.parquet")

# split: index t encodes day*1e6+timestamp
day = pd.Series(mid.index // 1_000_000, index=mid.index)
mask_train = (day == 2)
mask_test  = (day != 2)
print(f"Train ticks per symbol: {int(mask_train.sum())}, Test ticks: {int(mask_test.sum())}")

def metrics(pos, price):
    pos = pos.fillna(0).clip(-LIMIT,LIMIT).values
    p = price.values
    dp = np.diff(p, prepend=p[0])
    pnl = pos*dp
    cum = np.cumsum(pnl)
    dd = float((cum-np.maximum.accumulate(cum)).min())
    sh = float(pnl.mean()/pnl.std()*np.sqrt(len(pnl))) if pnl.std()>0 else 0
    return dict(sharpe=round(sh,3), max_dd=round(dd,2), total_pnl=round(float(cum[-1]),2))

print("\n=== Momentum N=M=5 OOS ===")
for s in SYMBOLS:
    p_all = mid[s].dropna()
    sig = np.sign(p_all.diff(5))
    tgt = pd.Series(0.0, index=p_all.index)
    i = 0
    while i < len(p_all):
        si = sig.iloc[i] if not np.isnan(sig.iloc[i]) else 0
        tgt.iloc[i:i+5] = si*LIMIT
        i += 5
    p_train = p_all[mask_train.reindex(p_all.index).fillna(False).values]
    p_test  = p_all[mask_test.reindex(p_all.index).fillna(False).values]
    t_train = tgt.reindex(p_train.index)
    t_test  = tgt.reindex(p_test.index)
    m_tr = metrics(t_train, p_train); m_te = metrics(t_test, p_test)
    print(f"  {s:<22} train PnL={m_tr['total_pnl']:>8.0f} sh={m_tr['sharpe']:>6.2f} | test PnL={m_te['total_pnl']:>8.0f} sh={m_te['sharpe']:>6.2f}")

print("\n=== Pair PISTACHIO|STRAWBERRY OOS ===")
a, b = "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"
pa_full = np.log(mid[a].dropna()); pb_full = np.log(mid[b].dropna())
common = pa_full.index.intersection(pb_full.index)
pa, pb = pa_full.loc[common], pb_full.loc[common]
mask_tr = pd.Series((common // 1_000_000) == 2, index=common)
# fit hedge ratio on train only
Xb = sm.add_constant(pb[mask_tr]); ols = sm.OLS(pa[mask_tr], Xb).fit()
beta = ols.params.iloc[1]; print(f"  hedge beta (train only): {beta:.4f}")
spr = pa - beta*pb
mu = spr.rolling(200).mean(); sd = spr.rolling(200).std()
z = (spr-mu)/sd
pos_a = pd.Series(0.0, index=common); pos_b = pd.Series(0.0, index=common)
cur = 0
for i in range(len(common)):
    zi = z.iloc[i]
    if np.isnan(zi): pos_a.iloc[i]=cur*LIMIT; pos_b.iloc[i]=-cur*LIMIT; continue
    if cur==0:
        if zi<-1.5: cur=1
        elif zi>1.5: cur=-1
    else:
        if abs(zi)<0.3: cur=0
    pos_a.iloc[i]=cur*LIMIT; pos_b.iloc[i]=-cur*LIMIT

p_a = mid[a].loc[common]; p_b = mid[b].loc[common]
def pair_pnl(pos_a, pos_b, p_a, p_b):
    da = np.diff(p_a.values, prepend=p_a.values[0])
    db = np.diff(p_b.values, prepend=p_b.values[0])
    pnl = pos_a.values * da + pos_b.values * db
    cum = np.cumsum(pnl)
    dd = float((cum-np.maximum.accumulate(cum)).min())
    sh = float(pnl.mean()/pnl.std()*np.sqrt(len(pnl))) if pnl.std()>0 else 0
    return round(float(cum[-1]),2), round(sh,3), round(dd,2)
m_tr = pair_pnl(pos_a[mask_tr], pos_b[mask_tr], p_a[mask_tr], p_b[mask_tr])
m_te = pair_pnl(pos_a[~mask_tr], pos_b[~mask_tr], p_a[~mask_tr], p_b[~mask_tr])
print(f"  train: PnL={m_tr[0]:>8.0f}  sh={m_tr[1]:>6.2f}  dd={m_tr[2]}")
print(f"  test : PnL={m_te[0]:>8.0f}  sh={m_te[1]:>6.2f}  dd={m_te[2]}")

# Per-day breakdown of momentum strategy on each symbol
print("\n=== Momentum N=M=5 per-day PnL ===")
for s in SYMBOLS:
    p_all = mid[s].dropna()
    sig = np.sign(p_all.diff(5))
    tgt = pd.Series(0.0, index=p_all.index)
    i = 0
    while i < len(p_all):
        si = sig.iloc[i] if not np.isnan(sig.iloc[i]) else 0
        tgt.iloc[i:i+5] = si*LIMIT
        i += 5
    daypnl = {}
    for d in (2,3,4):
        msk = (p_all.index // 1_000_000) == d
        m = metrics(tgt[msk], p_all[msk])
        daypnl[d] = m["total_pnl"]
    print(f"  {s:<22} d2={daypnl[2]:>8.0f}  d3={daypnl[3]:>8.0f}  d4={daypnl[4]:>8.0f}")
