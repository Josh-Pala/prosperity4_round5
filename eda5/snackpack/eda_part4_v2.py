"""SNACKPACK Part 4 v2: cleaner backtests with consistent PnL accounting.

Per-tick MTM PnL = position_held_during_tick * mid_price_change.
Sharpe computed from per-tick PnL; total realized PnL is the cum sum.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm

OUT = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/snackpack"
SYMBOLS = ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO",
           "SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"]
LIMIT = 10
mid = pd.read_parquet(f"{OUT}/_mid.parquet")
vol = pd.read_parquet(f"{OUT}/_vol.parquet")

def metrics(pos, price):
    """pos held entering tick i applied to dp[i]=price[i]-price[i-1]."""
    pos = pos.fillna(0).clip(-LIMIT, LIMIT).values
    p = price.values
    dp = np.diff(p, prepend=p[0])
    pnl = pos * dp
    cum = np.cumsum(pnl)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    sharpe = float(pnl.mean()/pnl.std()*np.sqrt(len(pnl))) if pnl.std()>0 else 0.0
    # round-trip trades: position changes
    trades = []
    cur_pos, cur_entry = 0, None
    for i in range(len(pos)):
        if pos[i] != cur_pos:
            if cur_pos != 0 and cur_entry is not None:
                trades.append(cur_pos * (p[i] - cur_entry))
            cur_entry = p[i] if pos[i] != 0 else None
            cur_pos = pos[i]
    win = float(np.mean([t>0 for t in trades])) if trades else 0
    mean_t = float(np.mean(trades)) if trades else 0
    return dict(sharpe=round(sharpe,3), win=round(win,3),
                mean_pnl_trade=round(mean_t,3), max_dd=round(dd,2),
                n_trades=len(trades), total_pnl=round(float(cum[-1]),2))

# ---- 4.1 Mean reversion z-score
print("=== 4.1 Mean-reversion z-score (window=50, ±1.5 / 0.3) ===")
mr_rows = []
for s in SYMBOLS:
    p = mid[s].dropna()
    mu, sd = p.rolling(50).mean(), p.rolling(50).std()
    z = (p-mu)/sd
    pos = pd.Series(0.0, index=p.index)
    cur = 0
    for i in range(len(p)):
        zi = z.iloc[i]
        if np.isnan(zi): pos.iloc[i] = cur; continue
        if cur==0:
            if zi<-1.5: cur = LIMIT
            elif zi>1.5: cur = -LIMIT
        else:
            if abs(zi)<0.3: cur = 0
        pos.iloc[i] = cur
    m = metrics(pos, p); m["symbol"]=s; m["strat"]="zscore"
    mr_rows.append(m)
mr = pd.DataFrame(mr_rows); print(mr.to_string(index=False))
mr.to_csv(f"{OUT}/p4_mean_revert.csv", index=False)

# ---- 4.2 Momentum (N-tick lookback, hold M with overlap allowed via discrete decisions every M)
print("\n=== 4.2 Momentum (best N,M per symbol) ===")
mom_rows = []
for s in SYMBOLS:
    p = mid[s].dropna()
    best = None
    for N in (5,10,20):
        for M in (5,10,20):
            sig = np.sign(p.diff(N))
            tgt = pd.Series(0.0, index=p.index)
            i = 0
            while i < len(p):
                si = sig.iloc[i] if not np.isnan(sig.iloc[i]) else 0
                tgt.iloc[i:i+M] = si * LIMIT
                i += M
            mtr = metrics(tgt, p); mtr.update({"symbol":s,"N":N,"M":M,"strat":"momentum"})
            if best is None or mtr["total_pnl"] > best["total_pnl"]:
                best = mtr
    mom_rows.append(best)
mom = pd.DataFrame(mom_rows); print(mom.to_string(index=False))
mom.to_csv(f"{OUT}/p4_momentum.csv", index=False)

# ---- 4.3 Volume-triggered momentum (try lower threshold)
print("\n=== 4.3 Volume-triggered momentum (vol > 1.5x rolling, hold 5) ===")
vt_rows = []
for s in SYMBOLS:
    p = mid[s].dropna()
    v = vol[s].reindex(p.index)
    r = p.diff()
    vmean = v.rolling(20).mean()
    HOLD = 5
    tgt = pd.Series(0.0, index=p.index)
    i = 0
    fires = 0
    while i < len(p):
        if (not np.isnan(v.iloc[i]) and not np.isnan(vmean.iloc[i]) and
            v.iloc[i] > 1.5*vmean.iloc[i] and not np.isnan(r.iloc[i]) and r.iloc[i]!=0):
            tgt.iloc[i:i+HOLD] = np.sign(r.iloc[i]) * LIMIT
            fires += 1
            i += HOLD
        else:
            i += 1
    m = metrics(tgt, p); m.update({"symbol":s,"strat":"vol_momentum","fires":fires})
    vt_rows.append(m)
vt = pd.DataFrame(vt_rows); print(vt.to_string(index=False))
vt.to_csv(f"{OUT}/p4_vol_momentum.csv", index=False)

# ---- 4.3b OBI-fade (large book imbalance => fade direction, hold 10)
# Per Part 3.4: post-spike decay is negative (price reverts after OBI spike).
print("\n=== 4.3b OBI fade (top 5% |signed_v|, hold 10) ===")
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

obi_rows = []
for s in SYMBOLS:
    sub = raw[raw["product"]==s].copy().reset_index(drop=True)
    p = sub["sub_t" if False else "mid_price"]
    sv = sub["signed_v"]
    thr = sv.abs().quantile(0.95)
    HOLD = 10
    tgt = np.zeros(len(sub))
    i = 0
    while i < len(sub):
        if abs(sv.iloc[i]) >= thr and sv.iloc[i] != 0:
            tgt[i:i+HOLD] = -np.sign(sv.iloc[i]) * LIMIT  # FADE
            i += HOLD
        else:
            i += 1
    pos = pd.Series(tgt, index=sub.index)
    m = metrics(pos, p); m.update({"symbol":s,"strat":"obi_fade"})
    obi_rows.append(m)
obi = pd.DataFrame(obi_rows); print(obi.to_string(index=False))
obi.to_csv(f"{OUT}/p4_obi_fade.csv", index=False)

# ---- 4.4 Lead-lag cross signal
print("\n=== 4.4 Lead-lag (Granger-best pairs) ===")
ll_pairs = [("SNACKPACK_CHOCOLATE","SNACKPACK_STRAWBERRY"),
            ("SNACKPACK_STRAWBERRY","SNACKPACK_PISTACHIO"),
            ("SNACKPACK_VANILLA","SNACKPACK_CHOCOLATE")]
ll_rows = []
for leader, follower in ll_pairs:
    pf = mid[follower].dropna(); pl = mid[leader].dropna()
    common = pf.index.intersection(pl.index)
    pf = pf.loc[common]; pl = pl.loc[common]
    sig = np.sign(pl.diff()).shift(1).fillna(0)
    HOLD = 3
    tgt = pd.Series(0.0, index=common)
    i = 0
    while i < len(common):
        si = sig.iloc[i]
        if si != 0:
            tgt.iloc[i:i+HOLD] = si * LIMIT
            i += HOLD
        else:
            i += 1
    m = metrics(tgt, pf); m.update({"symbol":follower, "leader":leader, "strat":"leadlag"})
    ll_rows.append(m)
ll = pd.DataFrame(ll_rows); print(ll.to_string(index=False))
ll.to_csv(f"{OUT}/p4_leadlag.csv", index=False)

# ---- 4.5 Pairs / stat-arb
print("\n=== 4.5 Pairs trading (cointegrated spread, ±1.5σ entry, 0.3σ exit) ===")
def pair_bt(a, b, win=200):
    pa = np.log(mid[a].dropna()); pb = np.log(mid[b].dropna())
    common = pa.index.intersection(pb.index)
    pa, pb = pa.loc[common], pb.loc[common]
    Xb = sm.add_constant(pb); ols = sm.OLS(pa, Xb).fit()
    beta = ols.params.iloc[1]
    spr = pa - beta*pb
    mu, sd = spr.rolling(win).mean(), spr.rolling(win).std()
    z = (spr - mu)/sd
    pos_a = pd.Series(0.0, index=common)
    pos_b = pd.Series(0.0, index=common)
    cur = 0
    for i in range(len(common)):
        zi = z.iloc[i]
        if np.isnan(zi): pos_a.iloc[i]=cur*LIMIT; pos_b.iloc[i]=-cur*LIMIT; continue
        if cur==0:
            if zi<-1.5: cur = 1
            elif zi>1.5: cur = -1
        else:
            if abs(zi)<0.3: cur = 0
        pos_a.iloc[i] = cur * LIMIT
        pos_b.iloc[i] = -cur * LIMIT
    p_a = mid[a].loc[common]; p_b = mid[b].loc[common]
    m_a = metrics(pos_a, p_a); m_b = metrics(pos_b, p_b)
    # combined PnL
    combined = (pos_a.fillna(0).values * np.diff(p_a.values, prepend=p_a.values[0])
                + pos_b.fillna(0).values * np.diff(p_b.values, prepend=p_b.values[0]))
    cum = np.cumsum(combined)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    sharpe = float(combined.mean()/combined.std()*np.sqrt(len(combined))) if combined.std()>0 else 0
    # trade count: pos_a transitions
    trades = []
    cur_pos = 0; entry_pa = entry_pb = 0
    for i in range(len(pos_a)):
        if pos_a.iloc[i] != cur_pos:
            if cur_pos != 0:
                trades.append(cur_pos*(p_a.iloc[i]-entry_pa) - cur_pos*(p_b.iloc[i]-entry_pb))
            entry_pa = p_a.iloc[i]; entry_pb = p_b.iloc[i]
            cur_pos = pos_a.iloc[i]
    win_rt = float(np.mean([t>0 for t in trades])) if trades else 0
    mean_t = float(np.mean(trades)) if trades else 0
    return dict(beta=round(beta,3), sharpe=round(sharpe,3), win=round(win_rt,3),
                mean_pnl_trade=round(mean_t,3), max_dd=round(dd,2),
                n_trades=len(trades), total_pnl=round(float(cum[-1]),2))

pair_rows = []
for a,b in [("SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY"),
            ("SNACKPACK_CHOCOLATE","SNACKPACK_STRAWBERRY"),
            ("SNACKPACK_VANILLA","SNACKPACK_PISTACHIO"),
            ("SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA"),
            ("SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"),
            ("SNACKPACK_PISTACHIO","SNACKPACK_RASPBERRY")]:
    r = pair_bt(a,b); r.update({"pair":f"{a}|{b}", "strat":"pair"})
    pair_rows.append(r)
pp = pd.DataFrame(pair_rows); print(pp.to_string(index=False))
pp.to_csv(f"{OUT}/p4_pairs.csv", index=False)

print("\n=== Part 4 v2 complete ===")
