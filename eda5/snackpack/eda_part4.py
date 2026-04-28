"""SNACKPACK EDA — Part 4: Strategy signal backtests.

Uses mid-price and assumes execution at mid (zero-cost) — a relative-comparison
backtest, not a P&L-realistic one. Position limit ±10. We compute Sharpe, win
rate, mean PnL/trade, max drawdown for each signal.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

OUT_DIR = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/snackpack"
SYMBOLS = ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
           "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"]
POS_LIMIT = 10

mid = pd.read_parquet(f"{OUT_DIR}/_mid.parquet")
ret = pd.read_parquet(f"{OUT_DIR}/_ret.parquet")

def stats_from_pnl(pnl_per_tick, trades):
    """pnl_per_tick: pd.Series of per-tick mark-to-market PnL.
       trades: list of per-trade PnL realizations."""
    pnl_per_tick = pnl_per_tick.fillna(0)
    cum = pnl_per_tick.cumsum()
    dd = (cum - cum.cummax()).min()
    sharpe = (pnl_per_tick.mean() / pnl_per_tick.std() * np.sqrt(len(pnl_per_tick))) if pnl_per_tick.std()>0 else 0
    if not trades:
        return dict(sharpe=0, win=0, mean=0, dd=0, n_trades=0, total=0)
    wr = float(np.mean([t>0 for t in trades]))
    return dict(sharpe=round(float(sharpe),3),
                win=round(wr,3),
                mean=round(float(np.mean(trades)),3),
                dd=round(float(dd),2),
                n_trades=len(trades),
                total=round(float(np.sum(trades)),2))

def run_position_strategy(price, signal_pos):
    """signal_pos: target position in {-POS_LIMIT,..,POS_LIMIT} per tick.
       Returns (per_tick_pnl, list_of_trade_pnls)."""
    price = price.values
    pos = signal_pos.reindex(signal_pos.index).fillna(0).clip(-POS_LIMIT,POS_LIMIT).values.astype(float)
    # PnL: position held over the next move
    dp = np.diff(price, prepend=price[0])
    per_tick = pos * dp  # PnL on existing position from price move
    # Now identify trades: changes in position. Each round-trip = entry-to-flat sequence.
    trades = []
    entry_px = None; entry_pos = 0
    for i in range(1, len(pos)):
        if pos[i-1]==0 and pos[i]!=0:
            entry_px = price[i]; entry_pos = pos[i]
        elif entry_pos!=0 and (pos[i]==0 or np.sign(pos[i])!=np.sign(entry_pos)):
            exit_px = price[i]
            trades.append(entry_pos * (exit_px - entry_px))
            entry_px = None; entry_pos = 0
            if pos[i]!=0:
                entry_px = price[i]; entry_pos = pos[i]
    return pd.Series(per_tick, index=signal_pos.index), trades

# ===== 4.1 Mean reversion (z-score on price) =====
print("=== 4.1 Mean reversion (z=±1.5 entry, |z|<0.3 exit) ===")
mr_rows = []
for s in SYMBOLS:
    p = mid[s].dropna()
    mu = p.rolling(50).mean(); sd = p.rolling(50).std()
    z = (p - mu)/sd
    pos = pd.Series(0.0, index=p.index)
    cur = 0
    for i in range(len(p)):
        zi = z.iloc[i]
        if np.isnan(zi): pos.iloc[i]=cur; continue
        if cur==0:
            if zi<-1.5: cur = POS_LIMIT
            elif zi>1.5: cur = -POS_LIMIT
        else:
            if abs(zi)<0.3: cur = 0
        pos.iloc[i] = cur
    pnl, trades = run_position_strategy(p, pos)
    st = stats_from_pnl(pnl, trades); st["symbol"]=s; st["strat"]="zscore_revert"
    mr_rows.append(st)
mr_df = pd.DataFrame(mr_rows); print(mr_df.to_string(index=False))
mr_df.to_csv(f"{OUT_DIR}/p4_mean_revert.csv", index=False)

# ===== 4.2 Momentum =====
print("\n=== 4.2 Momentum (sign of N-tick return, hold M) ===")
mom_rows = []
for s in SYMBOLS:
    p = mid[s].dropna()
    best = None
    for N in (5,10,20):
        for M in (5,10,20):
            sig = np.sign(p.diff(N))
            # hold M ticks: position = sig at entry, decays after M ticks
            pos = sig.copy()
            # forward-fill nothing — instead replicate signal but cap at every M ticks: decision every M
            pos = sig.fillna(0)
            # take signal only every M-th tick, hold for M ticks
            tgt = pd.Series(0.0, index=p.index)
            i = 0
            while i < len(p):
                s_i = pos.iloc[i] if not np.isnan(pos.iloc[i]) else 0
                tgt.iloc[i:i+M] = s_i * POS_LIMIT
                i += M
            pnl, trades = run_position_strategy(p, tgt)
            st = stats_from_pnl(pnl, trades); st.update({"symbol":s, "N":N, "M":M, "strat":"momentum"})
            if best is None or st["sharpe"] > best["sharpe"]:
                best = st
    mom_rows.append(best)
mom_df = pd.DataFrame(mom_rows); print(mom_df.to_string(index=False))
mom_df.to_csv(f"{OUT_DIR}/p4_momentum.csv", index=False)

# ===== 4.3 Volume-triggered momentum =====
print("\n=== 4.3 Volume-triggered momentum ===")
vol = pd.read_parquet(f"{OUT_DIR}/_vol.parquet")
vt_rows = []
for s in SYMBOLS:
    p = mid[s].dropna()
    v = vol[s].reindex(p.index)
    r = p.diff()
    vmean = v.rolling(20).mean()
    HOLD = 5
    tgt = pd.Series(0.0, index=p.index)
    i = 0
    while i < len(p):
        if not np.isnan(v.iloc[i]) and not np.isnan(vmean.iloc[i]) and v.iloc[i] > 2*vmean.iloc[i] and not np.isnan(r.iloc[i]) and r.iloc[i]!=0:
            tgt.iloc[i:i+HOLD] = np.sign(r.iloc[i]) * POS_LIMIT
            i += HOLD
        else:
            i += 1
    pnl, trades = run_position_strategy(p, tgt)
    st = stats_from_pnl(pnl, trades); st.update({"symbol":s, "strat":"vol_momentum"})
    vt_rows.append(st)
vt_df = pd.DataFrame(vt_rows); print(vt_df.to_string(index=False))
vt_df.to_csv(f"{OUT_DIR}/p4_vol_momentum.csv", index=False)

# ===== 4.4 Lead-lag cross signal =====
# Use Granger best: SNACKPACK_CHOCOLATE -> SNACKPACK_STRAWBERRY (min p)
print("\n=== 4.4 Lead-lag cross signal ===")
ll_rows = []
leader_follower_pairs = [
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_STRAWBERRY"),
    ("SNACKPACK_STRAWBERRY", "SNACKPACK_PISTACHIO"),
    ("SNACKPACK_VANILLA", "SNACKPACK_CHOCOLATE"),
]
for leader, follower in leader_follower_pairs:
    p_f = mid[follower].dropna()
    p_l = mid[leader].dropna()
    common = p_f.index.intersection(p_l.index)
    p_f = p_f.loc[common]; p_l = p_l.loc[common]
    sig = np.sign(p_l.diff()).shift(1)  # leader's t-1 return sign
    HOLD = 3
    tgt = pd.Series(0.0, index=common)
    i = 0
    while i < len(common):
        s_i = sig.iloc[i] if not np.isnan(sig.iloc[i]) else 0
        if s_i != 0:
            tgt.iloc[i:i+HOLD] = s_i * POS_LIMIT
            i += HOLD
        else:
            i += 1
    pnl, trades = run_position_strategy(p_f, tgt)
    st = stats_from_pnl(pnl, trades); st.update({"symbol":follower, "leader":leader, "strat":"leadlag"})
    ll_rows.append(st)
ll_df = pd.DataFrame(ll_rows); print(ll_df.to_string(index=False))
ll_df.to_csv(f"{OUT_DIR}/p4_leadlag.csv", index=False)

# ===== 4.5 Pairs / stat-arb =====
# Best pair from p3: PISTACHIO|STRAWBERRY (EG p≈0.036). But CHOCOLATE|VANILLA has corr -0.91 — try too.
# Backtest spread mean reversion.
print("\n=== 4.5 Pairs / stat-arb ===")
import statsmodels.api as sm
def backtest_pair(a, b):
    pa = np.log(mid[a].dropna()); pb = np.log(mid[b].dropna())
    common = pa.index.intersection(pb.index)
    pa = pa.loc[common]; pb = pb.loc[common]
    Xb = sm.add_constant(pb); ols = sm.OLS(pa, Xb).fit()
    beta = ols.params.iloc[1]
    spread = pa - beta*pb
    mu = spread.rolling(200).mean(); sd = spread.rolling(200).std()
    z = (spread - mu) / sd
    # signal: long A short B when z<-1.5; short A long B when z>+1.5; flat when |z|<0.3
    pos_a = pd.Series(0.0, index=common)
    pos_b = pd.Series(0.0, index=common)
    cur = 0
    for i in range(len(common)):
        zi = z.iloc[i]
        if np.isnan(zi): continue
        if cur==0:
            if zi<-1.5: cur = 1
            elif zi>1.5: cur = -1
        else:
            if abs(zi)<0.3: cur = 0
        pos_a.iloc[i] = cur * POS_LIMIT
        pos_b.iloc[i] = -cur * POS_LIMIT  # symmetric notional cap by limit, ignoring beta scaling
    pnl_a, trades_a = run_position_strategy(mid[a].loc[common], pos_a)
    pnl_b, trades_b = run_position_strategy(mid[b].loc[common], pos_b)
    pnl = pnl_a + pnl_b
    # combine round-trips
    trades = []
    in_pos = False; entry_pa = entry_pb = pa_dir = 0
    for i in range(1, len(common)):
        if not in_pos and pos_a.iloc[i]!=0 and pos_a.iloc[i-1]==0:
            in_pos = True; pa_dir = pos_a.iloc[i]
            entry_pa = mid[a].iloc[i]; entry_pb = mid[b].iloc[i]
        elif in_pos and pos_a.iloc[i]==0:
            in_pos = False
            trades.append(pa_dir * (mid[a].iloc[i]-entry_pa) - pa_dir * (mid[b].iloc[i]-entry_pb))
    st = stats_from_pnl(pnl, trades)
    return st

pair_rows = []
for a,b in [("SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY"),
            ("SNACKPACK_CHOCOLATE","SNACKPACK_STRAWBERRY"),
            ("SNACKPACK_VANILLA","SNACKPACK_PISTACHIO"),
            ("SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA")]:
    st = backtest_pair(a,b); st.update({"pair":f"{a}|{b}", "strat":"pair"})
    pair_rows.append(st)
pair_df = pd.DataFrame(pair_rows); print(pair_df.to_string(index=False))
pair_df.to_csv(f"{OUT_DIR}/p4_pairs.csv", index=False)

print("\nPart 4 complete.")
