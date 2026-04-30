"""
02 - Predictive power test.

Per ogni prodotto e ogni segnale candidato (S), calcola il futuro return del
mid_price su finestre di 1, 5, 10, 30, 100 tick:
    fut_ret_h(t) = mid(t+h) - mid(t)

Segnali OB (testati ad ogni tick):
- BIG_BID_LVL1: bid_volume_1 >= P95 di quella distribuzione
- BIG_ASK_LVL1: ask_volume_1 >= P95
- IMBAL_TOTAL: (sum_bid - sum_ask) / (sum_bid + sum_ask)
- BID_DOMINANT: imbal > +0.3 ; ASK_DOMINANT: imbal < -0.3

Segnali TRADES (per ogni tick con trade):
- TRADE_BIG_VOL: quantity >= P95
- Per direction: se trade.price >= mid → "buy" else "sell"

Output: tabella per (symbol, signal, horizon) con:
- n eventi
- mean fut_ret quando segnale ON
- mean fut_ret quando segnale OFF
- t-stat differenza, win rate
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5")
OUT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/volume_signals")
DAYS = [2, 3, 4]
HORIZONS = [1, 5, 10, 30, 100]


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return df


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"trades_round_5_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_future_returns(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for h in HORIZONS:
        g[f"fr_{h}"] = g["mid_price"].shift(-h) - g["mid_price"]
    return g


def total_volumes(row: pd.Series) -> tuple[float, float]:
    bid = sum(row.get(f"bid_volume_{i}", 0) or 0 for i in [1, 2, 3])
    ask = sum(row.get(f"ask_volume_{i}", 0) or 0 for i in [1, 2, 3])
    return bid, ask


def signal_stats(s_on: np.ndarray, s_off: np.ndarray) -> dict:
    if len(s_on) < 30 or len(s_off) < 30:
        return {}
    m_on = np.nanmean(s_on)
    m_off = np.nanmean(s_off)
    sd_on = np.nanstd(s_on, ddof=1)
    sd_off = np.nanstd(s_off, ddof=1)
    se = np.sqrt(sd_on**2 / len(s_on) + sd_off**2 / len(s_off))
    t = (m_on - m_off) / se if se > 0 else 0.0
    win_on = np.nanmean(np.sign(s_on) > 0)
    return {
        "n_on": int(np.sum(~np.isnan(s_on))),
        "mean_on": round(float(m_on), 4),
        "mean_off": round(float(m_off), 4),
        "diff": round(float(m_on - m_off), 4),
        "t_stat": round(float(t), 2),
        "win_rate_on": round(float(win_on), 3),
    }


def analyze_symbol(g_prices: pd.DataFrame, trades_sym: pd.DataFrame) -> list[dict]:
    g = build_future_returns(g_prices)

    # OB total volumes per row
    bid_tot = (
        g[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].fillna(0).sum(axis=1)
    )
    ask_tot = (
        g[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].fillna(0).sum(axis=1)
    )
    g = g.assign(bid_tot=bid_tot, ask_tot=ask_tot)
    g["imbal"] = (g["bid_tot"] - g["ask_tot"]) / (g["bid_tot"] + g["ask_tot"]).replace(0, np.nan)

    # Pre-compute thresholds
    bv1_p95 = g["bid_volume_1"].dropna().quantile(0.95)
    av1_p95 = g["ask_volume_1"].dropna().quantile(0.95)
    bv1_p99 = g["bid_volume_1"].dropna().quantile(0.99)
    av1_p99 = g["ask_volume_1"].dropna().quantile(0.99)

    rows = []
    for h in HORIZONS:
        col = f"fr_{h}"
        valid = g.dropna(subset=[col])

        # BIG_BID_LVL1 P95: ci si aspetterebbe pressione buy → mid up
        m_on = valid.loc[valid["bid_volume_1"] >= bv1_p95, col].values
        m_off = valid.loc[valid["bid_volume_1"] < bv1_p95, col].values
        st = signal_stats(m_on, m_off)
        if st:
            rows.append({"signal": "BIG_BID_LVL1_P95", "horizon": h, **st})

        m_on = valid.loc[valid["ask_volume_1"] >= av1_p95, col].values
        m_off = valid.loc[valid["ask_volume_1"] < av1_p95, col].values
        st = signal_stats(m_on, m_off)
        if st:
            rows.append({"signal": "BIG_ASK_LVL1_P95", "horizon": h, **st})

        m_on = valid.loc[valid["bid_volume_1"] >= bv1_p99, col].values
        m_off = valid.loc[valid["bid_volume_1"] < bv1_p99, col].values
        st = signal_stats(m_on, m_off)
        if st:
            rows.append({"signal": "BIG_BID_LVL1_P99", "horizon": h, **st})

        m_on = valid.loc[valid["ask_volume_1"] >= av1_p99, col].values
        m_off = valid.loc[valid["ask_volume_1"] < av1_p99, col].values
        st = signal_stats(m_on, m_off)
        if st:
            rows.append({"signal": "BIG_ASK_LVL1_P99", "horizon": h, **st})

        # Imbalance
        m_on = valid.loc[valid["imbal"] > 0.3, col].values
        m_off = valid.loc[valid["imbal"] < -0.3, col].values
        st = signal_stats(m_on, m_off)
        if st:
            rows.append({"signal": "IMBAL_BID>ASK_03", "horizon": h, **st})

        m_on = valid.loc[valid["imbal"] > 0.5, col].values
        m_off = valid.loc[valid["imbal"] < -0.5, col].values
        st = signal_stats(m_on, m_off)
        if st:
            rows.append({"signal": "IMBAL_BID>ASK_05", "horizon": h, **st})

    # Trades-based: aggrega trades su ogni timestamp, segnale = max trade volume
    if not trades_sym.empty:
        tr_p95 = trades_sym["quantity"].quantile(0.95)
        tr_max_per_ts = trades_sym.groupby("timestamp")["quantity"].max()
        trade_lookup = tr_max_per_ts.to_dict()

        # Sub usa solo day=2 timestamps; il trade file sembra avere day implicito.
        # Per sicurezza, uniamo via (day, timestamp) se trades hanno day.
        if "day" in trades_sym.columns:
            tr_grp = trades_sym.groupby(["day", "timestamp"])["quantity"].max().to_dict()
            g["trade_max"] = [
                tr_grp.get((d, t), 0) for d, t in zip(g["day"], g["timestamp"])
            ]
        else:
            g["trade_max"] = g["timestamp"].map(trade_lookup).fillna(0)

        for h in HORIZONS:
            col = f"fr_{h}"
            valid = g.dropna(subset=[col])
            m_on = valid.loc[valid["trade_max"] >= tr_p95, col].values
            m_off = valid.loc[
                (valid["trade_max"] > 0) & (valid["trade_max"] < tr_p95), col
            ].values
            st = signal_stats(m_on, m_off)
            if st:
                rows.append({"signal": f"TRADE_BIG_P95(>={tr_p95:.0f})", "horizon": h, **st})

    return rows


def main() -> None:
    prices = load_prices()
    trades = load_trades()

    all_rows = []
    for sym, g in prices.groupby("product", sort=False):
        tr = trades[trades["symbol"] == sym]
        for r in analyze_symbol(g, tr):
            r["symbol"] = sym
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    df = df[["symbol", "signal", "horizon", "n_on", "mean_on", "mean_off", "diff",
             "t_stat", "win_rate_on"]]
    df.to_csv(OUT / "predictive_power_all.csv", index=False)

    # Top edges by |t_stat|
    print("\n=== Top 30 |t_stat| signals across all products ===")
    top = df.reindex(df["t_stat"].abs().sort_values(ascending=False).index).head(30)
    print(top.to_string(index=False))

    # Also: per signal type, average effect across products
    agg = (
        df.groupby(["signal", "horizon"])
        .agg(n_symbols=("symbol", "nunique"),
             mean_diff=("diff", "mean"),
             mean_t=("t_stat", "mean"),
             max_abs_t=("t_stat", lambda s: s.abs().max()))
        .round(3)
        .sort_values("max_abs_t", ascending=False)
    )
    agg.to_csv(OUT / "predictive_power_agg.csv")
    print("\n=== Aggregate by (signal, horizon), top 20 ===")
    print(agg.head(20))


if __name__ == "__main__":
    main()
