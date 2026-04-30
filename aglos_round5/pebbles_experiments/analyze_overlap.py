"""Analyze overlap: when v01 (pair) and v02 (MM) trade XS or S,
do they go in the same or opposite direction?

We log ticks where each strategy holds non-zero target on XS/S,
and check if signs agree."""
import subprocess
import re
import json
from pathlib import Path

BTX = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_windteam/.venv/bin/prosperity4btx"
DATA = "/tmp/p4data"


def collect_positions(strategy_file: str, day: int):
    """Run backtest, parse activities CSV from sandbox/log to extract positions per tick."""
    out_file = Path(f"/tmp/bt_{Path(strategy_file).stem}_{day}.log")
    if out_file.exists():
        out_file.unlink()
    subprocess.run(
        [BTX, str(Path(strategy_file).resolve()), f"5-{day}",
         "--out", str(out_file), "--data", DATA, "--no-progress"],
        capture_output=True, text=True,
    )
    return out_file


def parse_log(log_file: Path):
    """Extract Activities log section: timestamp + product + position is in own_trades / market data.
    The Prosperity4 .log format has a sandbox section with structured JSON per tick."""
    tick_pos = {}  # (timestamp, product) -> position after own_trades
    text = log_file.read_text()
    # Sandbox section: each tick is a line `{"sandboxLog":"...","lambdaLog":"<json>"}` or similar
    # Easier: look for "Activities log" CSV at the end
    if "Activities log:" in text:
        chunk = text.split("Activities log:")[1].split("Trade History:")[0]
        # The chunk is CSV with header day;timestamp;product;... ;mid_price;profit_and_loss
        return chunk
    return None


def own_trades_from_log(log_file: Path, products):
    """Extract own_trades per (timestamp, product) by parsing Trade History section."""
    text = log_file.read_text()
    if "Trade History:" not in text:
        return []
    th = text.split("Trade History:")[1].strip()
    try:
        trades = json.loads(th)
    except Exception:
        return []
    out = []
    for t in trades:
        if t.get("symbol") in products and (t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"):
            qty = t["quantity"] if t.get("buyer") == "SUBMISSION" else -t["quantity"]
            out.append((t["timestamp"], t["symbol"], qty, t["price"]))
    return out


if __name__ == "__main__":
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    for strat in ["v01_pair_xs_s_only.py", "v02_constant_sum_mm.py"]:
        for d in [2]:
            log = collect_positions(strat, d)
            trades = own_trades_from_log(log, PRODUCTS)
            print(f"\n{strat} day {d}: {len(trades)} own trades")
            # aggregate per product
            agg = {}
            for ts, sym, q, px in trades:
                agg.setdefault(sym, {"buys": 0, "sells": 0, "buy_vol": 0, "sell_vol": 0})
                if q > 0:
                    agg[sym]["buys"] += 1; agg[sym]["buy_vol"] += q
                else:
                    agg[sym]["sells"] += 1; agg[sym]["sell_vol"] += -q
            for sym, a in sorted(agg.items()):
                print(f"  {sym:14s} buys={a['buys']:4d} ({a['buy_vol']:5d}u)  sells={a['sells']:4d} ({a['sell_vol']:5d}u)")
