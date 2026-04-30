# OXYGEN_SHAKE — squeeze-the-spread MM strategies

**Date:** 2026-04-30
**Question:** can we extract more PnL from the 12-15 tick BBO by tweaking the MM layer?
**Answer:** **YES — switching OXYGEN_SHAKE to join-best (offset=0) yields +7.5k.**
Nothing else moved the needle in backtest.

---

## Setup

Baseline: `FINAL_GLAUCO.py` MM layer = `offset=+1`, `size=5`, no skew.
- 3-day total: **787,931** (OXY contribution ≈ 67.2k)
- BBO spread distribution: median 12-15 ticks, p10 ≈ p90 (very stable)
- Trades per symbol per day: 230-255 (avg qty 2.5, max 4)
- Order book depth: ~18 lots at L1, ~31 at L2, both sides
- All trades happen exactly at BBO (no trades inside or outside the spread)

Critical micro-structure facts that constrained the search:
1. **No trades go past BBO** → quotes deeper than `bb`/`ba` never fill
2. **Max trade size is 4 lots/tick** → size > 4 per side wastes capacity
3. **Half-life of mid > 1,500 ticks** → no useful mean-reversion to a rolling fair
4. **Imbalance corr w/ next-tick mid ≈ 0.05** → too weak for skew without losing fills

## Strategies tried (real backtester, days 2/3/4)

| File                          | Strategy                                       | Total PnL  | Δ vs base  |
|-------------------------------|------------------------------------------------|------------|------------|
| `FINAL_GLAUCO.py`             | baseline: offset+1, size 5                     | 787,931    | —          |
| `oxy_mm_v4_join.py`           | **offset=0 (join-best), size 5**               | **795,482**| **+7,551** |
| `oxy_mm_v10_join_size10.py`   | offset=0, size 10                              | 795,482    | +7,551     |
| `oxy_mm_v11_imbalance_skew.py`| offset=0, ±1 tick on imbalance > 0.6           | 792,980    | +5,049     |
| `oxy_mm_v12_pos_skew.py`      | offset=0, step both quotes 1-2 ticks on `|pos|≥4` | 768,588 | −19,343    |
| `oxy_mm_v13_dual_layer.py`    | offset=0 + 2nd tier at bb−1/ba+1 size 5        | 795,482    | +7,551     |
| `oxy_mm_v9_two_levels.py`     | offset+1 + 2nd tier at bb/ba                   | 787,931    | 0          |

(`oxy_mm_v10`, `v13` produce identical PnL to `v4` because no trades go past
BBO so the extra capacity / extra layers never fill.)

## Per-symbol gain from v4 (3-day sum vs baseline)

| Symbol            | Baseline | v4 (join) | Δ      |
|-------------------|----------|-----------|--------|
| CHOCOLATE         | 23,573   | 25,083    | +1,510 |
| EVENING_BREATH    | 20,865   | 22,375    | +1,510 |
| GARLIC            |  9,400   | 10,911    | +1,511 |
| MINT              |   −558   |    952    | +1,510 |
| MORNING_BREATH    | 13,710   | 15,221    | +1,511 |
| **Family total**  | **66,990** | **74,542** | **+7,552** |

Identical +1,510 per symbol per day (3 days × 5 symbols × ~500 = 7,552).
The +1 tick extra captured per round-trip × ~500 round-trips per symbol per day.

## Why v4 works (and other ideas don't)

`runner.py:142` shows the matching engine: a passive buy at `order.price`
fills via market trades whose price is **≤ order.price**. Since all OXY
sell-aggressive trades happen at exactly `best_bid`, our bid at `bb` and at
`bb+1` fill *the same number of times*, but at `bb` we pay 1 less tick.
Symmetric on the ask. So join-best captures the full BBO spread per
round-trip instead of `spread − 2`.

The other strategies fail for predictable reasons:
- **size > 5**: trades cap at 4 lots/tick → extra capacity unused.
- **deeper layers (bb−1, etc.)**: never reached by market orders.
- **imbalance skew**: the alpha (~0.05 corr) is too weak; the lost-fills
  cost more than the directional benefit.
- **position skew**: stepping quotes by even 1 tick costs more lost
  fills than the inventory hedge is worth — spread is wide but trade
  flow is tight.

## Recommendation

**Promote `v4`**: add `OXYGEN_SHAKE_` to the `depth_off=0` branch in the
MM layer of `FINAL_GLAUCO.py:284`. Single-line change.
Expected gain: +7,551 across days 2-4 (validated with backtester).

## Files

- `mm_strategy_sim.py`             — analytical fill sim across 9 strategies
- `mm_strategy_results.csv`        — per-(symbol, strategy) PnL from sim
- `oxy_mm_v{4,9,10,11,12,13}_*.py` — backtester-validated variants
- `MM_VERDICT.md`                  — this file
