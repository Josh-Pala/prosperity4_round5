# OXYGEN_SHAKE pair-trading EDA — VERDICT

**Date:** 2026-04-30
**Question:** can the OXYGEN_SHAKE family benefit from pair-trading on top of the
existing market-making layer in `FINAL_GLAUCO.py`?

**Answer: NO. Pair-trading destroys ~6-27k of MM PnL across the 3 backtest days.**
Keep the family on pure MM.

---

## 1. Family is structurally non-cointegrated

From `descriptives.csv` and `corr_returns_*.csv`:

| Metric                                | OXYGEN_SHAKE          | PEBBLES (reference) |
|---------------------------------------|-----------------------|---------------------|
| Sum-invariant range across 3 days     | **9.6%** (variable)   | 0.0% (fixed 50,000) |
| Max abs(corr) on 1-tick returns       | **0.013**             | > 0.95              |
| Max abs(corr) on 10-tick returns      | **0.023**             | > 0.95              |
| Min OLS-residual half-life (any pair) | **1,391 ticks**       | < 200 ticks         |
| Best pair level corr                  | 0.65 (CHOC-GARLIC)    | > 0.99              |

The 5 OXYGEN_SHAKE variants are essentially **independent random walks**.
There is no economic linkage like the PEBBLES sum-invariant or the
SLEEP_POD POLY-COTTON cointegrating vector.

## 2. The pair scanner finds high-PnL configs in simulation

Top mid-price pair simulations (file: `pair_scan_results.csv`):

| Pair                          | Form        | entry_z | Sim PnL | pos_days |
|-------------------------------|-------------|---------|---------|----------|
| MINT vs MORNING_BREATH        | spread      | 1.2     | +29,925 | 3 / 3    |
| EVENING_BREATH vs MINT        | sum         | 1.0     | +27,640 | 3 / 3    |
| MINT vs MORNING_BREATH        | spread      | 1.0     | +26,640 | 3 / 3    |
| MINT vs MORNING_BREATH        | ols_spread  | 1.8     | +25,943 | 3 / 3    |

`robustness_results.csv` confirms these survive entry_z ±0.2, exit_z swap,
warmup 2000, and 5x subsampling — so **the simulator output is internally
consistent and not a tuning fluke**.

## 3. But the simulator is wrong about execution cost

| Variant                  | Total backtest PnL | Δ vs FINAL_GLAUCO | Notes                                                    |
|--------------------------|--------------------|-------------------|----------------------------------------------------------|
| FINAL_GLAUCO (baseline)  | 787,931            | —                 | OXY = pure MM, ~67k                                       |
| `oxy_v1_two_pairs.py`    | 760,627            | **−27,304**       | both pairs, aggressive entry                              |
| `oxy_v2_passive.py`      | 773,777            | **−14,154**       | both pairs, passive entry (OXY ∈ PASSIVE_ENTRY_FAMILIES) |
| `oxy_v3_one_pair.py`     | 781,641            | **−6,290**        | only MINT-MORN spread @ 1.2, passive                      |

Per-symbol breakdown (`oxy_v3` vs baseline, 3-day sum):

| Symbol           | Baseline | v3      | Δ        |
|------------------|----------|---------|----------|
| CHOCOLATE        | +23,573  | +23,573 | 0        |
| EVENING_BREATH   | +20,865  | +20,865 | 0        |
| GARLIC           |  +9,400  |  +9,400 | 0        |
| MINT             |   −558   |  +4,784 | **+5,342** |
| MORNING_BREATH   | +13,710  |  +2,079 | **−11,631** |

The pair lifts MINT (+5.3k) but breaks MORNING_BREATH (−11.6k).
**Net = −6.3k**.

## 4. Why the simulator over-estimates

OXYGEN_SHAKE BBO spread is **12-15 ticks median** (file: `descriptives.csv`).
The pair simulator marks every leg at mid; the real exchange charges spread/2
per fill at minimum. Even with passive offset+1, every entry/exit costs
~6 ticks per leg vs the simulator's 0. Pair signal SD ≈ 730-870 ticks,
half-life > 1,400 ticks → average MTM gain per round-trip is small relative
to the round-trip transaction cost.

The same execution-cost argument is why TRANSLATOR was *removed* from
PASSIVE_ENTRY_FAMILIES: the spread there was small enough that aggressive
*was* cheaper than passive. OXYGEN_SHAKE has the **opposite problem**:
the spread is too large for *either* execution mode to be profitable on
the available pair signal.

## 5. Why the existing pure-MM layer works

Pure MM at offset+1 captures the BBO edge directly: each round-trip
earns ~`spread - 2` ticks. With spreads of 12-15 ticks, that's a 10-13
tick captured edge per filled round-trip — which compounds to ~67k
across 3 days without any directional view.

Pair-trading would only beat MM if the cointegrating signal had an
expected-return-per-trade larger than the BBO. It does not.

## 6. Decision

- **Do not add OXYGEN_SHAKE pairs to `PAIRS` in `FINAL_GLAUCO.py`.**
- Keep all 5 OXYGEN_SHAKE symbols in `MM_UNIVERSE`.
- File the experiments in `aglos_round5/hybrid_experiments/oxy_v{1,2,3}_*.py`
  for reference; do not promote.

## Files

- `eda_oxygen_shake.py`        — descriptive EDA, sum-check, corr matrices
- `descriptives.csv`           — per-symbol mean / sd / spread / drift / half-life
- `sum_check.csv`              — failed PEBBLES-style invariant test
- `corr_levels.csv` / `corr_returns_{1,10}.csv` — pairwise correlations
- `spread_stats.csv`           — pair spread / sum / OLS-resid stats
- `prices_norm.png`            — visual: 5 independent walks
- `rolling_corr.png`           — rolling-1000 return correlation (top pairs)
- `pair_scanner.py`            — full grid scan
- `pair_scan_results.csv`      — 180 (pair × form × entry_z) PnL rows
- `robustness_check.py`        — perturbation tests
- `robustness_results.csv`     — robustness PnL
- `VERDICT.md`                 — this file
