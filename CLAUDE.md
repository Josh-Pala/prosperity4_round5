# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

IMC Prosperity 4 — a competitive algorithmic trading challenge. You write a Python `Trader` class that runs on IMC's exchange simulation, submitting orders each tick to maximize XIRECs (in-game profit). This repo covers Round 5 (the final round).

## Repo Layout

```
aglos_round5/
  trader.py        # main submission file — edit this
  datamodel.py     # official IMC datamodel — do not modify
Data_ROUND_5/
  prices_round_5_day_{2,3,4}.csv   # per-tick order book snapshots (semicolon-separated)
  trades_round_5_day_{2,3,4}.csv   # executed market trades
eda5/              # EDA scripts (currently empty)
Reference/
  game_mechanics.pdf
  Python_documentation.pdf
  prosperity_general.pdf
Round 5 - "The Final Stretch".pdf  # round brief
```

## Submission Workflow

Algorithms are submitted via the IMC Prosperity 4 dashboard — drag-and-drop the `.py` file onto the XIREN capsule in the Upload & Changelog window, or use "Select File". There is no local build/test runner; iteration happens by uploading `aglos_round5/trader.py` and reviewing the debug logs from the platform.

## Architecture

### The Trader Contract

Every submission must be a single `.py` file containing a `Trader` class with a `run` method:

```python
def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
    # returns: orders, conversions, traderData
```

The simulation is stateless (AWS Lambda): **class-level variables do not persist between ticks**. All cross-tick state must be serialised into the returned `traderData` string and read back from `state.traderData` on the next call. The platform truncates `traderData` to 50,000 characters.

### Key Datamodel Types (`aglos_round5/datamodel.py`)

- `TradingState` — injected each tick; contains `order_depths`, `own_trades`, `market_trades`, `position`, `observations`, `timestamp`, `traderData`
- `OrderDepth` — `buy_orders: Dict[price, qty]` and `sell_orders: Dict[price, negative_qty]`; sell quantities are negative
- `Order(symbol, price, quantity)` — positive qty = buy, negative qty = sell
- `Observation.conversionObservations` — per-product `ConversionObservation` with external market data (bid/ask prices, transport fees, tariffs, sunlight index, sugar price)

**Do not modify `datamodel.py`** — it is the official IMC-provided file and must stay in sync with the platform version.

### Logger Pattern

`aglos_round5/trader.py` already contains the standard `Logger` class. Call `logger.flush(state, result, conversions, trader_data)` at the end of `run()` — this emits the structured JSON the platform's log viewer parses. The log budget is 3,750 characters total; `Logger.truncate` uses binary search to fit within that limit.

### Position Limit Enforcement

The exchange cancels **all** orders for a symbol if the net quantity submitted in one tick would breach the position limit (long or short). Always track `buy_cap = limit - position` and `sell_cap = limit + position` and clamp order quantities to these caps.

### Round 5 Products

50 tradeable products across 10 families (5 variants each):

| Family | Symbols |
|---|---|
| GALAXY_SOUNDS | BLACK_HOLES, DARK_MATTER, PLANETARY_RINGS, SOLAR_FLAMES, SOLAR_WINDS |
| MICROCHIP | CIRCLE, OVAL, RECTANGLE, SQUARE, TRIANGLE |
| OXYGEN_SHAKE | CHOCOLATE, EVENING_BREATH, GARLIC, MINT, MORNING_BREATH |
| PANEL | 1X2, 1X4, 2X2, 2X4, 4X4 |
| PEBBLES | L, M, S, XL, XS |
| ROBOT | DISHES, IRONING, LAUNDRY, MOPPING, VACUUMING |
| SLEEP_POD | COTTON, LAMB_WOOL, NYLON, POLYESTER, SUEDE |
| SNACKPACK | CHOCOLATE, PISTACHIO, RASPBERRY, STRAWBERRY, VANILLA |
| TRANSLATOR | ASTRO_BLACK, ECLIPSE_CHARCOAL, GRAPHITE_MIST, SPACE_GRAY, VOID_BLUE |
| UV_VISOR | AMBER, MAGENTA, ORANGE, RED, YELLOW |

Full symbol name is `FAMILY_VARIANT` (e.g. `PEBBLES_L`, `UV_VISOR_RED`).

### Data Files

Historical data lives in `Data_ROUND_5/`:
- `prices_round_5_day_{2,3,4}.csv` — per-tick order book snapshots (semicolon-separated); columns: `day;timestamp;product;bid_price_1;bid_volume_1;...;mid_price;profit_and_loss`
- `trades_round_5_day_{2,3,4}.csv` — executed market trades; columns: `timestamp;buyer;seller;symbol;currency;price;quantity`

## Backtesting

Backtesting is done with **`prosperity4btx`** — do not write custom backtest scripts.

Command format:
```bash
prosperity4btx <trader_file> <round>-<day> --vis 2>&1 | tail -5
```

Example for Round 5, total PnL at once across all days
```bash
prosperity4btx aglos_round5/trader.py 5 --vis 2>&1 | tail -5
```

Example for Round 5, for each day:
```bash
prosperity4btx aglos_round5/trader.py 5-2 --vis 2>&1 | tail -5
prosperity4btx aglos_round5/trader.py 5-3 --vis 2>&1 | tail -5
prosperity4btx aglos_round5/trader.py 5-4 --vis 2>&1 | tail -5
```