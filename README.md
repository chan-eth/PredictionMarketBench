# Oddpool PredictionMarketBench

A benchmark for evaluating prediction market trading agents using real Kalshi market replay data.

## Overview

PredictionMarketBench replays real Kalshi episodes (an "episode" = one Kalshi event with N tickers/markets) and evaluates an agent's trading performance under realistic execution constraints.

- **Input**: Recorded market data (orderbook snapshots; later trade prints)
- **Agent**: Black-box policy with tool access (place orders, check positions, etc.)
- **Output**: Equity curve + metrics (PnL, Sharpe, drawdown, turnover…)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/FirmTracker/PredictionMarketBench.git
cd PredictionMarketBench

# Install in development mode
pip install -e .

# Optional: Install with parquet support
pip install -e ".[parquet]"
```

### Running Your First Benchmark

```python
from oddpool_bench import BenchmarkHarness, Agent, AgentContext

# Define your agent
class MyAgent(Agent):
    def act(self, ctx: AgentContext) -> None:
        # Get available markets
        markets = ctx.get_markets()
        
        # Check current positions and cash
        positions = ctx.get_positions()
        cash = ctx.get_cash()
        
        # Make trading decisions...
        # Place orders using ctx.place_order(order)
        pass

# Run the benchmark
harness = BenchmarkHarness(episodes_dir="./episodes")
result = harness.run(MyAgent())
result.print_summary()

# Save outputs
result.save("results/summary.json")           # Summary metrics
result.save_trades("results/trades.json")     # All trade details
result.save_equity_csv("results/equity.csv")  # Equity curve data
result.save_equity_curve("results/equity.png")  # PnL chart (requires matplotlib)
```

## Agent Interface

Agents must implement the `Agent` base class:

```python
from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType

class MyAgent(Agent):
    def act(self, ctx: AgentContext) -> None:
        """Called at each agent step to make trading decisions."""
        pass
    
    def on_episode_start(self, metadata: dict) -> None:
        """Optional: Called at the start of each episode."""
        pass
    
    def on_episode_end(self, result: dict) -> None:
        """Optional: Called at the end of each episode."""
        pass
```

### Available Tools (Kalshi-like API)

Within the `act` method, agents have access to these tools via the `AgentContext`:

#### `ctx.get_markets() -> list[MarketInfo]`
Returns all markets with best bid/ask and status.

```python
markets = ctx.get_markets()
for m in markets:
    print(f"{m.ticker}: YES bid={m.yes_best_bid} ask={m.yes_best_ask}")
```

#### `ctx.get_orderbook(ticker, depth=5) -> dict`
Returns full orderbook depth for a ticker.

```python
book = ctx.get_orderbook("KXBTC-T95000")
# book["yes_bids"] = [(price_cents, size), ...]
# book["yes_asks"] = [(price_cents, size), ...]
```

#### `ctx.place_order(order) -> dict`
Place an order. Returns fill information.

```python
from oddpool_bench import Order, Side, Action, OrderType

order = Order(
    ticker="KXBTC-T95000",
    side=Side.YES,
    action=Action.BUY,
    order_type=OrderType.MARKET,
    count=10,
)
result = ctx.place_order(order)
# result["filled_count"], result["average_price"], etc.
```

#### `ctx.get_positions() -> dict`
Returns current positions per ticker.

```python
positions = ctx.get_positions()
# positions["TICKER"] = {"yes_contracts": 5, "no_contracts": 0, ...}
```

#### `ctx.get_cash() -> dict`
Returns cash and equity.

```python
cash = ctx.get_cash()
# cash["cash_cents"], cash["equity_cents"]
```

### Order Types

**v0 (taker-only mode):** Only orders that execute immediately are allowed.

- `OrderType.MARKET`: Always executes as taker
- `OrderType.LIMIT`: Must be "crossing" (price >= ask for buys, price <= bid for sells)

```python
# Market order
Order(ticker="X", side=Side.YES, action=Action.BUY, order_type=OrderType.MARKET, count=10)

# Limit order (must cross to fill in taker-only mode)
Order(ticker="X", side=Side.YES, action=Action.BUY, order_type=OrderType.LIMIT, 
      count=10, limit_price_cents=50)
```

## Episode Data Format

Episodes are stored in folders under `episodes/`:

```
episodes/{episode_id}/
  metadata.json        # Episode configuration
  orderbook.parquet    # or orderbook.csv.gz - market data
  settlement.json      # Outcome for each ticker
```

### metadata.json

```json
{
  "episode_id": "KXBTCD-25DEC3017",
  "event_slug": "KXBTCD-25DEC3017",
  "tickers": ["KXBTCD-25DEC3017-T93249.99", ...],
  "start_ts": "2025-12-30T07:27:02+00:00",
  "end_ts": "2025-12-30T22:00:56+00:00",
  "initial_bankroll_cents": 10000,
  "fee_model_version": "kalshi_oct_2025",
  "execution_mode": "taker_only",
  "observation_depth": 5,
  "description": "Bitcoin price at Dec 30, 2025 5:00 PM EST"
}
```

### settlement.json

```json
{
  "settlements": [
    {"ticker": "KXBTCD-25DEC3017-T93249.99", "result": "YES", "settled_ts": "..."},
    {"ticker": "KXBTCD-25DEC3017-T93749.99", "result": "NO", "settled_ts": "..."}
  ]
}
```

## Included Episodes

The benchmark currently includes 3 episodes:

| Episode | Event | Tickers | Description |
|---------|-------|---------|-------------|
| `KXBTCD-25DEC3017` | Bitcoin Price | 40 | BTC price thresholds at Dec 30 5PM EST |
| `KXHIGHNY-25DEC30` | NYC Temperature | 6 | NYC high temp on Dec 30, 2025 |
| `KXNFLGAME-26JAN04DALNYG` | NFL Game | 2 | Dallas @ NY Giants, Jan 4, 2026 |

## Metrics

The benchmark computes per episode:

- **Total PnL**: Profit/loss in cents and percentage
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return (1-min sampling)
- **Total Contracts Traded**: Volume
- **Total Notional**: Dollar value of trades
- **Total Fees**: Trading costs
- **Slippage**: Cost vs mid price at order time
- **Fill Ratio**: Fraction of requested contracts filled

Aggregate metrics (mean, median) are computed across episodes.

## Output Methods

The `BenchmarkResult` object provides several methods for saving results:

### `result.save(path)` - Summary JSON
Saves aggregate metrics and per-episode summaries.

### `result.save_trades(path)` - Trade Log JSON
Saves detailed information for every trade:
```json
{
  "n_trades": 1560,
  "total_contracts": 1560,
  "total_fees_cents": 1560,
  "trades": [
    {
      "episode_id": "KXBTCD-25DEC3017",
      "timestamp": "2025-12-30T07:40:43+00:00",
      "ticker": "KXBTCD-25DEC3017-T87749.99",
      "side": "yes",
      "action": "buy",
      "order_type": "market",
      "requested_count": 1,
      "filled_count": 1,
      "fills": [{"price_cents": 44, "size": 1, "fee_cents": 1}],
      "total_cost_cents": 45,
      "average_fill_price": 44.0,
      "rejected": false
    }
  ]
}
```

### `result.save_equity_csv(path)` - Equity Curve CSV
Saves time-series equity data:
```csv
episode_id,timestamp,cash_cents,position_value_cents,equity_cents,pnl_cents
KXBTCD-25DEC3017,2025-12-30T07:27:02+00:00,10000,0,10000,0
KXBTCD-25DEC3017,2025-12-30T07:41:01+00:00,9955,40,9995,-5
```

### `result.save_equity_curve(path)` - PnL Chart
Saves a matplotlib chart showing PnL and drawdown over time.
Requires matplotlib: `pip install matplotlib`

```python
result.save_equity_curve("equity.png", figsize=(12, 8), show_episodes=True)
```

## Execution Model

### Orderbook Convention (Kalshi)

Kalshi returns only bids for both YES and NO sides. Asks are derived:
- `yes_asks = flip(no_bids)` where `yes_ask_price = 100 - no_bid_price`
- `no_asks = flip(yes_bids)` where `no_ask_price = 100 - yes_bid_price`

### Fill Mechanics

1. Build executable book (asks for buys, bids for sells)
2. Walk price levels, filling until order complete or liquidity exhausted
3. Partial fills allowed; remainder canceled

### Fee Model (Kalshi Oct 2025)

- **Taker fee**: 2% of potential payout
- **Potential payout** = `min(price, 100-price)` per contract
- Fees rounded up to nearest cent
- No settlement fee

Example: Buy 10 contracts at 50¢
- Potential payout = min(50, 50) = 50¢
- Fee = ceil(50 × 10 × 0.02) = 10¢

### Mark-to-Market (Liquidation-based)

- Long positions valued at best bid
- Short positions valued at best ask
- Conservative approach for robustness

## Configuration

```python
from oddpool_bench import SimulatorConfig, BenchmarkHarness

config = SimulatorConfig(
    agent_call_cadence_seconds=5.0,  # Call agent every 5 seconds
    equity_sample_interval_seconds=60.0,  # Sample equity every minute
    max_tool_calls_per_step=100,  # Tool call budget
    verbose=True,
)

harness = BenchmarkHarness(episodes_dir="./episodes", config=config)
```

## Reproducibility

- **Deterministic replay**: No randomness in fills
- **Fixed tool budgets**: Prevents infinite querying
- **Versioned data**: Fee model and dataset versions tracked

## Adding New Episodes

Use the conversion script to add new data:

```bash
python scripts/convert_raw_data.py new_orderbook_data.csv --output-dir episodes/
```

Update `settlement.json` with actual outcomes when available.

## Future Extensions (v1+)

- **Maker orders**: Queue simulation using trades tape
- **Trades data**: `trades.parquet` with actual trade prints
- **More events**: Sports, politics, economics
- **Multi-event episodes**: Concurrent trading across events

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## License

MIT

## Links

- [Kalshi API Documentation](https://kalshi.com/api)
- [Kalshi Fee Schedule](https://kalshi.com/fees)
