# PredictionMarketBench

A benchmark for evaluating prediction market trading agents using real Kalshi market replay data.

## Overview

PredictionMarketBench replays real Kalshi episodes and evaluates trading agent performance under realistic execution constraints. An "episode" is one Kalshi event with multiple tickers/markets.

- **Input**: Recorded orderbook snapshots and trade prints from Kalshi
- **Agent**: Implements a trading policy using provided market data tools
- **Output**: Equity curve, PnL, Sharpe ratio, drawdown, and other metrics

## Installation

```bash
git clone https://github.com/Oddpool/PredictionMarketBench.git
cd PredictionMarketBench

# Install in development mode
pip install -e .

# With plotting support (for charts and GIFs)
pip install -e ".[plotting]"

# With parquet support (faster data loading)
pip install -e ".[parquet]"

# All optional dependencies
pip install -e ".[all]"
```

## Quick Start

```python
from oddpool_bench import Agent, AgentContext, BenchmarkHarness

class MyAgent(Agent):
    def act(self, ctx: AgentContext) -> None:
        markets = ctx.get_markets()
        positions = ctx.get_positions()
        cash = ctx.get_cash()
        
        # Your trading logic here
        # Use ctx.place_order(order) to trade

# Run benchmark
harness = BenchmarkHarness(episodes_dir="./episodes")
result = harness.run(MyAgent())
result.print_summary()

# Save results
result.save("results/summary.json")
result.save_trades("results/trades.json")
result.save_equity_csv("results/equity.csv")
result.save_equity_curve("results/equity.png")
result.save_equity_gif("results/")  # Animated PnL GIFs
```

## Agent Interface

Agents implement the `Agent` base class:

```python
from oddpool_bench import Agent, AgentContext

class MyAgent(Agent):
    def act(self, ctx: AgentContext) -> None:
        """Called periodically to make trading decisions."""
        pass
    
    def on_episode_start(self, metadata: dict) -> None:
        """Called at the start of each episode."""
        pass
    
    def on_episode_end(self, result: dict) -> None:
        """Called at the end of each episode."""
        pass
```

### Context Methods

The `AgentContext` provides these methods:

| Method | Description |
|--------|-------------|
| `get_markets()` | List of all markets with best bid/ask |
| `get_orderbook(ticker, depth)` | Full orderbook for a ticker |
| `get_positions()` | Current positions per ticker |
| `get_cash()` | Cash and equity balances |
| `place_order(order)` | Submit an order |
| `get_resting_orders()` | View resting limit orders |
| `cancel_order(order_id)` | Cancel a resting order |

### Order Types

```python
from oddpool_bench import Order, Side, Action, OrderType, TimeInForce

# Market order (immediate fill or reject)
Order(ticker="TICKER", side=Side.YES, action=Action.BUY,
      order_type=OrderType.MARKET, count=10)

# Limit IOC (fill immediately or cancel)
Order(ticker="TICKER", side=Side.YES, action=Action.BUY,
      order_type=OrderType.LIMIT, count=10, limit_price_cents=50,
      time_in_force=TimeInForce.IOC)

# Limit GTC (fill what crosses, rest the remainder)
Order(ticker="TICKER", side=Side.YES, action=Action.BUY,
      order_type=OrderType.LIMIT, count=10, limit_price_cents=48,
      time_in_force=TimeInForce.GTC)

# Post-only (reject if would cross, always rest as maker)
Order(ticker="TICKER", side=Side.YES, action=Action.BUY,
      order_type=OrderType.LIMIT, count=10, limit_price_cents=45,
      time_in_force=TimeInForce.POST_ONLY)
```

## Included Episodes

| Episode | Event Type | Tickers | Description |
|---------|------------|---------|-------------|
| `KXBTCD-26JAN2017` | Bitcoin Price | 23 | BTC price thresholds at Jan 20, 2026 5PM EST |
| `KXHIGHNY-26JAN20` | Weather | 6 | NYC high temperature on Jan 20, 2026 |
| `KXNFLGAME-26JAN11BUFJAC` | Sports | 2 | NFL Playoffs: Buffalo vs Jacksonville |
| `KXNCAAF-26` | Sports | 2 | College Football: Indiana vs Miami |

## Execution Modes

### Taker-Only
Orders execute immediately against the displayed orderbook. Orders that don't cross are rejected.

### Maker-Taker
Orders can rest in the book and receive fills when historical trades match your price. Queue position is simulated based on displayed size and pro-rata fill allocation.

## Fee Model

Fees follow the Kalshi fee schedule (October 2025):

- **Taker**: 7% of `contracts * P * (1-P)` (rounded up)
- **Maker**: 1.75% of `contracts * P * (1-P)` (rounded up)

Where P is the price as a decimal. Fees are highest at 50 cents and decrease toward extreme prices.

Examples (1 contract):
- At 50c taker: ceil(0.07 * 1 * 0.50 * 0.50 * 100) = 2 cents
- At 50c maker: ceil(0.0175 * 1 * 0.50 * 0.50 * 100) = 1 cent
- At 10c taker: ceil(0.07 * 1 * 0.10 * 0.90 * 100) = 1 cent

## Metrics

Per-episode metrics:
- Total PnL (cents and percentage)
- Max drawdown
- Sharpe ratio
- Contracts traded
- Fees paid
- Maker vs taker fills

## Output Methods

```python
result.save("summary.json")            # Aggregate and per-episode metrics
result.save_trades("trades.json")      # Detailed trade log
result.save_equity_csv("equity.csv")   # Time-series equity data
result.save_equity_curve("equity.png") # Static PnL chart
result.save_equity_gif("output_dir/")  # Animated PnL GIF per episode
```

### Animated GIF Output

Generate animated PnL visualizations:

```python
result.save_equity_gif(
    path="./gifs/",           # Output directory
    duration_seconds=6.0,     # Animation length
    fps=15,                   # Frames per second
)
```

Creates one GIF per episode showing the equity curve over time.

## Configuration

```python
from oddpool_bench import SimulatorConfig, BenchmarkHarness

config = SimulatorConfig(
    agent_call_cadence_seconds=5.0,      # How often to call agent
    equity_sample_interval_seconds=60.0, # Equity curve sampling
    max_tool_calls_per_step=100,         # Tool call budget per step
    verbose=True,
)

harness = BenchmarkHarness(episodes_dir="./episodes", config=config)
```

## Example Agents

See `examples/example_agents.py` for reference implementations:
- `PassiveAgent`: Never trades (baseline)
- `RandomAgent`: Random trades (demonstrates interface)
- `MomentumAgent`: Follows recent price direction

## Episode Data Format

Episodes are stored in `episodes/{episode_id}/`:

```
metadata.json      # Episode configuration
orderbook.parquet  # Market data (or orderbook.csv.gz)
trades.parquet     # Trade prints (or trades.csv.gz)
settlement.json    # Final outcomes
```

## License

MIT

## Links

- Repository: https://github.com/Oddpool/PredictionMarketBench
- Oddpool: https://oddpool.com
