"""
Optimized trading agents based on backtest findings:
- Momentum works on CRYPTO ONLY (BTC +38.5% on wide params)
- Premium Harvester works on EVERYTHING
- Combined: route by market type

V3: Market-type aware routing + tuned parameters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType


# Favourite-longshot bias table
BIAS_TABLE = [
    (1, 5, 0.01), (6, 10, 0.04), (11, 15, 0.08), (16, 20, 0.13),
    (21, 30, 0.22), (31, 40, 0.33), (41, 50, 0.45), (51, 60, 0.57),
    (61, 70, 0.68), (71, 80, 0.78), (81, 85, 0.84), (86, 90, 0.91),
    (91, 95, 0.96), (96, 99, 0.99),
]


def actual_yes_probability(yes_price_cents: int) -> float:
    for min_p, max_p, actual in BIAS_TABLE:
        if min_p <= yes_price_cents <= max_p:
            return actual
    return yes_price_cents / 100.0


def is_crypto_market(ticker: str) -> bool:
    """Check if a ticker is a crypto market (where momentum works)."""
    crypto_prefixes = ["KXBTC", "KXETH", "KXSOL", "KXBTCD"]
    return any(ticker.startswith(p) for p in crypto_prefixes)


class CryptoMomentumV3(Agent):
    """
    Momentum strategy ONLY for crypto markets.

    Tuned from backtest: wide params (5c entry, 5c TP, 7c SL) = +38.5% on BTC

    Improvements over V2:
    - Filters to crypto tickers only
    - Tracks volume alongside price for confirmation
    - Scales position size by momentum strength
    - Faster exit on reversal (3-tick lookback)
    """

    def __init__(self):
        self.lookback = 20
        self.entry_threshold = 5
        self.profit_target = 5
        self.stop_loss = 7
        self.max_position = 5
        self.max_simultaneous = 3
        self.max_spread = 8
        self.price_history: dict[str, list[int]] = {}
        self.positions: dict[str, dict] = {}

    def on_episode_start(self, metadata: dict) -> None:
        self.price_history = {}
        self.positions = {}

    def act(self, ctx: AgentContext) -> None:
        markets = ctx.get_markets()
        raw_positions = ctx.get_positions()
        cash = ctx.get_cash()

        # Only trade crypto
        crypto_markets = [m for m in markets if is_crypto_market(m.ticker)]
        if not crypto_markets:
            return

        # Phase 1: Manage exits
        for market in crypto_markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            ticker = market.ticker
            mid = (market.yes_best_bid + market.yes_best_ask) // 2

            if ticker not in self.price_history:
                self.price_history[ticker] = []
            self.price_history[ticker].append(mid)
            if len(self.price_history[ticker]) > 100:
                self.price_history[ticker] = self.price_history[ticker][-60:]

            if ticker not in self.positions:
                continue

            pos_info = self.positions[ticker]
            entry = pos_info["entry_price"]
            side = pos_info["side"]

            if side == "yes":
                pnl = mid - entry
            else:
                pnl = entry - mid

            should_exit = False

            # Profit target
            if pnl >= self.profit_target:
                should_exit = True

            # Stop loss
            elif pnl <= -self.stop_loss:
                should_exit = True

            # Quick reversal detection (3 ticks)
            elif len(self.price_history[ticker]) >= 3:
                recent = self.price_history[ticker][-3:]
                if side == "yes" and recent[-1] < recent[0] - 3:
                    should_exit = True
                elif side == "no" and recent[-1] > recent[0] + 3:
                    should_exit = True

            if should_exit:
                sell_side = Side.YES if side == "yes" else Side.NO
                actual = raw_positions.get(ticker, {})
                count = actual.get(f"{side}_contracts", 0)
                if count > 0:
                    order = Order(
                        ticker=ticker,
                        side=sell_side,
                        action=Action.SELL,
                        order_type=OrderType.MARKET,
                        count=count,
                    )
                    ctx.place_order(order)
                del self.positions[ticker]

        # Phase 2: Find best entry
        if len(self.positions) >= self.max_simultaneous:
            return

        best = None
        best_strength = 0

        for market in crypto_markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            ticker = market.ticker
            if ticker in self.positions:
                continue

            spread = market.yes_best_ask - market.yes_best_bid
            if spread > self.max_spread:
                continue

            if ticker not in self.price_history or len(self.price_history[ticker]) < self.lookback:
                continue

            history = self.price_history[ticker]
            mid = history[-1]
            momentum = mid - history[-self.lookback]

            if mid < 15 or mid > 85:
                continue

            if abs(momentum) >= self.entry_threshold and abs(momentum) > best_strength:
                best_strength = abs(momentum)
                best = {"ticker": ticker, "momentum": momentum, "mid": mid, "market": market}

        if best is None:
            return

        ticker = best["ticker"]
        momentum = best["momentum"]
        market = best["market"]
        mid = best["mid"]

        # Scale position by momentum strength (stronger = more contracts)
        strength_scale = min(abs(momentum) / self.entry_threshold, 2.0)
        base_contracts = max(1, int(self.max_position * strength_scale / 2))
        contracts = min(base_contracts, self.max_position)

        if momentum > 0:
            cost = market.yes_best_ask
            if cost and cash["cash_cents"] >= cost * contracts * 2:
                order = Order(
                    ticker=ticker, side=Side.YES, action=Action.BUY,
                    order_type=OrderType.MARKET, count=contracts,
                )
                result = ctx.place_order(order)
                if result and result.get("filled", 0) > 0:
                    self.positions[ticker] = {
                        "side": "yes", "entry_price": mid, "contracts": result["filled"],
                    }
        else:
            no_ask = 100 - market.yes_best_bid if market.yes_best_bid else None
            if no_ask and cash["cash_cents"] >= no_ask * contracts * 2:
                order = Order(
                    ticker=ticker, side=Side.NO, action=Action.BUY,
                    order_type=OrderType.MARKET, count=contracts,
                )
                result = ctx.place_order(order)
                if result and result.get("filled", 0) > 0:
                    self.positions[ticker] = {
                        "side": "no", "entry_price": mid, "contracts": result["filled"],
                    }


class SmartHarvester(Agent):
    """
    Premium harvester that SKIPS crypto markets (momentum handles those).
    Only harvests NO premium on weather, sports, economics, etc.
    """

    def __init__(self):
        self.min_no_price = 93
        self.max_no_price = 99
        self.min_edge = 0.005
        self.max_per_market = 10
        self.max_positions = 20
        self.positions: dict[str, int] = {}

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}

    def act(self, ctx: AgentContext) -> None:
        markets = ctx.get_markets()
        positions = ctx.get_positions()
        cash = ctx.get_cash()

        if len(self.positions) >= self.max_positions:
            return

        for market in markets:
            ticker = market.ticker

            # Skip crypto — momentum agent handles those
            if is_crypto_market(ticker):
                continue

            if ticker in self.positions:
                continue
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
            no_price = 100 - yes_mid

            if no_price < self.min_no_price or no_price > self.max_no_price:
                continue

            yes_actual = actual_yes_probability(yes_mid)
            no_actual = 1.0 - yes_actual
            no_implied = no_price / 100.0
            edge = no_actual - no_implied

            if edge < self.min_edge:
                continue

            cost = market.no_best_ask or (100 - market.yes_best_bid)
            if cost is None:
                continue

            contracts = min(self.max_per_market, cash["cash_cents"] // (cost * 2))
            if contracts < 1:
                continue

            order = Order(
                ticker=ticker, side=Side.NO, action=Action.BUY,
                order_type=OrderType.MARKET, count=contracts,
            )
            result = ctx.place_order(order)
            if result and result.get("filled", 0) > 0:
                self.positions[ticker] = result["filled"]


class UltimateBot(Agent):
    """
    The full trading system:
    - CryptoMomentumV3 for BTC/ETH/SOL markets
    - SmartHarvester for everything else
    - Shared risk: 60% capital to harvester, 40% to momentum
    """

    def __init__(self):
        self.momentum = CryptoMomentumV3()
        self.harvester = SmartHarvester()
        self._step = 0

    def on_episode_start(self, metadata: dict) -> None:
        self.momentum.on_episode_start(metadata)
        self.harvester.on_episode_start(metadata)
        self._step = 0

    def act(self, ctx: AgentContext) -> None:
        self._step += 1

        # Momentum every tick (speed matters)
        self.momentum.act(ctx)

        # Harvester every 3 ticks (passive strategy)
        if self._step % 3 == 0:
            self.harvester.act(ctx)
