"""
Post-backtest overhaul strategies.

Changes vs original our_agents.py:
1. PremiumHarvesterV2 — maker-only (POST_ONLY), 14-row FLB, YES 1-20c, time-decay bonus
2. PremiumHarvesterOld — original taker version (for A/B comparison)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType, TimeInForce


# Full 14-row favourite-longshot bias table
BIAS_TABLE = [
    (1, 5, 0.01),
    (6, 10, 0.04),
    (11, 15, 0.08),
    (16, 20, 0.13),
    (21, 30, 0.22),
    (31, 40, 0.33),
    (41, 50, 0.45),
    (51, 60, 0.57),
    (61, 70, 0.68),
    (71, 80, 0.78),
    (81, 85, 0.84),
    (86, 90, 0.91),
    (91, 95, 0.96),
    (96, 99, 0.99),
]


def actual_yes_probability(yes_price_cents: int) -> float:
    for min_p, max_p, actual in BIAS_TABLE:
        if min_p <= yes_price_cents <= max_p:
            return actual
    return yes_price_cents / 100.0


def time_decay_bonus(hours_to_expiry: float) -> float:
    if hours_to_expiry < 2.0:
        return 0.02
    elif hours_to_expiry < 6.0:
        return 0.01
    elif hours_to_expiry < 24.0:
        return 0.005
    else:
        return 0.0


class PremiumHarvesterV2(Agent):
    """
    Overhauled PremiumHarvester:
    - POST_ONLY maker orders (1.75% fee instead of 7% taker)
    - Full 14-row FLB table (was 4-row)
    - YES 1-20c range (was 1-7c)
    - Time-decay bonus for near-expiry markets
    - Maker-side pricing: join NO bid + 1c (never cross spread)
    - Skip if spread < 3c (too tight for maker)
    - Fill-rate awareness: skip thin markets, price aggressively in moderate ones
    - Stale order cancellation
    """

    def __init__(
        self,
        min_yes_price: int = 1,
        max_yes_price: int = 20,
        min_edge: float = 0.005,
        max_position_per_market: int = 10,
        max_total_positions: int = 20,
        min_spread: int = 3,
        fill_rate_window: int = 5,
        min_fill_rate: float = 0.30,
        stale_order_steps: int = 60,
    ):
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.min_spread = min_spread
        self.fill_rate_window = fill_rate_window
        self.min_fill_rate = min_fill_rate
        self.stale_order_steps = stale_order_steps
        self.positions: dict[str, int] = {}
        # Fill-rate tracking: ticker → list of (placed_step, was_filled)
        self._fill_history: dict[str, list[tuple[int, bool]]] = {}
        # Track when orders were placed: order_id → placed_step
        self._order_placed_step: dict[str, int] = {}
        self._step = 0

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}
        self._fill_history = {}
        self._order_placed_step = {}
        self._step = 0

    def _get_fill_rate(self, ticker: str) -> float | None:
        """Return fill rate for ticker over last N attempts, or None if insufficient data."""
        history = self._fill_history.get(ticker, [])
        recent = history[-self.fill_rate_window:]
        if len(recent) < 2:
            return None
        return sum(1 for _, filled in recent if filled) / len(recent)

    def _record_attempt(self, ticker: str, filled: bool):
        if ticker not in self._fill_history:
            self._fill_history[ticker] = []
        self._fill_history[ticker].append((self._step, filled))

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        # Cancel stale resting orders
        resting = ctx.get_resting_orders()
        for ro in resting:
            order_id = ro["order_id"]
            placed_step = self._order_placed_step.get(order_id, self._step)
            age = self._step - placed_step

            if ro["remaining_count"] == ro["original_count"]:
                # Unfilled order — cancel if too old or deep in queue
                if age > self.stale_order_steps or ro["env_ahead"] > 50:
                    ctx.cancel_order(order_id)
                    self._order_placed_step.pop(order_id, None)
                    self._record_attempt(ro["ticker"], filled=False)

        if len(self.positions) >= self.max_total_positions:
            return

        for market in markets:
            ticker = market.ticker
            if ticker in self.positions:
                continue
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2

            # Widened range: YES 1-20c
            if yes_mid < self.min_yes_price or yes_mid > self.max_yes_price:
                continue

            no_price = 100 - yes_mid

            # Spread check
            yes_spread = market.yes_best_ask - market.yes_best_bid
            if yes_spread < self.min_spread:
                continue

            # Fill-rate gate: skip consistently thin markets
            fill_rate = self._get_fill_rate(ticker)
            if fill_rate is not None and fill_rate < self.min_fill_rate:
                continue

            # Edge from 14-row FLB table
            yes_actual = actual_yes_probability(yes_mid)
            no_actual = 1.0 - yes_actual
            no_implied = no_price / 100.0
            base_edge = no_actual - no_implied

            # Time-decay bonus
            hours = (market.time_to_close_seconds or 999999) / 3600.0
            decay = time_decay_bonus(hours)
            edge = base_edge + decay

            if edge < self.min_edge:
                continue

            # Maker-side pricing: NO bid + 1c
            # NO best bid = 100 - YES best ask
            no_bid = 100 - market.yes_best_ask
            maker_price = no_bid + 1

            # Moderate fill-rate → price 1c more aggressively (closer to mid)
            if fill_rate is not None and fill_rate < 0.70:
                maker_price = no_bid + 2  # 1c closer to mid for better fills

            # NO best ask = 100 - YES best bid
            no_ask = 100 - market.yes_best_bid
            if maker_price >= no_ask:
                continue  # would cross

            # Size
            contracts = min(
                self.max_position_per_market,
                cash["cash_cents"] // (maker_price * 2),
            )
            if contracts < 1:
                continue

            # POST_ONLY limit order — maker only, 1.75% fee
            order = Order(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                order_type=OrderType.LIMIT,
                count=contracts,
                limit_price_cents=maker_price,
                time_in_force=TimeInForce.POST_ONLY,
            )
            result = ctx.place_order(order)

            # POST_ONLY either rests or gets rejected (never crosses)
            if result and not result.get("rejected", False):
                # May have resting order or immediate fill via maker queue
                resting_order = result.get("resting")
                fill = result.get("fill")
                filled = fill["count"] if fill else 0
                if resting_order:
                    oid = resting_order.get("order_id") or resting_order.get("id", "")
                    if oid:
                        self._order_placed_step[oid] = self._step
                if resting_order or filled > 0:
                    self.positions[ticker] = filled if filled > 0 else contracts
                    if filled > 0:
                        self._record_attempt(ticker, filled=True)
            else:
                self._record_attempt(ticker, filled=False)


class PremiumHarvesterOld(Agent):
    """
    Original PremiumHarvester (taker, YES 1-7c) for A/B comparison.
    """

    def __init__(
        self,
        min_no_price: int = 93,
        max_no_price: int = 99,
        min_edge: float = 0.005,
        max_position_per_market: int = 10,
        max_total_positions: int = 20,
    ):
        self.min_no_price = min_no_price
        self.max_no_price = max_no_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.positions: dict[str, int] = {}

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}

    def act(self, ctx: AgentContext) -> None:
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        if len(self.positions) >= self.max_total_positions:
            return

        for market in markets:
            ticker = market.ticker
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

            contracts = min(
                self.max_position_per_market,
                cash["cash_cents"] // (cost * 2),
            )
            if contracts < 1:
                continue

            # MARKET order — taker, 7% fee
            order = Order(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                order_type=OrderType.MARKET,
                count=contracts,
            )
            result = ctx.place_order(order)
            if result:
                fill = result.get("fill")
                if fill and fill.get("count", 0) > 0:
                    self.positions[ticker] = fill["count"]
