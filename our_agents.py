"""
Our trading strategies for backtesting against Kalshi historical data.

Strategies:
1. PremiumHarvester - Buy NO on high-probability markets (favourite-longshot bias)
2. MomentumSniper - Detect momentum and trade direction in 15-min crypto markets
3. CombinedAgent - All strategies running together with portfolio management
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType, TimeInForce


# Favourite-longshot bias table: YES price range → actual win rate
# From academic study of 300,000+ Kalshi contracts
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


def kelly_size(
    model_prob: float,
    price_cents: int,
    bankroll_cents: int,
    kelly_mult: float = 0.25,
) -> int:
    """Fractional Kelly sizing — exact Python port of crates/trading-engine/src/kelly.rs.

    model_prob: probability the contract wins (e.g. 1 - yes_prob for a NO contract)
    price_cents: cost of one contract in cents
    bankroll_cents: total bankroll in cents
    kelly_mult: fraction of full Kelly to bet (0.25 = quarter-Kelly)
    Returns: number of contracts to buy (0 if no edge)
    """
    if price_cents <= 0 or price_cents >= 100 or bankroll_cents <= 0:
        return 0
    p = max(0.001, min(0.999, model_prob))
    q = 1.0 - p
    price = float(price_cents)
    b = (100.0 - price) / price  # net odds
    f_star = (p * b - q) / b
    if f_star <= 0.0:
        return 0
    bet_cents = f_star * kelly_mult * bankroll_cents
    return int(bet_cents / price)


def actual_yes_probability(yes_price_cents: int) -> float:
    """Look up actual YES win probability given market price."""
    for min_p, max_p, actual in BIAS_TABLE:
        if min_p <= yes_price_cents <= max_p:
            return actual
    return yes_price_cents / 100.0


class PremiumHarvester(Agent):
    """
    Buys NO on markets where YES is overpriced (favourite-longshot bias).

    Strategy:
    - Scan all markets for YES contracts priced 1-7¢ (NO at 93-99¢)
    - The bias says these YES contracts win LESS than their price implies
    - Buy NO, collect premium when market resolves NO
    - Diversify across many uncorrelated markets
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
        positions = ctx.get_positions()
        cash = ctx.get_cash()

        if len(self.positions) >= self.max_total_positions:
            return

        for market in markets:
            ticker = market.ticker

            # Skip if already positioned
            if ticker in self.positions:
                continue

            # Need quotes
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
            no_price = 100 - yes_mid

            # Filter: only high-NO-price markets
            if no_price < self.min_no_price or no_price > self.max_no_price:
                continue

            # Calculate edge using bias table
            yes_actual = actual_yes_probability(yes_mid)
            no_actual = 1.0 - yes_actual
            no_implied = no_price / 100.0
            edge = no_actual - no_implied

            if edge < self.min_edge:
                continue

            # Check we have cash
            cost = market.no_best_ask or (100 - market.yes_best_bid)
            if cost is None:
                continue

            contracts = min(
                self.max_position_per_market,
                cash["cash_cents"] // (cost * 2),  # keep 50% cash reserve
            )

            if contracts < 1:
                continue

            # Buy NO
            order = Order(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                order_type=OrderType.MARKET,
                count=contracts,
            )
            result = ctx.place_order(order)
            if result and result.get("filled", 0) > 0:
                self.positions[ticker] = result["filled"]


class PremiumHarvesterKelly(Agent):
    """
    PremiumHarvester with Kelly-criterion sizing instead of fixed contracts.

    Mirrors today's Rust engine change: scanner.rs line 279.
    Sizing: kelly_size(1 - yes_actual, no_price, bankroll, kelly_fraction)
    capped at max_position_per_market.
    """

    def __init__(
        self,
        min_no_price: int = 93,
        max_no_price: int = 99,
        min_edge: float = 0.005,
        max_position_per_market: int = 10,
        max_total_positions: int = 20,
        bankroll_cents: int = 50_000,   # $500 — matches BANKROLL_CENTS default
        kelly_fraction: float = 0.25,   # quarter-Kelly — matches KELLY_FRACTION default
    ):
        self.min_no_price = min_no_price
        self.max_no_price = max_no_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.bankroll_cents = bankroll_cents
        self.kelly_fraction = kelly_fraction
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

            # Kelly sizing: model_prob for NO = no_actual
            kelly_qty = kelly_size(
                model_prob=no_actual,
                price_cents=cost,
                bankroll_cents=self.bankroll_cents,
                kelly_mult=self.kelly_fraction,
            )
            contracts = max(1, kelly_qty)
            contracts = min(contracts, self.max_position_per_market)

            # Cash safety: never spend more than available
            if cost * contracts > cash["cash_cents"]:
                contracts = cash["cash_cents"] // cost
            if contracts < 1:
                continue

            order = Order(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                order_type=OrderType.MARKET,
                count=contracts,
            )
            result = ctx.place_order(order)
            if result and result.get("filled", 0) > 0:
                self.positions[ticker] = result["filled"]


class MomentumSniper(Agent):
    """
    Detects short-term momentum and trades ONE direction per market.

    V2 fixes:
    - Only trade ONE side per ticker (no self-canceling YES+NO)
    - Focus on the STRONGEST momentum ticker, not all of them
    - Proper exit logic: sell when momentum fades or reverses
    - Profit target and stop loss per position
    - Only enter when spread is tight (liquid market)
    """

    def __init__(
        self,
        lookback: int = 15,
        entry_threshold_cents: int = 4,
        profit_target_cents: int = 3,
        stop_loss_cents: int = 5,
        max_position: int = 5,
        max_simultaneous: int = 3,
        min_spread: int = 6,  # max spread to enter (tighter = more liquid)
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold_cents
        self.profit_target = profit_target_cents
        self.stop_loss = stop_loss_cents
        self.max_position = max_position
        self.max_simultaneous = max_simultaneous
        self.min_spread = min_spread
        self.price_history: dict[str, list[int]] = {}
        self.positions: dict[str, dict] = {}  # ticker → {side, entry_price, contracts}

    def on_episode_start(self, metadata: dict) -> None:
        self.price_history = {}
        self.positions = {}

    def act(self, ctx: AgentContext) -> None:
        markets = ctx.get_markets()
        raw_positions = ctx.get_positions()
        cash = ctx.get_cash()

        # Phase 1: Update prices and check exits
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            ticker = market.ticker
            mid = (market.yes_best_bid + market.yes_best_ask) // 2

            # Update history
            if ticker not in self.price_history:
                self.price_history[ticker] = []
            self.price_history[ticker].append(mid)
            if len(self.price_history[ticker]) > self.lookback * 3:
                self.price_history[ticker] = self.price_history[ticker][-self.lookback * 2:]

            # Check exits for open positions
            if ticker in self.positions:
                pos_info = self.positions[ticker]
                entry = pos_info["entry_price"]
                side = pos_info["side"]
                contracts = pos_info["contracts"]

                should_exit = False
                reason = ""

                if side == "yes":
                    pnl_cents = mid - entry
                    if pnl_cents >= self.profit_target:
                        should_exit = True
                        reason = "profit target"
                    elif pnl_cents <= -self.stop_loss:
                        should_exit = True
                        reason = "stop loss"
                    # Momentum reversal: price dropping
                    elif len(self.price_history[ticker]) >= 5:
                        recent = self.price_history[ticker][-5:]
                        if recent[-1] < recent[0] - 2:
                            should_exit = True
                            reason = "momentum reversal"

                elif side == "no":
                    # For NO: we profit when YES price drops
                    pnl_cents = entry - mid
                    if pnl_cents >= self.profit_target:
                        should_exit = True
                        reason = "profit target"
                    elif pnl_cents <= -self.stop_loss:
                        should_exit = True
                        reason = "stop loss"
                    elif len(self.price_history[ticker]) >= 5:
                        recent = self.price_history[ticker][-5:]
                        if recent[-1] > recent[0] + 2:
                            should_exit = True
                            reason = "momentum reversal"

                if should_exit:
                    sell_side = Side.YES if side == "yes" else Side.NO
                    actual_pos = raw_positions.get(ticker, {})
                    actual_count = actual_pos.get(f"{side}_contracts", 0)
                    if actual_count > 0:
                        order = Order(
                            ticker=ticker,
                            side=sell_side,
                            action=Action.SELL,
                            order_type=OrderType.MARKET,
                            count=actual_count,
                        )
                        ctx.place_order(order)
                    del self.positions[ticker]

        # Phase 2: Find best entry opportunity
        if len(self.positions) >= self.max_simultaneous:
            return

        best_signal = None
        best_strength = 0

        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            ticker = market.ticker
            if ticker in self.positions:
                continue

            # Spread filter — only trade liquid markets
            spread = market.yes_best_ask - market.yes_best_bid
            if spread > self.min_spread:
                continue

            if ticker not in self.price_history:
                continue
            if len(self.price_history[ticker]) < self.lookback:
                continue

            history = self.price_history[ticker]
            mid = history[-1]
            momentum = mid - history[-self.lookback]
            abs_momentum = abs(momentum)

            # Only consider strong signals
            if abs_momentum < self.entry_threshold:
                continue

            # Skip extreme prices (near 0 or 100) — not enough room to profit
            if mid < 15 or mid > 85:
                continue

            if abs_momentum > best_strength:
                best_strength = abs_momentum
                best_signal = {
                    "ticker": ticker,
                    "momentum": momentum,
                    "mid": mid,
                    "market": market,
                }

        # Enter the single best opportunity
        if best_signal is None:
            return

        ticker = best_signal["ticker"]
        momentum = best_signal["momentum"]
        market = best_signal["market"]
        mid = best_signal["mid"]

        if momentum > 0:
            # Upward momentum → buy YES
            cost = market.yes_best_ask
            if cost and cash["cash_cents"] >= cost * self.max_position * 2:
                contracts = min(self.max_position, cash["cash_cents"] // (cost * 3))
                if contracts >= 1:
                    order = Order(
                        ticker=ticker,
                        side=Side.YES,
                        action=Action.BUY,
                        order_type=OrderType.MARKET,
                        count=contracts,
                    )
                    result = ctx.place_order(order)
                    if result and result.get("filled", 0) > 0:
                        self.positions[ticker] = {
                            "side": "yes",
                            "entry_price": mid,
                            "contracts": result["filled"],
                        }
        else:
            # Downward momentum → buy NO
            no_ask = 100 - market.yes_best_bid if market.yes_best_bid else None
            if no_ask and cash["cash_cents"] >= no_ask * self.max_position * 2:
                contracts = min(self.max_position, cash["cash_cents"] // (no_ask * 3))
                if contracts >= 1:
                    order = Order(
                        ticker=ticker,
                        side=Side.NO,
                        action=Action.BUY,
                        order_type=OrderType.MARKET,
                        count=contracts,
                    )
                    result = ctx.place_order(order)
                    if result and result.get("filled", 0) > 0:
                        self.positions[ticker] = {
                            "side": "no",
                            "entry_price": mid,
                            "contracts": result["filled"],
                        }


class CombinedAgent(Agent):
    """
    Runs multiple strategies together with shared risk management.

    - PremiumHarvester for steady income
    - MomentumSniper for directional trades
    - Shared capital pool with allocation limits
    """

    def __init__(self):
        self.harvester = PremiumHarvester(
            max_position_per_market=5,
            max_total_positions=15,
        )
        self.momentum = MomentumSniper(
            lookback=15,
            entry_threshold_cents=3,
            max_position=3,
        )
        self._step = 0

    def on_episode_start(self, metadata: dict) -> None:
        self.harvester.on_episode_start(metadata)
        self.momentum.on_episode_start(metadata)
        self._step = 0

    def act(self, ctx: AgentContext) -> None:
        self._step += 1

        # Run harvester every 5 steps (less frequent, passive strategy)
        if self._step % 5 == 0:
            self.harvester.act(ctx)

        # Run momentum every step (needs to be responsive)
        self.momentum.act(ctx)
