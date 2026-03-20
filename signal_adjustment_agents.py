"""
Signal-as-adjustment agents for backtesting the retired-trigger architecture.

Instead of signals generating independent orders (proven unprofitable),
signals write probability adjustments that PremiumHarvester reads to
refine its FLB edge estimate.

Agents:
1. PHWithSignalAdjustment — PH V2 + signal-derived probability deltas
2. PHBaseline             — PH V2 standalone (control, no signal input)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType, TimeInForce
from signal_agents import TradeTapeAnalyzer, TapeRecord, SignalScorer
from overhauled_agents import actual_yes_probability, time_decay_bonus


class PHWithSignalAdjustment(Agent):
    """
    PremiumHarvester V2 with signal-derived probability adjustments.

    Mirrors the production Rust change:
    - Tape + momentum signals flow into SignalScorer
    - Scored signals compute a probability delta (NOT an order)
    - Delta is applied to FLB actual_yes_prob before edge calculation
    - Same trade count as PH standalone, but better probability estimates

    Key formula:
        delta = direction_sign * signal_adjustment_max * scale
        adjusted_yes_prob = clamp(actual_yes_prob + delta, 0.001, 0.999)
        edge = (1 - adjusted_yes_prob) - (no_price / 100)
    """

    def __init__(
        self,
        # PH V2 params
        min_yes_price: int = 1,
        max_yes_price: int = 20,
        min_edge: float = 0.005,
        max_position_per_market: int = 10,
        max_total_positions: int = 20,
        min_spread: int = 3,
        stale_order_steps: int = 60,
        # Signal adjustment params
        signal_adjustment_max: float = 0.03,
        scorer_min_composite: float = 0.5,
        confluence_min_sources: int = 2,
        # Momentum signal params
        momentum_lookback: int = 15,
        momentum_threshold_cents: int = 3,
    ):
        # PH params
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.min_spread = min_spread
        self.stale_order_steps = stale_order_steps

        # Signal params
        self.signal_adjustment_max = signal_adjustment_max
        self.scorer_min_composite = scorer_min_composite
        self.momentum_lookback = momentum_lookback
        self.momentum_threshold = momentum_threshold_cents

        # State
        self.positions: dict[str, int] = {}
        self._order_placed_step: dict[str, int] = {}
        self._step = 0

        # Signal infrastructure
        self.tape = TradeTapeAnalyzer()
        self.scorer = SignalScorer(
            min_composite=scorer_min_composite,
            confluence_min_sources=confluence_min_sources,
        )
        self._prev_book: dict[str, tuple[int, int]] = {}
        self._price_history: dict[str, list[int]] = {}
        # Shared adjustment map: ticker → delta (mirrors Rust DashMap)
        self._adjustments: dict[str, dict] = {}

        # Diagnostics
        self.adjustment_log: list[dict] = []
        self.trade_log: list[dict] = []

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}
        self._order_placed_step = {}
        self._step = 0
        self.tape = TradeTapeAnalyzer()
        self.scorer = SignalScorer(
            min_composite=self.scorer_min_composite,
            confluence_min_sources=2,
        )
        self._prev_book = {}
        self._price_history = {}
        self._adjustments = {}
        self.adjustment_log = []
        self.trade_log = []

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        # ---- Phase 1: Generate signals (same as SignalFusionAgent) ----
        self._generate_signals(ctx, markets)

        # ---- Phase 2: Score signals → probability adjustments (NOT orders) ----
        self._score_to_adjustments()

        # ---- Phase 3: Prune stale adjustments (>60 steps old) ----
        stale_keys = [
            k for k, v in self._adjustments.items()
            if self._step - v["step"] > 60
        ]
        for k in stale_keys:
            del self._adjustments[k]

        # ---- Phase 4: Cancel stale resting orders ----
        resting = ctx.get_resting_orders()
        for ro in resting:
            order_id = ro["order_id"]
            placed_step = self._order_placed_step.get(order_id, self._step)
            age = self._step - placed_step
            if ro["remaining_count"] == ro["original_count"]:
                if age > self.stale_order_steps or ro["env_ahead"] > 50:
                    ctx.cancel_order(order_id)
                    self._order_placed_step.pop(order_id, None)

        # ---- Phase 5: Premium Harvester with adjusted probabilities ----
        if len(self.positions) >= self.max_total_positions:
            return

        for market in markets:
            ticker = market.ticker
            if ticker in self.positions:
                continue
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
            if yes_mid < self.min_yes_price or yes_mid > self.max_yes_price:
                continue

            no_price = 100 - yes_mid

            # Spread check
            yes_spread = market.yes_best_ask - market.yes_best_bid
            if yes_spread < self.min_spread:
                continue

            # FLB lookup
            yes_actual = actual_yes_probability(yes_mid)

            # Apply signal adjustment (the key integration point)
            adjustment = 0.0
            adj_entry = self._adjustments.get(ticker)
            if adj_entry and self._step - adj_entry["step"] < 60:
                adjustment = adj_entry["delta"]

            adjusted_yes_prob = max(0.001, min(0.999, yes_actual + adjustment))

            # Edge calculation with adjusted probability
            no_actual = 1.0 - adjusted_yes_prob
            no_implied = no_price / 100.0
            base_edge = no_actual - no_implied

            # Time-decay bonus
            hours = (market.time_to_close_seconds or 999999) / 3600.0
            decay = time_decay_bonus(hours)
            edge = base_edge + decay

            if edge < self.min_edge:
                continue

            # Maker-side pricing
            no_bid = 100 - market.yes_best_ask
            maker_price = no_bid + 1
            no_ask = 100 - market.yes_best_bid
            if maker_price >= no_ask:
                continue

            contracts = min(
                self.max_position_per_market,
                cash["cash_cents"] // (maker_price * 2),
            )
            if contracts < 1:
                continue

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

            if result and not result.get("rejected", False):
                resting_order = result.get("resting")
                fill = result.get("fill")
                filled = fill["count"] if fill else 0
                if resting_order:
                    oid = resting_order.get("order_id") or resting_order.get("id", "")
                    if oid:
                        self._order_placed_step[oid] = self._step
                if resting_order or filled > 0:
                    self.positions[ticker] = filled if filled > 0 else contracts
                    self.trade_log.append({
                        "step": self._step, "ticker": ticker,
                        "edge": round(edge, 4),
                        "adjustment": round(adjustment, 4),
                        "yes_actual": round(yes_actual, 4),
                        "adjusted": round(adjusted_yes_prob, 4),
                    })

    def _generate_signals(self, ctx: AgentContext, markets):
        """Generate signals from momentum + tape (mirrors Rust signal sources)."""
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            ticker = market.ticker
            mid = (market.yes_best_bid + market.yes_best_ask) // 2

            # Momentum signals
            if ticker not in self._price_history:
                self._price_history[ticker] = []
            self._price_history[ticker].append(mid)
            hist = self._price_history[ticker]
            if len(hist) > self.momentum_lookback * 3:
                self._price_history[ticker] = hist[-self.momentum_lookback * 2:]
                hist = self._price_history[ticker]

            if len(hist) >= self.momentum_lookback:
                momentum = hist[-1] - hist[-self.momentum_lookback]
                if abs(momentum) >= self.momentum_threshold:
                    direction = "bullish" if momentum > 0 else "bearish"
                    conf = min(abs(momentum) / 10.0, 1.0)
                    self.scorer.push(ticker, "crypto_momentum", direction, conf, self._step)

            # Tape signals (orderbook delta)
            book = ctx.get_orderbook(ticker)
            if book:
                bid_vol = sum(level[1] for level in book.get("yes_bids", [])[:3])
                ask_vol = sum(level[1] for level in book.get("yes_asks", [])[:3])

                prev = self._prev_book.get(ticker)
                self._prev_book[ticker] = (bid_vol, ask_vol)

                if prev is not None:
                    prev_bid, prev_ask = prev
                    bid_consumed = max(0, prev_bid - bid_vol)
                    ask_consumed = max(0, prev_ask - ask_vol)
                    bid_pct = bid_consumed / max(prev_bid, 1)
                    ask_pct = ask_consumed / max(prev_ask, 1)
                    delta = ask_pct - bid_pct
                    total_consumed = bid_consumed + ask_consumed

                    if total_consumed > 2 and abs(delta) > 0.05:
                        taker_side = "yes" if delta > 0 else "no"
                        tape_trades = [TapeRecord(
                            count=total_consumed, price=mid,
                            taker_side=taker_side, step=self._step,
                        )]
                        tape_sigs = self.tape.ingest(ticker, tape_trades, self._step)
                        for ts in tape_sigs:
                            self.scorer.push(
                                ticker, "trade_tape",
                                ts["direction"], ts["confidence"], self._step,
                            )

    def _score_to_adjustments(self):
        """Score signals and write probability adjustments (NOT orders)."""
        scored = self.scorer.score(self._step)

        for sig in scored:
            # Mirror Rust: direction_sign * adjustment_max * scale
            if sig["direction"] == "bullish":
                direction_sign = 1.0   # YES more likely → shrink NO edge
            else:
                direction_sign = -1.0  # YES less likely → grow NO edge

            scale = min(sig["composite"] / self.scorer_min_composite, 1.0)
            delta = direction_sign * self.signal_adjustment_max * scale

            self._adjustments[sig["ticker"]] = {
                "delta": delta,
                "composite": sig["composite"],
                "sources": sig["sources"],
                "step": self._step,
            }

            self.adjustment_log.append({
                "step": self._step,
                "ticker": sig["ticker"],
                "delta": round(delta, 4),
                "composite": round(sig["composite"], 3),
                "sources": sig["sources"],
            })
