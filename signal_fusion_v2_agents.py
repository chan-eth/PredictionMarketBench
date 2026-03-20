"""SignalFusion V2 agents with ML confidence gate and maker execution.

Pipeline: Signal Generation → Hard Gate → ML Gate → Maker Execution → Exit Management

Agents:
1. SignalFusionV2         — full pipeline (hard + ML gate + maker + exits)
2. SignalFusionV2HardOnly — hard gate only (no ML)
3. SignalFusionV2MakerOnly — maker execution only (no gates, isolates fee savings)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType, TimeInForce
from signal_agents import SignalScorer, TradeTapeAnalyzer, TapeRecord
from overhauled_agents import actual_yes_probability, time_decay_bonus
from gates.hard_gate import HardGate
from gates.confidence_gate import ConfidenceGate


DATA_DIR = Path(__file__).parent / "data"


class _SignalFusionV2Base(Agent):
    """Base class with shared signal generation, execution, and exit logic.

    Subclasses control which gates are active.
    """

    def __init__(
        self,
        use_hard_gate: bool = True,
        use_ml_gate: bool = True,
        max_positions: int = 10,
        max_per_market: int = 5,
        stale_order_steps: int = 60,
        stop_loss_cents: int = 8,
        take_profit_cents: int = 12,
        min_hours_for_exit: float = 2.0,
    ):
        self._use_hard_gate = use_hard_gate
        self._use_ml_gate = use_ml_gate
        self.max_positions = max_positions
        self.max_per_market = max_per_market
        self.stale_order_steps = stale_order_steps
        self.stop_loss = stop_loss_cents
        self.take_profit = take_profit_cents
        self.min_hours_for_exit = min_hours_for_exit

        # Signal infrastructure
        self.scorer = SignalScorer(min_composite=0.4, confluence_min_sources=2)
        self.tape = TradeTapeAnalyzer()

        # Gates
        self.hard_gate = HardGate() if use_hard_gate else None
        self.ml_gate = None
        if use_ml_gate:
            model_path = DATA_DIR / "confidence_gate_model.pkl"
            disabled_path = DATA_DIR / "ml_gate_disabled.txt"
            if model_path.exists() and not disabled_path.exists():
                self.ml_gate = ConfidenceGate(str(model_path))

        # State
        self._step = 0
        self._prev_book: dict[str, tuple[int, int]] = {}
        self._price_history: dict[str, list[int]] = {}
        self._order_placed_step: dict[str, int] = {}
        self.positions: dict[str, dict] = {}

        # Diagnostics
        self.trade_log: list[dict] = []
        self.gate_log: list[dict] = []
        self.signals_fired = 0
        self.signals_blocked_hard = 0
        self.signals_blocked_ml = 0

    def on_episode_start(self, metadata: dict) -> None:
        self._step = 0
        self._prev_book = {}
        self._price_history = {}
        self._order_placed_step = {}
        self.positions = {}
        self.scorer = SignalScorer(min_composite=0.4, confluence_min_sources=2)
        self.tape = TradeTapeAnalyzer()
        if self.hard_gate:
            self.hard_gate.reset_stats()

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        # Phase 1: Generate signals (identical to SignalFusionAgent)
        self._generate_signals(ctx, markets)

        # Phase 2: Score
        scored = self.scorer.score(self._step)

        # Phase 3: Cancel stale resting orders
        self._cancel_stale_orders(ctx)

        # Phase 4: Gate and execute
        for sig in scored:
            self.signals_fired += 1
            ticker = sig["ticker"]
            direction = sig["direction"]

            if ticker in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            market = next((m for m in markets if m.ticker == ticker), None)
            if market is None or market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
            spread = market.yes_best_ask - market.yes_best_bid
            hours_to_close = (market.time_to_close_seconds or 999999) / 3600.0

            book = ctx.get_orderbook(ticker)
            bid_depth = sum(level[1] for level in (book or {}).get("yes_bids", [])[:3])
            ask_depth = sum(level[1] for level in (book or {}).get("yes_asks", [])[:3])
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / max(total_depth, 1)

            # Derive category
            ep_id = getattr(self, '_episode_id', '')
            if "BTC" in ticker or "ETH" in ticker or "SOL" in ticker:
                category = "crypto"
            elif "HIGH" in ticker or "LOW" in ticker or "TEMP" in ticker:
                category = "weather"
            else:
                category = "sports"

            features = {
                "composite_score": sig["composite"],
                "n_sources": len(sig["sources"]),
                "source_ph": "premium_harvester" in sig["sources"],
                "source_momentum": "crypto_momentum" in sig["sources"],
                "source_tape": "trade_tape" in sig["sources"],
                "direction": direction,
                "yes_mid": yes_mid,
                "spread_cents": spread,
                "hours_to_close": hours_to_close,
                "book_depth_bid": bid_depth,
                "book_depth_ask": ask_depth,
                "book_imbalance": imbalance,
                "category": category,
            }

            # Hard gate
            if self.hard_gate:
                blocked, reason = self.hard_gate.should_block(features)
                if blocked:
                    self.signals_blocked_hard += 1
                    self.gate_log.append({"step": self._step, "ticker": ticker, "gate": "hard", "reason": reason})
                    continue

            # ML gate
            if self.ml_gate and self.ml_gate.enabled:
                if not self.ml_gate.should_pass(features):
                    self.signals_blocked_ml += 1
                    self.gate_log.append({"step": self._step, "ticker": ticker, "gate": "ml", "reason": "below_threshold"})
                    continue

            # Execute via maker pattern
            self._execute_signal(ctx, sig, market, features, cash)

        # Phase 5: Manage exits
        self._manage_exits(ctx, markets)

    def _generate_signals(self, ctx: AgentContext, markets) -> None:
        """Generate signals from PH edge, momentum, and tape sources."""
        # PH edge signals
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
            no_price = 100 - yes_mid
            if 93 <= no_price <= 99:
                yes_actual = actual_yes_probability(yes_mid)
                edge = (1.0 - yes_actual) - (no_price / 100.0)
                if edge > 0.005:
                    self.scorer.push(
                        market.ticker, "premium_harvester", "bearish",
                        min(edge * 10, 1.0), self._step,
                    )

        # Momentum signals
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            ticker = market.ticker
            mid = (market.yes_best_bid + market.yes_best_ask) // 2
            if ticker not in self._price_history:
                self._price_history[ticker] = []
            self._price_history[ticker].append(mid)
            hist = self._price_history[ticker]
            if len(hist) > 45:
                self._price_history[ticker] = hist[-30:]
                hist = self._price_history[ticker]
            if len(hist) >= 15:
                momentum = hist[-1] - hist[-15]
                if abs(momentum) >= 3 and 15 <= mid <= 85:
                    direction = "bullish" if momentum > 0 else "bearish"
                    conf = min(abs(momentum) / 10.0, 1.0)
                    self.scorer.push(ticker, "crypto_momentum", direction, conf, self._step)

        # Tape signals
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            ticker = market.ticker
            mid = (market.yes_best_bid + market.yes_best_ask) // 2
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

    def _execute_signal(self, ctx: AgentContext, sig: dict, market, features: dict, cash: dict) -> None:
        """Execute a signal via POST_ONLY maker order."""
        ticker = sig["ticker"]
        direction = sig["direction"]
        spread = features["spread_cents"]

        if direction == "bearish":
            # Buy NO: POST_ONLY at (100 - yes_ask) + 1c
            no_bid = 100 - market.yes_best_ask
            maker_price = no_bid + 1
            no_ask = 100 - market.yes_best_bid
            if maker_price >= no_ask:
                return  # would cross
            contracts = min(self.max_per_market, max(1, cash["cash_cents"] // (maker_price * 2)))
            if contracts < 1:
                return
            order = Order(
                ticker=ticker, side=Side.NO, action=Action.BUY,
                order_type=OrderType.LIMIT, count=contracts,
                limit_price_cents=maker_price, time_in_force=TimeInForce.POST_ONLY,
            )
        elif direction == "bullish":
            # Buy YES: POST_ONLY at yes_bid + 1c
            maker_price = market.yes_best_bid + 1
            if maker_price >= market.yes_best_ask:
                return  # would cross
            contracts = min(self.max_per_market, max(1, cash["cash_cents"] // (maker_price * 2)))
            if contracts < 1:
                return
            order = Order(
                ticker=ticker, side=Side.YES, action=Action.BUY,
                order_type=OrderType.LIMIT, count=contracts,
                limit_price_cents=maker_price, time_in_force=TimeInForce.POST_ONLY,
            )
        else:
            return

        result = ctx.place_order(order)
        if result and not result.get("rejected", False):
            resting = result.get("resting")
            fill = result.get("fill")
            filled = fill["count"] if fill else 0

            if resting:
                oid = resting.get("order_id") or resting.get("id", "")
                if oid:
                    self._order_placed_step[oid] = self._step

            if resting or filled > 0:
                yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
                side_str = "no" if direction == "bearish" else "yes"
                self.positions[ticker] = {
                    "side": side_str,
                    "entry_mid": yes_mid,
                    "entry_step": self._step,
                    "contracts": filled if filled > 0 else contracts,
                }
                self.trade_log.append({
                    "step": self._step, "ticker": ticker,
                    "direction": direction,
                    "composite": sig["composite"],
                    "sources": sig["sources"],
                    "maker_price": maker_price,
                })

    def _cancel_stale_orders(self, ctx: AgentContext) -> None:
        """Cancel resting orders that are too old or too far back in queue."""
        resting = ctx.get_resting_orders()
        for ro in resting:
            oid = ro["order_id"]
            placed = self._order_placed_step.get(oid, self._step)
            age = self._step - placed
            if ro["remaining_count"] == ro["original_count"]:
                if age > self.stale_order_steps or ro.get("env_ahead", 0) > 50:
                    ctx.cancel_order(oid)
                    self._order_placed_step.pop(oid, None)

    def _manage_exits(self, ctx: AgentContext, markets) -> None:
        """Check stop loss, take profit, and time-based exits."""
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            market = next((m for m in markets if m.ticker == ticker), None)
            if market is None or market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
            entry_mid = pos["entry_mid"]
            hours_left = (market.time_to_close_seconds or 999999) / 3600.0

            # PnL depends on side
            if pos["side"] == "yes":
                pnl_cents = yes_mid - entry_mid
            else:  # no
                pnl_cents = entry_mid - yes_mid

            should_exit = False
            reason = ""

            # Stop loss
            if pnl_cents <= -self.stop_loss:
                should_exit = True
                reason = "stop_loss"
            # Take profit
            elif pnl_cents >= self.take_profit:
                should_exit = True
                reason = "take_profit"
            # Time exit
            elif hours_left < self.min_hours_for_exit:
                should_exit = True
                reason = "time_exit"

            if should_exit:
                raw_positions = ctx.get_positions()
                actual = raw_positions.get(ticker, {})
                count = actual.get(f"{pos['side']}_contracts", 0)
                if count > 0:
                    side = Side.YES if pos["side"] == "yes" else Side.NO
                    order = Order(
                        ticker=ticker, side=side, action=Action.SELL,
                        order_type=OrderType.MARKET, count=count,
                    )
                    ctx.place_order(order)
                del self.positions[ticker]


class SignalFusionV2(_SignalFusionV2Base):
    """Full V2: hard gate + ML gate + maker execution + exits."""
    def __init__(self):
        super().__init__(use_hard_gate=True, use_ml_gate=True)


class SignalFusionV2HardOnly(_SignalFusionV2Base):
    """V2 with hard gate only (no ML)."""
    def __init__(self):
        super().__init__(use_hard_gate=True, use_ml_gate=False)


class SignalFusionV2MakerOnly(_SignalFusionV2Base):
    """V2 with maker execution only (no gates, isolates fee savings)."""
    def __init__(self):
        super().__init__(use_hard_gate=False, use_ml_gate=False)
