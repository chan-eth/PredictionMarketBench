"""
Signal Intelligence agents for backtesting the full pipeline.

Tests the unified signal scorer, trade tape analysis, and confluence logic
against historical Kalshi episode data via the oddpool framework.

Agents:
1. TapeScannerAgent     — trade tape pattern detection only
2. SignalFusionAgent    — full pipeline: tape + scorer + confluence
3. FullStackAgent       — PremiumHarvester + Momentum + SignalFusion combined
"""

import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType, TradePrint
from our_agents import PremiumHarvester, MomentumSniper, actual_yes_probability

# ---------------------------------------------------------------------------
# Trade-tape analysis (Python port of Rust TradeTapeAnalyzer)
# ---------------------------------------------------------------------------

class TapeRecord:
    __slots__ = ("count", "price", "taker_side", "step")
    def __init__(self, count: int, price: int, taker_side: str, step: int):
        self.count = count
        self.price = price
        self.taker_side = taker_side
        self.step = step


class TradeTapeAnalyzer:
    """Python port of crates/trading-engine/src/trade_tape.rs for backtesting."""

    def __init__(
        self,
        block_zscore: float = 2.0,
        volume_zscore: float = 2.5,
        imbalance_threshold: float = 0.7,
        lookback_steps: int = 60,  # ~5min at 5s cadence
    ):
        self.block_zscore = block_zscore
        self.volume_zscore = volume_zscore
        self.imbalance_threshold = imbalance_threshold
        self.lookback_steps = lookback_steps
        self.states: dict[str, dict] = {}

    def _get_state(self, ticker: str) -> dict:
        if ticker not in self.states:
            self.states[ticker] = {
                "trades": deque(maxlen=500),
                "interval_vols": deque(maxlen=200),
            }
        return self.states[ticker]

    def ingest(self, ticker: str, trades: list[TapeRecord], step: int) -> list[dict]:
        """Ingest trades and return detected signals."""
        if not trades:
            return []

        state = self._get_state(ticker)
        for t in trades:
            state["trades"].append(t)

        vol = sum(t.count for t in trades)
        if vol > 0:
            state["interval_vols"].append((step, vol))

        # Prune old
        cutoff = step - self.lookback_steps
        while state["trades"] and state["trades"][0].step < cutoff:
            state["trades"].popleft()
        while state["interval_vols"] and state["interval_vols"][0][0] < cutoff:
            state["interval_vols"].popleft()

        signals = []
        s = self._detect_large_block(ticker, state)
        if s: signals.append(s)
        s = self._detect_sweep(ticker, state, step)
        if s: signals.append(s)
        s = self._detect_volume_surge(ticker, state)
        if s: signals.append(s)
        s = self._detect_taker_imbalance(ticker, state, step)
        if s: signals.append(s)
        return signals

    def _detect_large_block(self, ticker: str, state: dict) -> dict | None:
        trades = state["trades"]
        if len(trades) < 10:
            return None
        counts = [t.count for t in trades]
        mean = sum(counts) / len(counts)
        var = sum((c - mean) ** 2 for c in counts) / len(counts)
        std = var ** 0.5
        if std < 1:
            return None
        last = trades[-1]
        zscore = (last.count - mean) / std
        if zscore > self.block_zscore:
            return {
                "ticker": ticker, "type": "large_block",
                "direction": "bullish" if last.taker_side == "yes" else "bearish",
                "confidence": min(max(zscore / 5.0, 0), 1),
            }
        return None

    def _detect_sweep(self, ticker: str, state: dict, step: int) -> dict | None:
        # Last 1 step (~5s) of trades
        recent = [t for t in state["trades"] if t.step >= step - 1]
        if len(recent) < 3:
            return None
        prices = sorted(set(t.price for t in recent))
        if len(prices) < 3:
            return None
        span = prices[-1] - prices[0]
        if span <= len(prices):
            direction = "bullish" if recent[-1].price > recent[0].price else "bearish"
            return {
                "ticker": ticker, "type": "sweep",
                "direction": direction,
                "confidence": min(max(len(prices) / 5.0, 0), 1),
            }
        return None

    def _detect_volume_surge(self, ticker: str, state: dict) -> dict | None:
        vols = state["interval_vols"]
        if len(vols) < 10:
            return None
        volumes = [v for _, v in vols]
        mean = sum(volumes) / len(volumes)
        var = sum((v - mean) ** 2 for v in volumes) / len(volumes)
        std = var ** 0.5
        if std < 1:
            return None
        zscore = (volumes[-1] - mean) / std
        if zscore > self.volume_zscore:
            recent = list(state["trades"])[-10:]
            yes_vol = sum(t.count for t in recent if t.taker_side == "yes")
            no_vol = sum(t.count for t in recent if t.taker_side == "no")
            return {
                "ticker": ticker, "type": "volume_surge",
                "direction": "bullish" if yes_vol >= no_vol else "bearish",
                "confidence": min(max(zscore / 5.0, 0), 1),
            }
        return None

    def _detect_taker_imbalance(self, ticker: str, state: dict, step: int) -> dict | None:
        recent = [t for t in state["trades"] if t.step >= step - 12]  # ~60s
        if len(recent) < 5:
            return None
        yes_vol = sum(t.count for t in recent if t.taker_side == "yes")
        no_vol = sum(t.count for t in recent if t.taker_side == "no")
        total = yes_vol + no_vol
        if total == 0:
            return None
        yes_r = yes_vol / total
        no_r = no_vol / total
        if yes_r >= self.imbalance_threshold:
            return {"ticker": ticker, "type": "taker_imbalance", "direction": "bullish", "confidence": yes_r}
        if no_r >= self.imbalance_threshold:
            return {"ticker": ticker, "type": "taker_imbalance", "direction": "bearish", "confidence": no_r}
        return None


# ---------------------------------------------------------------------------
# Signal Scorer (Python port of Rust SignalScorer)
# ---------------------------------------------------------------------------

class SignalScorer:
    """Python port of crates/trading-engine/src/signal_scorer.rs."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        min_composite: float = 0.5,
        confluence_bonus: float = 0.2,
        confluence_min_sources: int = 3,
        confluence_window_steps: int = 60,
    ):
        self.weights = weights or {
            "premium_harvester": 1.0,
            "crypto_momentum": 0.8,
            "trade_tape": 0.6,
            "leader_copy": 0.4,
            "community_sentiment": 0.3,
        }
        self.min_composite = min_composite
        self.confluence_bonus = confluence_bonus
        self.confluence_min_sources = confluence_min_sources
        self.confluence_window_steps = confluence_window_steps
        self.recent: list[dict] = []

    def push(self, ticker: str, source: str, direction: str, confidence: float, step: int):
        self.recent.append({
            "ticker": ticker, "source": source, "direction": direction,
            "confidence": max(0, min(confidence, 1)), "step": step,
        })

    def score(self, current_step: int) -> list[dict]:
        # Prune old
        cutoff = current_step - self.confluence_window_steps
        self.recent = [s for s in self.recent if s["step"] > cutoff]

        # Group by (ticker, direction)
        groups: dict[tuple, list] = defaultdict(list)
        for s in self.recent:
            groups[(s["ticker"], s["direction"])].append(s)

        results = []
        fired_keys = set()

        for (ticker, direction), sigs in groups.items():
            # Best confidence per source
            best: dict[str, float] = {}
            for s in sigs:
                if s["source"] not in best or s["confidence"] > best[s["source"]]:
                    best[s["source"]] = s["confidence"]

            sources = list(best.keys())

            # Leader alone guard
            if len(sources) == 1 and sources[0] == "leader_copy":
                continue

            composite = sum(
                self.weights.get(src, 0) * conf for src, conf in best.items()
            )

            if len(sources) >= self.confluence_min_sources:
                composite *= 1.0 + self.confluence_bonus

            if composite >= self.min_composite:
                results.append({
                    "ticker": ticker, "direction": direction,
                    "composite": composite, "sources": sources,
                })
                fired_keys.add((ticker, direction))

        # Remove fired signals
        if fired_keys:
            self.recent = [
                s for s in self.recent
                if (s["ticker"], s["direction"]) not in fired_keys
            ]

        return results


# ---------------------------------------------------------------------------
# Agent 1: TapeScannerAgent — tape signals only
# ---------------------------------------------------------------------------

class TapeScannerAgent(Agent):
    """
    Trades based purely on trade tape pattern detection.
    Tests the tape analyzer in isolation.

    Uses orderbook *delta* method: tracks volume changes between steps
    to infer trade direction (decrease in asks = buying, decrease in bids = selling).
    """

    def __init__(self, max_positions: int = 5, contracts_per_trade: int = 3):
        self.analyzer = TradeTapeAnalyzer()
        self.max_positions = max_positions
        self.contracts = contracts_per_trade
        self.positions: dict[str, dict] = {}
        self._step = 0
        self._prev_book: dict[str, tuple[int, int]] = {}  # ticker → (bid_vol, ask_vol)
        # Logging
        self.signal_log: list[dict] = []

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}
        self._step = 0
        self._prev_book = {}
        self.signal_log = []
        self.analyzer = TradeTapeAnalyzer()

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        # Infer trades from orderbook volume *deltas* between steps
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            ticker = market.ticker
            mid = (market.yes_best_bid + market.yes_best_ask) // 2

            book = ctx.get_orderbook(ticker)
            bid_vol = sum(level[1] for level in (book or {}).get("yes_bids", [])[:3])
            ask_vol = sum(level[1] for level in (book or {}).get("yes_asks", [])[:3])

            prev = self._prev_book.get(ticker)
            self._prev_book[ticker] = (bid_vol, ask_vol)

            if prev is None:
                continue  # need previous snapshot to compute delta

            prev_bid, prev_ask = prev
            bid_consumed = max(0, prev_bid - bid_vol)  # bids eaten = selling
            ask_consumed = max(0, prev_ask - ask_vol)   # asks eaten = buying

            # Proportional delta: normalize by previous volume to avoid
            # structural bias from asymmetric book sizes
            bid_pct = bid_consumed / max(prev_bid, 1)
            ask_pct = ask_consumed / max(prev_ask, 1)
            delta = ask_pct - bid_pct  # positive = bullish, negative = bearish

            total_consumed = bid_consumed + ask_consumed
            if total_consumed > 2 and abs(delta) > 0.05:
                taker_side = "yes" if delta > 0 else "no"
                trades = [TapeRecord(count=total_consumed, price=mid, taker_side=taker_side, step=self._step)]
                signals = self.analyzer.ingest(ticker, trades, self._step)

                for sig in signals:
                    self.signal_log.append({**sig, "step": self._step})
                    self._maybe_trade(ctx, sig, market, cash)

        # Check exits (simple time-based: exit after 20 steps)
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            if self._step - pos["entry_step"] > 20:
                self._exit_position(ctx, ticker)

    def _maybe_trade(self, ctx, sig, market, cash):
        if len(self.positions) >= self.max_positions:
            return
        ticker = sig["ticker"]
        if ticker in self.positions:
            return
        if sig["confidence"] < 0.5:
            return

        if sig["direction"] == "bullish":
            cost = market.yes_best_ask
            if cost and cash["cash_cents"] >= cost * self.contracts * 2:
                order = Order(ticker=ticker, side=Side.YES, action=Action.BUY,
                              order_type=OrderType.MARKET, count=self.contracts)
                result = ctx.place_order(order)
                if result and result.get("filled", 0) > 0:
                    self.positions[ticker] = {"side": "yes", "entry_step": self._step, "contracts": result["filled"]}
        elif sig["direction"] == "bearish":
            cost = market.no_best_ask or (100 - (market.yes_best_bid or 50))
            if cost and cash["cash_cents"] >= cost * self.contracts * 2:
                order = Order(ticker=ticker, side=Side.NO, action=Action.BUY,
                              order_type=OrderType.MARKET, count=self.contracts)
                result = ctx.place_order(order)
                if result and result.get("filled", 0) > 0:
                    self.positions[ticker] = {"side": "no", "entry_step": self._step, "contracts": result["filled"]}

    def _exit_position(self, ctx, ticker):
        pos = self.positions.pop(ticker, None)
        if not pos:
            return
        raw = ctx.get_positions()
        actual = raw.get(ticker, {})
        count = actual.get(f"{pos['side']}_contracts", 0)
        if count > 0:
            side = Side.YES if pos["side"] == "yes" else Side.NO
            order = Order(ticker=ticker, side=side, action=Action.SELL,
                          order_type=OrderType.MARKET, count=count)
            ctx.place_order(order)


# ---------------------------------------------------------------------------
# Agent 2: SignalFusionAgent — full signal pipeline
# ---------------------------------------------------------------------------

class SignalFusionAgent(Agent):
    """
    Full signal intelligence pipeline: Premium Harvester + Momentum + Tape → Scorer.

    Tests the signal bus pattern with weighted fusion and confluence bonuses.
    """

    def __init__(self):
        self.harvester = PremiumHarvester(max_position_per_market=5, max_total_positions=10)
        self.momentum = MomentumSniper(lookback=15, entry_threshold_cents=3, max_position=3)
        self.tape = TradeTapeAnalyzer()
        self.scorer = SignalScorer(min_composite=0.4, confluence_min_sources=2)
        self._step = 0
        self.positions: dict[str, dict] = {}
        self.signal_log: list[dict] = []
        self.trade_log: list[dict] = []
        self._prev_book: dict[str, tuple[int, int]] = {}  # ticker → (bid_vol, ask_vol)

    def on_episode_start(self, metadata: dict) -> None:
        self.harvester.on_episode_start(metadata)
        self.momentum.on_episode_start(metadata)
        self._step = 0
        self.positions = {}
        self.signal_log = []
        self.trade_log = []
        self._prev_book = {}
        self.tape = TradeTapeAnalyzer()
        self.scorer = SignalScorer(min_composite=0.4, confluence_min_sources=2)

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        # --- Phase 1: Generate signals from all sources ---

        # Premium Harvester signals
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
            no_price = 100 - yes_mid
            if 93 <= no_price <= 99:
                yes_actual = actual_yes_probability(yes_mid)
                edge = (1.0 - yes_actual) - (no_price / 100.0)
                if edge > 0.005:
                    self.scorer.push(market.ticker, "premium_harvester", "bearish", min(edge * 10, 1.0), self._step)

        # Momentum signals
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            ticker = market.ticker
            mid = (market.yes_best_bid + market.yes_best_ask) // 2
            if ticker not in self.momentum.price_history:
                self.momentum.price_history[ticker] = []
            self.momentum.price_history[ticker].append(mid)
            hist = self.momentum.price_history[ticker]
            if len(hist) >= self.momentum.lookback:
                momentum = hist[-1] - hist[-self.momentum.lookback]
                if abs(momentum) >= self.momentum.entry_threshold and 15 <= mid <= 85:
                    direction = "bullish" if momentum > 0 else "bearish"
                    conf = min(abs(momentum) / 10.0, 1.0)
                    self.scorer.push(ticker, "crypto_momentum", direction, conf, self._step)

        # Tape signals (orderbook delta method — track consumed volume)
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

                    # Proportional delta to avoid structural bias
                    bid_pct = bid_consumed / max(prev_bid, 1)
                    ask_pct = ask_consumed / max(prev_ask, 1)
                    delta = ask_pct - bid_pct

                    total_consumed = bid_consumed + ask_consumed
                    if total_consumed > 2 and abs(delta) > 0.05:
                        taker_side = "yes" if delta > 0 else "no"
                        tape_trades = [TapeRecord(count=total_consumed, price=mid, taker_side=taker_side, step=self._step)]
                        tape_sigs = self.tape.ingest(ticker, tape_trades, self._step)
                        for ts in tape_sigs:
                            self.scorer.push(ticker, "trade_tape", ts["direction"], ts["confidence"], self._step)
                            self.signal_log.append({**ts, "step": self._step})

        # --- Phase 2: Score and execute ---
        scored = self.scorer.score(self._step)

        for sig in scored:
            self.signal_log.append({
                "step": self._step, "ticker": sig["ticker"], "type": "scored",
                "direction": sig["direction"], "composite": sig["composite"],
                "sources": sig["sources"],
            })

            ticker = sig["ticker"]
            if ticker in self.positions:
                continue
            if len(self.positions) >= 15:
                break

            market = next((m for m in markets if m.ticker == ticker), None)
            if not market or market.yes_best_bid is None:
                continue

            if sig["direction"] == "bullish":
                cost = market.yes_best_ask
                if cost and cash["cash_cents"] >= cost * 5:
                    contracts = min(3, cash["cash_cents"] // (cost * 3))
                    if contracts >= 1:
                        order = Order(ticker=ticker, side=Side.YES, action=Action.BUY,
                                      order_type=OrderType.MARKET, count=contracts)
                        result = ctx.place_order(order)
                        if result and result.get("filled", 0) > 0:
                            mid = (market.yes_best_bid + market.yes_best_ask) // 2
                            self.positions[ticker] = {"side": "yes", "entry": mid, "step": self._step}
                            self.trade_log.append({"step": self._step, "ticker": ticker, "action": "buy_yes",
                                                    "composite": sig["composite"], "sources": sig["sources"]})

            elif sig["direction"] == "bearish":
                no_ask = 100 - market.yes_best_bid if market.yes_best_bid else None
                if no_ask and cash["cash_cents"] >= no_ask * 5:
                    contracts = min(3, cash["cash_cents"] // (no_ask * 3))
                    if contracts >= 1:
                        order = Order(ticker=ticker, side=Side.NO, action=Action.BUY,
                                      order_type=OrderType.MARKET, count=contracts)
                        result = ctx.place_order(order)
                        if result and result.get("filled", 0) > 0:
                            mid = (market.yes_best_bid + market.yes_best_ask) // 2
                            self.positions[ticker] = {"side": "no", "entry": mid, "step": self._step}
                            self.trade_log.append({"step": self._step, "ticker": ticker, "action": "buy_no",
                                                    "composite": sig["composite"], "sources": sig["sources"]})

        # --- Phase 3: Exit management ---
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            market = next((m for m in markets if m.ticker == ticker), None)
            if not market or market.yes_best_bid is None:
                continue
            mid = (market.yes_best_bid + market.yes_best_ask) // 2
            entry = pos["entry"]
            age = self._step - pos["step"]

            should_exit = False
            if pos["side"] == "yes":
                should_exit = (mid - entry >= 3) or (mid - entry <= -5) or (age > 30)
            else:
                should_exit = (entry - mid >= 3) or (entry - mid <= -5) or (age > 30)

            if should_exit:
                raw = ctx.get_positions()
                actual = raw.get(ticker, {})
                count = actual.get(f"{pos['side']}_contracts", 0)
                if count > 0:
                    side = Side.YES if pos["side"] == "yes" else Side.NO
                    order = Order(ticker=ticker, side=side, action=Action.SELL,
                                  order_type=OrderType.MARKET, count=count)
                    ctx.place_order(order)
                    self.trade_log.append({"step": self._step, "ticker": ticker, "action": f"sell_{pos['side']}",
                                            "pnl_cents": (mid - entry) if pos["side"] == "yes" else (entry - mid)})
                del self.positions[ticker]


# ---------------------------------------------------------------------------
# Agent 3: FullStackAgent — everything combined
# ---------------------------------------------------------------------------

class FullStackAgent(Agent):
    """
    Runs PremiumHarvester standalone (proven edge) + SignalFusion for directional trades.
    This is the closest simulation to how the production engine will operate.
    """

    def __init__(self):
        self.harvester = PremiumHarvester(max_position_per_market=5, max_total_positions=15)
        self.fusion = SignalFusionAgent()
        self._step = 0

    def on_episode_start(self, metadata: dict) -> None:
        self.harvester.on_episode_start(metadata)
        self.fusion.on_episode_start(metadata)
        self._step = 0

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        # Harvester runs every 5 steps (slow cadence, proven edge)
        if self._step % 5 == 0:
            self.harvester.act(ctx)
        # Fusion runs every step (responsive to signals)
        self.fusion.act(ctx)
