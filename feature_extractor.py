"""Feature extraction agent: replays SignalFusion pipeline, logs every scored signal
with rich features and settlement labels for ML gate training.

Trades nothing — only observes and records.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import Agent, AgentContext
from signal_agents import SignalScorer, TradeTapeAnalyzer, TapeRecord
from our_agents import actual_yes_probability


class FeatureExtractorAgent(Agent):
    """Runs exact SignalFusion signal pipeline but records features instead of trading.

    Uses min_composite=0.0 and confluence_min_sources=1 to capture ALL scored signals,
    including those the original agent would have filtered out.
    """

    def __init__(self):
        self.scorer = SignalScorer(
            min_composite=0.0,
            confluence_min_sources=1,
            confluence_window_steps=60,
        )
        self.tape = TradeTapeAnalyzer()
        self._step = 0
        self._prev_book: dict[str, tuple[int, int]] = {}
        self._price_history: dict[str, list[int]] = {}
        self._ph_edges: dict[str, float] = {}
        self._momentum_vals: dict[str, float] = {}
        self._tape_confs: dict[str, float] = {}

        # Settlement data loaded per episode
        self._settlements: dict[str, str] = {}  # ticker -> "YES"|"NO"
        self._episode_id: str = ""

        # Collected feature rows across all episodes
        self.feature_rows: list[dict] = []

    def on_episode_start(self, metadata: dict) -> None:
        self._step = 0
        self._prev_book = {}
        self._price_history = {}
        self._ph_edges = {}
        self._momentum_vals = {}
        self._tape_confs = {}
        self.scorer = SignalScorer(
            min_composite=0.0,
            confluence_min_sources=1,
            confluence_window_steps=60,
        )
        self.tape = TradeTapeAnalyzer()

        self._episode_id = metadata.get("episode_id", "")

        # Load settlement data from episode directory
        episode_dir = Path(__file__).parent / "episodes" / self._episode_id
        settlement_file = episode_dir / "settlement.json"
        if settlement_file.exists():
            with open(settlement_file) as f:
                raw = json.load(f)
            self._settlements = {
                ticker: info["result"] for ticker, info in raw.items()
            }
        else:
            self._settlements = {}

    def on_episode_end(self, result: dict) -> None:
        pass  # feature_rows already accumulated during act()

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()

        # Clear per-step signal context
        self._ph_edges.clear()
        self._momentum_vals.clear()
        self._tape_confs.clear()

        # --- Generate signals (identical to SignalFusionAgent) ---

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
                    self._ph_edges[market.ticker] = edge
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
                    self._momentum_vals[ticker] = abs(momentum)
                    direction = "bullish" if momentum > 0 else "bearish"
                    conf = min(abs(momentum) / 10.0, 1.0)
                    self.scorer.push(ticker, "crypto_momentum", direction, conf, self._step)

        # Tape signals (orderbook delta)
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
                            best = self._tape_confs.get(ticker, 0.0)
                            self._tape_confs[ticker] = max(best, ts["confidence"])
                            self.scorer.push(
                                ticker, "trade_tape",
                                ts["direction"], ts["confidence"], self._step,
                            )

        # --- Score signals ---
        scored = self.scorer.score(self._step)

        # --- Record features for each scored signal ---
        for sig in scored:
            ticker = sig["ticker"]
            direction = sig["direction"]
            sources = sig["sources"]

            # Find the market for this ticker
            market = next((m for m in markets if m.ticker == ticker), None)
            if market is None or market.yes_best_bid is None or market.yes_best_ask is None:
                continue

            yes_mid = (market.yes_best_bid + market.yes_best_ask) // 2
            spread = market.yes_best_ask - market.yes_best_bid
            hours_to_close = (market.time_to_close_seconds or 999999) / 3600.0

            # Orderbook depth
            book = ctx.get_orderbook(ticker)
            if book:
                bid_depth = sum(level[1] for level in book.get("yes_bids", [])[:3])
                ask_depth = sum(level[1] for level in book.get("yes_asks", [])[:3])
            else:
                bid_depth = 0
                ask_depth = 0
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / max(total_depth, 1)

            # Settlement label
            settlement_result = self._settlements.get(ticker, "UNKNOWN")
            if settlement_result == "UNKNOWN":
                signal_correct = None  # can't label
            elif direction == "bearish" and settlement_result == "NO":
                signal_correct = True
            elif direction == "bullish" and settlement_result == "YES":
                signal_correct = True
            else:
                signal_correct = False

            # Derive category from episode_id
            if "BTC" in self._episode_id or "ETH" in self._episode_id or "SOL" in self._episode_id:
                category = "crypto"
            elif "HIGH" in self._episode_id or "LOW" in self._episode_id or "TEMP" in self._episode_id:
                category = "weather"
            else:
                category = "sports"

            row = {
                "episode_id": self._episode_id,
                "step": self._step,
                "ticker": ticker,
                "direction": direction,
                "composite_score": sig["composite"],
                "n_sources": len(sources),
                "source_ph": "premium_harvester" in sources,
                "source_momentum": "crypto_momentum" in sources,
                "source_tape": "trade_tape" in sources,
                "ph_edge": self._ph_edges.get(ticker, 0.0),
                "momentum_magnitude": self._momentum_vals.get(ticker, 0.0),
                "tape_confidence": self._tape_confs.get(ticker, 0.0),
                "yes_mid": yes_mid,
                "spread_cents": spread,
                "hours_to_close": hours_to_close,
                "book_depth_bid": bid_depth,
                "book_depth_ask": ask_depth,
                "book_imbalance": imbalance,
                "category": category,
                "settlement_result": settlement_result,
                "signal_correct": signal_correct,
            }
            self.feature_rows.append(row)
