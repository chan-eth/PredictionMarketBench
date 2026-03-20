"""
Expansion strategy agents for backtesting the 3 new strategies.

Agents:
1. BundleArbAgent          — Scans for YES_ask + NO_ask < 100c, buys both sides
2. PHWithEnsembleSignals   — PH V2 + simulated LLM ensemble consensus gating
3. PHWithFuzzyXMarket      — PH V2 + simulated fuzzy cross-market matching
4. FullExpansionAgent      — All three combined (PH + bundle arb + ensemble + fuzzy)
"""

import sys
import random
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType, TimeInForce
from signal_agents import TradeTapeAnalyzer, TapeRecord, SignalScorer
from overhauled_agents import actual_yes_probability, time_decay_bonus


# ============================================================
# STRATEGY 1: Bundle Arbitrage Agent
# ============================================================

class BundleArbAgent(Agent):
    """
    Scans every market for YES_ask + NO_ask < 100c.
    When found, buys both sides — guaranteed profit at resolution.

    This is pure structural arbitrage: no directional view, no probability
    estimate. Profit = 100 - (yes_cost + no_cost) per contract pair.
    """

    def __init__(
        self,
        min_edge_cents: int = 3,
        max_contracts: int = 25,
    ):
        self.min_edge_cents = min_edge_cents
        self.max_contracts = max_contracts
        # Track arb positions: ticker → {"yes": count, "no": count, "gap": cents}
        self.arb_positions: dict[str, dict] = {}
        self._step = 0

        # Diagnostics
        self.arb_log: list[dict] = []

    def on_episode_start(self, metadata: dict) -> None:
        self.arb_positions = {}
        self._step = 0
        self.arb_log = []

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        for market in markets:
            ticker = market.ticker
            if ticker in self.arb_positions:
                continue

            # Need both sides to have quotes
            if market.yes_best_ask is None or market.no_best_ask is None:
                continue

            yes_ask = market.yes_best_ask
            no_ask = market.no_best_ask

            if yes_ask <= 0 or yes_ask >= 100 or no_ask <= 0 or no_ask >= 100:
                continue

            combined = yes_ask + no_ask
            if combined >= 100:
                continue

            gap = 100 - combined
            if gap < self.min_edge_cents:
                continue

            # Check we can afford both sides
            cost_per_pair = yes_ask + no_ask
            max_by_cash = cash["cash_cents"] // max(cost_per_pair * 2, 1)
            qty = min(self.max_contracts, max(max_by_cash, 1))
            if qty < 1:
                continue

            # Buy YES side
            yes_order = Order(
                ticker=ticker,
                side=Side.YES,
                action=Action.BUY,
                order_type=OrderType.LIMIT,
                count=qty,
                limit_price_cents=yes_ask,
                time_in_force=TimeInForce.IOC,
            )
            yes_result = ctx.place_order(yes_order)
            yes_filled = 0
            if yes_result and not yes_result.get("rejected", False):
                fill = yes_result.get("fill")
                if fill:
                    yes_filled = fill["count"]

            if yes_filled == 0:
                continue

            # Buy NO side (match YES fill quantity)
            no_order = Order(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                order_type=OrderType.LIMIT,
                count=yes_filled,
                limit_price_cents=no_ask,
                time_in_force=TimeInForce.IOC,
            )
            no_result = ctx.place_order(no_order)
            no_filled = 0
            if no_result and not no_result.get("rejected", False):
                fill = no_result.get("fill")
                if fill:
                    no_filled = fill["count"]

            if no_filled > 0:
                self.arb_positions[ticker] = {
                    "yes": yes_filled,
                    "no": no_filled,
                    "gap": gap,
                    "yes_cost": yes_ask,
                    "no_cost": no_ask,
                }
                self.arb_log.append({
                    "step": self._step,
                    "ticker": ticker,
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                    "gap": gap,
                    "qty": min(yes_filled, no_filled),
                    "profit_cents": gap * min(yes_filled, no_filled),
                })


# ============================================================
# STRATEGY 2: PH with Simulated LLM Ensemble Signals
# ============================================================

class PHWithEnsembleSignals(Agent):
    """
    PremiumHarvester V2 + simulated LLM ensemble probability adjustments.

    Simulates what the real ensemble does:
    - Multiple "models" generate probability estimates (with noise)
    - If models disagree (std > consensus_spread), no adjustment (consensus gate)
    - If models agree, the median estimate adjusts FLB probability
    - Better calibration → better edge estimates → fewer bad trades

    The simulation uses the FLB actual probability as ground truth and adds
    model noise to simulate LLM estimation error.
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
        # Ensemble params
        n_models: int = 2,
        model_noise_std: float = 0.08,
        consensus_spread: float = 0.15,
        adjustment_weight: float = 0.3,
    ):
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.min_spread = min_spread
        self.stale_order_steps = stale_order_steps

        self.n_models = n_models
        self.model_noise_std = model_noise_std
        self.consensus_spread = consensus_spread
        self.adjustment_weight = adjustment_weight

        self.positions: dict[str, int] = {}
        self._order_placed_step: dict[str, int] = {}
        self._step = 0
        self._rng = random.Random(42)  # deterministic for reproducibility

        # Diagnostics
        self.trade_log: list[dict] = []
        self.ensemble_log: list[dict] = []

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}
        self._order_placed_step = {}
        self._step = 0
        self._rng = random.Random(42)
        self.trade_log = []
        self.ensemble_log = []

    def _simulate_ensemble(self, true_yes_prob: float) -> tuple[float | None, float]:
        """Simulate N models estimating probability with noise.

        Returns (adjusted_prob, consensus_std) or (None, std) if consensus fails.
        """
        estimates = []
        for _ in range(self.n_models):
            noise = self._rng.gauss(0, self.model_noise_std)
            est = max(0.01, min(0.99, true_yes_prob + noise))
            estimates.append(est)

        median = statistics.median(estimates)
        std = statistics.stdev(estimates) if len(estimates) > 1 else 0.0

        if std > self.consensus_spread:
            return None, std  # consensus failed

        return median, std

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        # Cancel stale orders
        resting = ctx.get_resting_orders()
        for ro in resting:
            order_id = ro["order_id"]
            placed_step = self._order_placed_step.get(order_id, self._step)
            age = self._step - placed_step
            if ro["remaining_count"] == ro["original_count"]:
                if age > self.stale_order_steps or ro["env_ahead"] > 50:
                    ctx.cancel_order(order_id)
                    self._order_placed_step.pop(order_id, None)

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
            yes_spread = market.yes_best_ask - market.yes_best_bid
            if yes_spread < self.min_spread:
                continue

            # FLB base probability
            yes_actual = actual_yes_probability(yes_mid)

            # Simulate ensemble consensus
            ensemble_est, ensemble_std = self._simulate_ensemble(yes_actual)

            if ensemble_est is not None:
                # Blend ensemble estimate with FLB
                adjusted_yes_prob = (
                    yes_actual * (1 - self.adjustment_weight)
                    + ensemble_est * self.adjustment_weight
                )
                adjusted_yes_prob = max(0.001, min(0.999, adjusted_yes_prob))
            else:
                adjusted_yes_prob = yes_actual  # no ensemble signal

            self.ensemble_log.append({
                "step": self._step,
                "ticker": ticker,
                "flb_prob": round(yes_actual, 4),
                "ensemble_est": round(ensemble_est, 4) if ensemble_est else None,
                "ensemble_std": round(ensemble_std, 4),
                "consensus": ensemble_est is not None,
                "adjusted": round(adjusted_yes_prob, 4),
            })

            # Edge with adjusted probability
            no_actual = 1.0 - adjusted_yes_prob
            no_implied = no_price / 100.0
            base_edge = no_actual - no_implied

            hours = (market.time_to_close_seconds or 999999) / 3600.0
            decay = time_decay_bonus(hours)
            edge = base_edge + decay

            if edge < self.min_edge:
                continue

            # Maker pricing
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
                        "ensemble_used": ensemble_est is not None,
                        "ensemble_std": round(ensemble_std, 4),
                    })


# ============================================================
# STRATEGY 3: PH with Simulated Fuzzy Cross-Market Signals
# ============================================================

class PHWithFuzzyXMarket(Agent):
    """
    PremiumHarvester V2 + simulated fuzzy cross-market arbitrage signals.

    Simulates what the enhanced cross-market matcher does:
    - For each market, simulate a "Polymarket price" with correlation + noise
    - Compute discrepancy between Kalshi and simulated Poly price
    - Apply as probability adjustment (same as production signal pipeline)

    Fuzzy matching improvement is modeled by: more markets get signals
    (keyword match rate ~40% → fuzzy brings it to ~60%).
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
        # Cross-market params
        match_probability: float = 0.6,   # 60% of markets have cross-platform pair
        price_correlation: float = 0.85,  # correlation between platforms
        min_discrepancy: float = 0.08,
        signal_adjustment_max: float = 0.03,
    ):
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.min_spread = min_spread
        self.stale_order_steps = stale_order_steps

        self.match_probability = match_probability
        self.price_correlation = price_correlation
        self.min_discrepancy = min_discrepancy
        self.signal_adjustment_max = signal_adjustment_max

        self.positions: dict[str, int] = {}
        self._order_placed_step: dict[str, int] = {}
        self._step = 0
        self._rng = random.Random(123)

        # Simulated cross-market state
        self._xmarket_adjustments: dict[str, float] = {}
        self._xmarket_matched: set[str] = set()

        # Diagnostics
        self.trade_log: list[dict] = []
        self.xmarket_log: list[dict] = []

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}
        self._order_placed_step = {}
        self._step = 0
        self._rng = random.Random(123)
        self._xmarket_adjustments = {}
        self._xmarket_matched = set()
        self.trade_log = []
        self.xmarket_log = []

    def _simulate_cross_market(self, ticker: str, kalshi_yes_prob: float) -> float | None:
        """Simulate a cross-platform price match.

        Returns a probability adjustment delta, or None if no match.
        """
        # Deterministic per-ticker match (simulate fuzzy finding more matches)
        if ticker not in self._xmarket_matched:
            if self._rng.random() < self.match_probability:
                self._xmarket_matched.add(ticker)
            else:
                return None

        if ticker not in self._xmarket_matched:
            return None

        # Simulate Polymarket price: correlated with Kalshi but with noise
        noise = self._rng.gauss(0, 1 - self.price_correlation)
        poly_prob = max(0.01, min(0.99,
            kalshi_yes_prob * self.price_correlation + noise * 0.15
        ))

        discrepancy = poly_prob - kalshi_yes_prob
        if abs(discrepancy) < self.min_discrepancy:
            return None

        # Convert to adjustment: positive disc = Poly thinks YES more likely
        scale = min(abs(discrepancy) / 0.20, 1.0)
        direction = 1.0 if discrepancy > 0 else -1.0
        delta = direction * self.signal_adjustment_max * scale

        self.xmarket_log.append({
            "step": self._step,
            "ticker": ticker,
            "kalshi_prob": round(kalshi_yes_prob, 3),
            "poly_prob": round(poly_prob, 3),
            "discrepancy": round(discrepancy, 3),
            "delta": round(delta, 4),
        })

        return delta

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        # Cancel stale orders
        resting = ctx.get_resting_orders()
        for ro in resting:
            order_id = ro["order_id"]
            placed_step = self._order_placed_step.get(order_id, self._step)
            age = self._step - placed_step
            if ro["remaining_count"] == ro["original_count"]:
                if age > self.stale_order_steps or ro["env_ahead"] > 50:
                    ctx.cancel_order(order_id)
                    self._order_placed_step.pop(order_id, None)

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
            yes_spread = market.yes_best_ask - market.yes_best_bid
            if yes_spread < self.min_spread:
                continue

            yes_actual = actual_yes_probability(yes_mid)

            # Simulate cross-market signal
            xm_delta = self._simulate_cross_market(ticker, yes_mid / 100.0)
            adjustment = xm_delta if xm_delta is not None else 0.0
            adjusted_yes_prob = max(0.001, min(0.999, yes_actual + adjustment))

            no_actual = 1.0 - adjusted_yes_prob
            no_implied = no_price / 100.0
            base_edge = no_actual - no_implied

            hours = (market.time_to_close_seconds or 999999) / 3600.0
            decay = time_decay_bonus(hours)
            edge = base_edge + decay

            if edge < self.min_edge:
                continue

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
                        "xmarket_signal": xm_delta is not None,
                    })


# ============================================================
# STRATEGY 4: Full Expansion (PH + Bundle Arb + Ensemble + XMarket)
# ============================================================

class FullExpansionAgent(Agent):
    """
    Combines all three expansion strategies with PH V2:
    - Bundle Arb: scan for YES+NO < 100c (zero-risk, separate from PH)
    - LLM Ensemble: consensus-gated probability adjustments
    - Fuzzy Cross-Market: enhanced matching probability adjustments
    - PH V2: core profitable strategy with adjusted probabilities
    """

    def __init__(
        self,
        # PH params
        min_yes_price: int = 1,
        max_yes_price: int = 20,
        min_edge: float = 0.005,
        max_position_per_market: int = 10,
        max_total_positions: int = 20,
        min_spread: int = 3,
        stale_order_steps: int = 60,
        # Bundle arb
        bundle_min_edge_cents: int = 3,
        bundle_max_contracts: int = 25,
        # Ensemble
        n_models: int = 2,
        model_noise_std: float = 0.08,
        consensus_spread: float = 0.15,
        ensemble_weight: float = 0.3,
        # Cross-market
        match_probability: float = 0.6,
        price_correlation: float = 0.85,
        min_discrepancy: float = 0.08,
        signal_adjustment_max: float = 0.03,
    ):
        # PH
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.min_spread = min_spread
        self.stale_order_steps = stale_order_steps
        # Bundle arb
        self.bundle_min_edge_cents = bundle_min_edge_cents
        self.bundle_max_contracts = bundle_max_contracts
        # Ensemble
        self.n_models = n_models
        self.model_noise_std = model_noise_std
        self.consensus_spread = consensus_spread
        self.ensemble_weight = ensemble_weight
        # XMarket
        self.match_probability = match_probability
        self.price_correlation = price_correlation
        self.min_discrepancy = min_discrepancy
        self.signal_adjustment_max = signal_adjustment_max

        # State
        self.positions: dict[str, int] = {}
        self.arb_positions: dict[str, dict] = {}
        self._order_placed_step: dict[str, int] = {}
        self._step = 0
        self._rng_ensemble = random.Random(42)
        self._rng_xmarket = random.Random(123)
        self._xmarket_matched: set[str] = set()

        # Diagnostics
        self.trade_log: list[dict] = []
        self.arb_log: list[dict] = []

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}
        self.arb_positions = {}
        self._order_placed_step = {}
        self._step = 0
        self._rng_ensemble = random.Random(42)
        self._rng_xmarket = random.Random(123)
        self._xmarket_matched = set()
        self.trade_log = []
        self.arb_log = []

    def _ensemble_estimate(self, true_prob: float) -> float | None:
        estimates = []
        for _ in range(self.n_models):
            noise = self._rng_ensemble.gauss(0, self.model_noise_std)
            est = max(0.01, min(0.99, true_prob + noise))
            estimates.append(est)
        std = statistics.stdev(estimates) if len(estimates) > 1 else 0.0
        if std > self.consensus_spread:
            return None
        return statistics.median(estimates)

    def _xmarket_delta(self, ticker: str, kalshi_prob: float) -> float | None:
        if ticker not in self._xmarket_matched:
            if self._rng_xmarket.random() < self.match_probability:
                self._xmarket_matched.add(ticker)
            else:
                return None
        noise = self._rng_xmarket.gauss(0, 1 - self.price_correlation)
        poly_prob = max(0.01, min(0.99,
            kalshi_prob * self.price_correlation + noise * 0.15
        ))
        disc = poly_prob - kalshi_prob
        if abs(disc) < self.min_discrepancy:
            return None
        scale = min(abs(disc) / 0.20, 1.0)
        return (1.0 if disc > 0 else -1.0) * self.signal_adjustment_max * scale

    def act(self, ctx: AgentContext) -> None:
        self._step += 1
        markets = ctx.get_markets()
        cash = ctx.get_cash()

        # ---- Bundle Arb Pass ----
        for market in markets:
            ticker = market.ticker
            if ticker in self.arb_positions:
                continue
            if market.yes_best_ask is None or market.no_best_ask is None:
                continue

            yes_ask = market.yes_best_ask
            no_ask = market.no_best_ask
            if yes_ask <= 0 or yes_ask >= 100 or no_ask <= 0 or no_ask >= 100:
                continue

            combined = yes_ask + no_ask
            if combined >= 100:
                continue
            gap = 100 - combined
            if gap < self.bundle_min_edge_cents:
                continue

            qty = min(self.bundle_max_contracts,
                      cash["cash_cents"] // max((yes_ask + no_ask) * 2, 1))
            if qty < 1:
                continue

            yes_order = Order(
                ticker=ticker, side=Side.YES, action=Action.BUY,
                order_type=OrderType.LIMIT, count=qty,
                limit_price_cents=yes_ask, time_in_force=TimeInForce.IOC,
            )
            yes_result = ctx.place_order(yes_order)
            yes_filled = 0
            if yes_result and not yes_result.get("rejected"):
                fill = yes_result.get("fill")
                if fill:
                    yes_filled = fill["count"]

            if yes_filled > 0:
                no_order = Order(
                    ticker=ticker, side=Side.NO, action=Action.BUY,
                    order_type=OrderType.LIMIT, count=yes_filled,
                    limit_price_cents=no_ask, time_in_force=TimeInForce.IOC,
                )
                no_result = ctx.place_order(no_order)
                no_filled = 0
                if no_result and not no_result.get("rejected"):
                    fill = no_result.get("fill")
                    if fill:
                        no_filled = fill["count"]

                if no_filled > 0:
                    self.arb_positions[ticker] = {"gap": gap, "qty": min(yes_filled, no_filled)}
                    self.arb_log.append({
                        "step": self._step, "ticker": ticker,
                        "gap": gap, "qty": min(yes_filled, no_filled),
                    })

        # ---- Cancel stale PH orders ----
        resting = ctx.get_resting_orders()
        for ro in resting:
            order_id = ro["order_id"]
            placed_step = self._order_placed_step.get(order_id, self._step)
            age = self._step - placed_step
            if ro["remaining_count"] == ro["original_count"]:
                if age > self.stale_order_steps or ro["env_ahead"] > 50:
                    ctx.cancel_order(order_id)
                    self._order_placed_step.pop(order_id, None)

        if len(self.positions) >= self.max_total_positions:
            return

        # ---- PH with combined ensemble + cross-market adjustments ----
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
            yes_spread = market.yes_best_ask - market.yes_best_bid
            if yes_spread < self.min_spread:
                continue

            yes_actual = actual_yes_probability(yes_mid)

            # Ensemble adjustment
            ensemble_est = self._ensemble_estimate(yes_actual)
            if ensemble_est is not None:
                adjusted = (
                    yes_actual * (1 - self.ensemble_weight)
                    + ensemble_est * self.ensemble_weight
                )
            else:
                adjusted = yes_actual

            # Cross-market adjustment (additive)
            xm_delta = self._xmarket_delta(ticker, yes_mid / 100.0)
            if xm_delta is not None:
                adjusted += xm_delta

            adjusted = max(0.001, min(0.999, adjusted))

            no_actual = 1.0 - adjusted
            no_implied = no_price / 100.0
            base_edge = no_actual - no_implied

            hours = (market.time_to_close_seconds or 999999) / 3600.0
            decay = time_decay_bonus(hours)
            edge = base_edge + decay

            if edge < self.min_edge:
                continue

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
                ticker=ticker, side=Side.NO, action=Action.BUY,
                order_type=OrderType.LIMIT, count=contracts,
                limit_price_cents=maker_price, time_in_force=TimeInForce.POST_ONLY,
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
                        "ensemble_used": ensemble_est is not None,
                        "xmarket_used": xm_delta is not None,
                    })
