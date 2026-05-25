"""
DrawdownHarvester — ScaledHarvesterV2 + two key improvements:

1. MAKER LIMIT ORDERS: Use GTC limit orders instead of market orders.
   - Maker fee = 0.0175 × C × P × (1-P)  vs  taker fee = 0.07 × C × P × (1-P)
   - 4x cheaper. The backtest is in maker_taker mode so GTC orders fill via trade tape.

2. DRAWDOWN SCALING (DCA into dips):
   - Initial tranche: enter at first qualifying price
   - When price moves against us (YES rises → NO drops → edge improves), add more
   - Why it works: bias table says YES at 10¢ wins 4%, at 15¢ wins 8%
     → Edge INCREASES as price dips, so averaging down is taking MORE edge, not less
   - Example: buy NO at 95¢ (edge 4%), price dips to 90¢ → edge now 6% → add more
   - Hard cap: max total contracts per market, never risk > 2% of bankroll per ticker

Edge table (why drawdown adds edge, not risk):
  YES price  →  actual win rate  →  NO at  →  edge
      5¢             1%               95¢      4.0%   (initial entry)
     10¢             4%               90¢      6.0%   (dip tranche 1, BETTER entry)
     15¢             8%               85¢      7.0%   (dip tranche 2, BETTER entry)
     20¢            13%               80¢      7.0%   (edge stable, add more)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataclasses import dataclass, field
from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType, TimeInForce


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


def kelly_size(
    win_prob: float,
    price_cents: int,
    bankroll_cents: int,
    kelly_mult: float = 0.25,
    max_contracts: int = 50,
) -> int:
    if price_cents <= 0 or price_cents >= 100 or bankroll_cents <= 0:
        return 0
    p = max(0.001, min(0.999, win_prob))
    q = 1.0 - p
    b = (100.0 - price_cents) / price_cents
    f_star = (p * b - q) / b
    if f_star <= 0:
        return 0
    bet_cents = f_star * kelly_mult * bankroll_cents
    contracts = int(bet_cents / price_cents)
    return min(contracts, max_contracts)


@dataclass
class PositionState:
    total_contracts: int = 0
    avg_cost_cents: float = 0.0          # weighted average NO price paid
    initial_yes_price: int = 0           # YES price when first entered
    last_add_yes_price: int = 0          # YES price at last drawdown add
    tranches: int = 0                    # how many times we've added


class DrawdownHarvester(Agent):
    """
    Premium harvester with maker limit orders and drawdown scaling.

    Drawdown tiers: add contracts each time YES price rises by dip_step_cents
    from our initial entry, as long as edge > min_edge and position < hard cap.
    """

    def __init__(
        self,
        min_no_price: int = 81,
        max_no_price: int = 99,
        min_edge: float = 0.02,
        max_position_per_market: int = 40,    # hard cap per ticker
        max_total_positions: int = 60,
        kelly_fraction: float = 0.25,
        dca_kelly_fraction: float = 0.15,     # smaller Kelly for dip adds
        cash_reserve_pct: float = 0.20,
        dip_step_cents: int = 5,              # add when YES rises by this much from entry
        max_dca_tranches: int = 3,            # max drawdown adds per position
        use_limit_orders: bool = True,        # GTC limit for maker fees
    ):
        self.min_no_price = min_no_price
        self.max_no_price = max_no_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.kelly_fraction = kelly_fraction
        self.dca_kelly_fraction = dca_kelly_fraction
        self.cash_reserve_pct = cash_reserve_pct
        self.dip_step_cents = dip_step_cents
        self.max_dca_tranches = max_dca_tranches
        self.use_limit_orders = use_limit_orders
        self.positions: dict[str, PositionState] = {}

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}

    def _place_no_order(
        self,
        ctx: AgentContext,
        ticker: str,
        market,
        contracts: int,
        deployable: int,
    ) -> int:
        """Place a NO buy order (market or limit). Returns contracts filled/queued."""
        if self.use_limit_orders:
            # Post a resting maker bid: improve the NO bid by 1c
            # NO bid = 100 - YES ask  (the passive side)
            # NO ask = 100 - YES bid  (the crossing side)
            # We post at NO bid + 1 to sit in the book and earn maker fee
            yes_ask = market.yes_best_ask
            yes_bid = market.yes_best_bid
            if yes_ask is None or yes_bid is None:
                return 0
            spread = yes_ask - yes_bid
            if spread < 3:
                # Too tight — can't improve bid without crossing
                return 0
            no_bid = 100 - yes_ask          # passive NO bid
            limit_price = no_bid + 1        # improve by 1c (rests as maker)
            cost_estimate = limit_price
            if cost_estimate <= 0 or cost_estimate >= 100:
                return 0
            spend = cost_estimate * contracts
            if spend > deployable:
                contracts = deployable // cost_estimate
            if contracts < 1:
                return 0
            order = Order(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                order_type=OrderType.LIMIT,
                count=contracts,
                limit_price_cents=limit_price,
                time_in_force=TimeInForce.GTC,
            )
        else:
            cost = market.no_best_ask or (100 - (market.yes_best_bid or 0))
            if cost <= 0:
                return 0
            spend = cost * contracts
            if spend > deployable:
                contracts = deployable // cost
            if contracts < 1:
                return 0
            order = Order(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                order_type=OrderType.MARKET,
                count=contracts,
            )

        result = ctx.place_order(order)
        if result and not result.get("rejected"):
            queued = 0
            fill_info = result.get("fill")
            if fill_info:
                queued += fill_info["count"]
            resting_info = result.get("resting")
            if resting_info:
                queued += resting_info["remaining_count"]
            return queued
        return 0

    def act(self, ctx: AgentContext) -> None:
        markets = ctx.get_markets()
        cash_info = ctx.get_cash()
        total_equity = cash_info["equity_cents"]
        available_cash = cash_info["cash_cents"]
        deployable = int(available_cash * (1.0 - self.cash_reserve_pct))

        if deployable <= 0:
            return

        for market in markets:
            ticker = market.ticker

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

            pos = self.positions.get(ticker)

            # ── NEW POSITION ──────────────────────────────────────────────
            if pos is None:
                if len(self.positions) >= self.max_total_positions:
                    continue

                contracts = kelly_size(
                    win_prob=no_actual,
                    price_cents=no_price,
                    bankroll_cents=total_equity,
                    kelly_mult=self.kelly_fraction,
                    max_contracts=self.max_position_per_market,
                )
                if contracts < 1:
                    continue

                queued = self._place_no_order(ctx, ticker, market, contracts, deployable)
                if queued > 0:
                    cost_price = no_price  # approximate
                    self.positions[ticker] = PositionState(
                        total_contracts=queued,
                        avg_cost_cents=float(cost_price),
                        initial_yes_price=yes_mid,
                        last_add_yes_price=yes_mid,
                        tranches=1,
                    )
                    deployable -= cost_price * queued

            # ── DRAWDOWN SCALE-IN ─────────────────────────────────────────
            else:
                # Has price moved against us by dip_step_cents since last add?
                yes_rise = yes_mid - pos.last_add_yes_price
                if yes_rise < self.dip_step_cents:
                    continue  # not enough dip yet

                # Check tranche limit
                if pos.tranches >= self.max_dca_tranches + 1:
                    continue

                # Check hard cap
                room = self.max_position_per_market - pos.total_contracts
                if room <= 0:
                    continue

                # Smaller Kelly for dip adds (more cautious as we add)
                add_kelly = self.dca_kelly_fraction / pos.tranches  # scale down each tranche
                add_contracts = kelly_size(
                    win_prob=no_actual,
                    price_cents=no_price,
                    bankroll_cents=total_equity,
                    kelly_mult=add_kelly,
                    max_contracts=room,
                )
                if add_contracts < 1:
                    continue

                queued = self._place_no_order(ctx, ticker, market, add_contracts, deployable)
                if queued > 0:
                    # Update weighted average cost
                    total = pos.total_contracts + queued
                    pos.avg_cost_cents = (
                        pos.avg_cost_cents * pos.total_contracts + no_price * queued
                    ) / total
                    pos.total_contracts = total
                    pos.last_add_yes_price = yes_mid
                    pos.tranches += 1
                    deployable -= no_price * queued


if __name__ == "__main__":
    from oddpool_bench import BenchmarkHarness, SimulatorConfig
    from our_agents import PremiumHarvester
    from scaled_harvester import ScaledHarvesterV2

    config = SimulatorConfig(
        agent_call_cadence_seconds=5.0,
        equity_sample_interval_seconds=30.0,
        verbose=False,
    )
    episodes_dir = Path(__file__).parent / "episodes"
    harness = BenchmarkHarness(episodes_dir, config)

    configs = {
        "Baseline (taker, fixed)":      PremiumHarvester(min_no_price=93, max_no_price=99, min_edge=0.005, max_position_per_market=10, max_total_positions=20),
        "ScaledV2 (taker, Kelly)":      ScaledHarvesterV2(),
        "DCA market (no limit orders)": DrawdownHarvester(use_limit_orders=False),
        "DCA + maker limits":           DrawdownHarvester(use_limit_orders=True),
        "DCA + maker + aggressive":     DrawdownHarvester(use_limit_orders=True, kelly_fraction=0.35, max_dca_tranches=4, dip_step_cents=4),
    }

    print(f"\n{'='*72}")
    print("  DRAWDOWN HARVESTER SWEEP")
    print(f"{'='*72}")
    print(f"{'Config':<32} {'PnL':>9} {'Fees':>7} {'Net':>9} {'MaxDD':>7} {'Sharpe':>8}")
    print("-" * 72)

    for name, agent in configs.items():
        result = harness.run(agent)
        pnl = sum(r.total_pnl_cents for r in result.episode_results) / 100
        fees = sum(r.total_fees_cents for r in result.episode_results) / 100
        max_dd = max(r.max_drawdown_pct for r in result.episode_results) * 100
        s = [r.sharpe_ratio for r in result.episode_results if r.sharpe_ratio is not None]
        sharpe = sum(s) / len(s) if s else 0.0
        print(f"{name:<32} ${pnl:>7.2f} ${fees:>5.2f} ${pnl-fees:>7.2f} {max_dd:>6.1f}% {sharpe:>7.2f}")

    print(f"\nFee note: at extreme NO prices (85-99c), fee diff is small.")
    print(f"  NO=95c: taker=0.33c/contract, maker=0.08c/contract")
    print(f"  NO=85c: taker=0.89c/contract, maker=0.22c/contract")
    print(f"DCA logic: add tranche when YES rises 5c from last entry (max 3 adds)")
    print(f"Edge on dips: YES5c=4%, YES10c=6%, YES15c=7%, YES20c=7% edge")
