"""
ScaledHarvesterV2 — a more aggressive PremiumHarvester.

Changes from baseline:
1. Wider band: YES 1-15¢ (NO 85-99¢) — bias stays positive through 15¢
2. Bankroll-aware Kelly sizing instead of fixed contracts
3. Lower cash reserve (25% vs 50%)
4. Higher position limit (50 vs 20)
5. Position topping: adds to existing positions when edge is large
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from oddpool_bench import Agent, AgentContext, Order, Side, Action, OrderType


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


class ScaledHarvesterV2(Agent):
    """
    Scaled PremiumHarvester with wider band, Kelly sizing, and position topping.
    """

    def __init__(
        self,
        min_no_price: int = 81,    # YES up to 19¢ — bias stays positive through ~20¢
        max_no_price: int = 99,
        min_edge: float = 0.02,    # slightly higher floor to filter noise
        max_position_per_market: int = 30,
        max_total_positions: int = 50,
        kelly_fraction: float = 0.25,
        cash_reserve_pct: float = 0.25,   # keep 25% in reserve (was 50%)
        top_up_edge_threshold: float = 0.05,  # add to position if edge > this
    ):
        self.min_no_price = min_no_price
        self.max_no_price = max_no_price
        self.min_edge = min_edge
        self.max_position_per_market = max_position_per_market
        self.max_total_positions = max_total_positions
        self.kelly_fraction = kelly_fraction
        self.cash_reserve_pct = cash_reserve_pct
        self.top_up_edge_threshold = top_up_edge_threshold
        self.positions: dict[str, int] = {}

    def on_episode_start(self, metadata: dict) -> None:
        self.positions = {}

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

            cost = market.no_best_ask or (100 - market.yes_best_bid)
            if cost is None or cost <= 0:
                continue

            already_held = self.positions.get(ticker, 0)

            # New position
            if already_held == 0:
                if len(self.positions) >= self.max_total_positions:
                    continue

                contracts = kelly_size(
                    win_prob=no_actual,
                    price_cents=cost,
                    bankroll_cents=total_equity,
                    kelly_mult=self.kelly_fraction,
                    max_contracts=self.max_position_per_market,
                )
                if contracts < 1:
                    continue

                spend = cost * contracts
                if spend > deployable:
                    contracts = deployable // cost
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
                if result and not result.get("rejected"):
                    fill_info = result.get("fill")
                    if fill_info and fill_info["count"] > 0:
                        filled = fill_info["count"]
                        self.positions[ticker] = filled
                        deployable -= cost * filled

            # Top-up: add to existing position when edge is large
            elif edge >= self.top_up_edge_threshold:
                room = self.max_position_per_market - already_held
                if room <= 0:
                    continue

                add_contracts = kelly_size(
                    win_prob=no_actual,
                    price_cents=cost,
                    bankroll_cents=total_equity,
                    kelly_mult=self.kelly_fraction / 2,  # half-Kelly for top-ups
                    max_contracts=room,
                )
                if add_contracts < 1:
                    continue

                spend = cost * add_contracts
                if spend > deployable:
                    add_contracts = deployable // cost
                if add_contracts < 1:
                    continue

                order = Order(
                    ticker=ticker,
                    side=Side.NO,
                    action=Action.BUY,
                    order_type=OrderType.MARKET,
                    count=add_contracts,
                )
                result = ctx.place_order(order)
                if result and not result.get("rejected"):
                    fill_info = result.get("fill")
                    if fill_info and fill_info["count"] > 0:
                        filled = fill_info["count"]
                        self.positions[ticker] += filled
                        deployable -= cost * filled


if __name__ == "__main__":
    from oddpool_bench import BenchmarkHarness, SimulatorConfig
    from our_agents import PremiumHarvester  # baseline

    config = SimulatorConfig(
        agent_call_cadence_seconds=5.0,
        equity_sample_interval_seconds=30.0,
        verbose=False,
    )
    episodes_dir = Path(__file__).parent / "episodes"
    harness = BenchmarkHarness(episodes_dir, config)

    # Parameter sweep over the two most impactful levers
    configs = {
        "Baseline (93-99, fixed)": PremiumHarvester(
            min_no_price=93, max_no_price=99, min_edge=0.005,
            max_position_per_market=10, max_total_positions=20,
        ),
        "Scaled (85-99, Kelly 0.25)": ScaledHarvesterV2(
            min_no_price=85, kelly_fraction=0.25, cash_reserve_pct=0.25,
        ),
        "Scaled (90-99, Kelly 0.25)": ScaledHarvesterV2(
            min_no_price=90, kelly_fraction=0.25, cash_reserve_pct=0.25,
        ),
        "Scaled (85-99, Kelly 0.35)": ScaledHarvesterV2(
            min_no_price=85, kelly_fraction=0.35, cash_reserve_pct=0.20,
        ),
        "Scaled (85-99, Kelly 0.50)": ScaledHarvesterV2(
            min_no_price=85, kelly_fraction=0.50, cash_reserve_pct=0.15,
        ),
    }

    print(f"\n{'='*70}")
    print("  PREMIUM HARVESTER SCALING SWEEP")
    print(f"{'='*70}")
    print(f"{'Config':<32} {'PnL':>9} {'Contracts':>10} {'MaxDD':>7} {'Sharpe':>8}")
    print("-" * 70)

    best_pnl = None
    best_name = None

    for name, agent in configs.items():
        result = harness.run(agent)
        pnl = sum(r.total_pnl_cents for r in result.episode_results) / 100
        contracts = sum(r.total_contracts_traded for r in result.episode_results)
        max_dd = max(r.max_drawdown_pct for r in result.episode_results) * 100
        sharpe_vals = [r.sharpe_ratio for r in result.episode_results if r.sharpe_ratio is not None]
        sharpe = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0.0
        print(f"{name:<32} ${pnl:>8.2f} {contracts:>10} {max_dd:>6.1f}% {sharpe:>7.2f}")

        if best_pnl is None or pnl > best_pnl:
            best_pnl = pnl
            best_name = name
            best_result = result

    print(f"\n  Winner: {best_name}  (PnL ${best_pnl:.2f})")

    # Save best result
    output_dir = Path("results_scaled_harvester")
    output_dir.mkdir(exist_ok=True)
    best_result.save(output_dir / "summary.json")
    best_result.save_trades(output_dir / "trades.json")
    best_result.save_equity_csv(output_dir / "equity_curve.csv")
    try:
        best_result.save_equity_curve(output_dir / "equity_curve.png")
    except Exception as e:
        print(f"  (Skipping plot: {e})")
    print(f"  Saved results to {output_dir}/")
