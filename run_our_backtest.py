#!/usr/bin/env python3
"""
Run our trading strategies against Kalshi historical data.

Usage:
    cd backtest-framework
    pip install -e .
    python run_our_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import BenchmarkHarness, SimulatorConfig
from our_agents import PremiumHarvester, MomentumSniper, CombinedAgent
from examples.example_agents import PassiveAgent, RandomAgent

def main():
    config = SimulatorConfig(
        agent_call_cadence_seconds=5.0,
        equity_sample_interval_seconds=30.0,
        verbose=True,
    )

    episodes_dir = Path(__file__).parent / "episodes"
    harness = BenchmarkHarness(episodes_dir, config)

    episodes = harness.list_episodes()
    print(f"\nAvailable episodes: {episodes}")

    if not episodes:
        print("No episodes found!")
        return

    # Our strategies + baselines
    agents = {
        "Passive (baseline)": PassiveAgent(),
        "Random (baseline)": RandomAgent(trade_probability=0.05),
        "PremiumHarvester": PremiumHarvester(
            min_no_price=93,
            max_no_price=99,
            min_edge=0.005,
            max_position_per_market=10,
            max_total_positions=20,
        ),
        "MomentumSniper": MomentumSniper(
            lookback=15,
            entry_threshold_cents=3,
            exit_threshold_cents=2,
            max_position=5,
        ),
        "Combined": CombinedAgent(),
    }

    results = {}
    for name, agent in agents.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        result = harness.run(agent)
        result.print_summary()
        results[name] = result

        # Save results
        output_dir = Path(f"results_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}")
        output_dir.mkdir(exist_ok=True)
        result.save(output_dir / "summary.json")
        result.save_trades(output_dir / "trades.json")
        result.save_equity_csv(output_dir / "equity_curve.csv")
        try:
            result.save_equity_curve(output_dir / "equity_curve.png")
        except (ImportError, Exception) as e:
            print(f"  (Skipping plot: {e})")

    # Print comparison
    print(f"\n{'='*60}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Strategy':<25} {'PnL':>10} {'Trades':>8} {'Win%':>8} {'Sharpe':>8}")
    print("-" * 60)
    for name, result in results.items():
        summary = result.get_summary()
        pnl = summary.get("total_pnl_cents", 0) / 100.0
        trades = summary.get("total_trades", 0)
        wins = summary.get("winning_trades", 0)
        win_pct = (wins / trades * 100) if trades > 0 else 0
        sharpe = summary.get("sharpe_ratio", 0)
        print(f"{name:<25} ${pnl:>8.2f} {trades:>8} {win_pct:>7.1f}% {sharpe:>7.2f}")


if __name__ == "__main__":
    main()
