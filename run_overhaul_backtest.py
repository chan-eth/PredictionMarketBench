#!/usr/bin/env python3
"""
Backtest the post-overhaul strategy vs the original.

A/B comparison:
  - PremiumHarvesterOld: taker orders, YES 1-7c, 4-row FLB, no time-decay
  - PremiumHarvesterV2:  maker-only (POST_ONLY), YES 1-20c, 14-row FLB, time-decay

Usage:
    cd backtest-framework
    python run_overhaul_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import BenchmarkHarness, SimulatorConfig
from overhauled_agents import PremiumHarvesterV2, PremiumHarvesterOld
from examples.example_agents import PassiveAgent


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

    agents = {
        "Passive (baseline)": PassiveAgent(),
        "PH-Old (taker, YES 1-7c)": PremiumHarvesterOld(
            min_no_price=93,
            max_no_price=99,
            min_edge=0.005,
            max_position_per_market=10,
            max_total_positions=20,
        ),
        "PH-V2 (maker, YES 1-20c)": PremiumHarvesterV2(
            min_yes_price=1,
            max_yes_price=20,
            min_edge=0.005,
            max_position_per_market=10,
            max_total_positions=20,
            min_spread=3,
        ),
    }

    results = {}
    for name, agent in agents.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        result = harness.run(agent)
        result.print_summary()
        results[name] = result

        output_dir = Path(
            f"results_{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}"
        )
        output_dir.mkdir(exist_ok=True)
        result.save(output_dir / "summary.json")
        result.save_trades(output_dir / "trades.json")
        result.save_equity_csv(output_dir / "equity_curve.csv")
        try:
            result.save_equity_curve(output_dir / "equity_curve.png")
        except (ImportError, Exception) as e:
            print(f"  (Skipping plot: {e})")

    # Comparison table
    print(f"\n{'='*70}")
    print("  OVERHAUL A/B COMPARISON")
    print(f"{'='*70}")
    print(
        f"{'Strategy':<30} {'PnL':>10} {'Fees':>8} {'Trades':>8} "
        f"{'Sharpe':>8} {'MaxDD':>8}"
    )
    print("-" * 70)

    for name, result in results.items():
        agg = result.aggregate
        pnl = agg.get("total_pnl_cents", {}).get("sum", 0) / 100.0
        fees = sum(r.total_fees_cents for r in result.episode_results) / 100.0
        trades = sum(r.total_contracts_traded for r in result.episode_results)
        sharpe = agg.get("sharpe_ratio", {}).get("mean", 0) or 0
        dd = agg.get("max_drawdown_pct", {}).get("max", 0) or 0
        print(
            f"{name:<30} ${pnl:>8.2f} ${fees:>6.2f} {trades:>8} "
            f"{sharpe:>7.2f} {dd:>7.2f}%"
        )

    print()
    print("Key metrics to compare:")
    print("  - PnL: V2 should be higher (wider range catches more edge)")
    print("  - Fees: V2 should be MUCH lower (1.75% maker vs 7% taker)")
    print("  - Sharpe: V2 should be higher (less fee drag)")
    print("  - MaxDD: V2 should be comparable or lower")


if __name__ == "__main__":
    main()
