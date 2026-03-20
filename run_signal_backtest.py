#!/usr/bin/env python3
"""
Backtest the signal intelligence pipeline against historical Kalshi episodes.

Runs all signal agents + baselines and produces comparison metrics.

Usage:
    cd backtest-framework
    pip install -e .
    python run_signal_backtest.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import BenchmarkHarness, SimulatorConfig
from our_agents import PremiumHarvester, MomentumSniper, CombinedAgent
from signal_agents import TapeScannerAgent, SignalFusionAgent, FullStackAgent
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

    # All agents to compare
    agents = {
        "Passive (baseline)":   PassiveAgent(),
        "PremiumHarvester":     PremiumHarvester(max_position_per_market=10, max_total_positions=20),
        "MomentumSniper":       MomentumSniper(lookback=15, entry_threshold_cents=3, max_position=5),
        "CombinedAgent":        CombinedAgent(),
        "TapeScanner":          TapeScannerAgent(max_positions=5, contracts_per_trade=3),
        "SignalFusion":         SignalFusionAgent(),
        "FullStack":            FullStackAgent(),
    }

    results = {}
    signal_diagnostics = {}

    for name, agent in agents.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        result = harness.run(agent)
        result.print_summary()
        results[name] = result

        # Save results
        output_dir = Path(f"results_signals/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.save(output_dir / "summary.json")
        result.save_trades(output_dir / "trades.json")
        result.save_equity_csv(output_dir / "equity_curve.csv")
        try:
            result.save_equity_curve(output_dir / "equity_curve.png")
        except (ImportError, Exception) as e:
            print(f"  (Skipping plot: {e})")

        # Extract signal diagnostics from fusion agents
        if hasattr(agent, "signal_log"):
            signal_diagnostics[name] = {
                "total_signals": len(agent.signal_log),
                "by_type": _count_by(agent.signal_log, "type"),
                "by_direction": _count_by(agent.signal_log, "direction"),
            }
        elif hasattr(agent, "fusion") and hasattr(agent.fusion, "signal_log"):
            signal_diagnostics[name] = {
                "total_signals": len(agent.fusion.signal_log),
                "by_type": _count_by(agent.fusion.signal_log, "type"),
                "by_direction": _count_by(agent.fusion.signal_log, "direction"),
            }

    # Print comparison table
    print(f"\n{'='*85}")
    print("  STRATEGY COMPARISON — Signal Intelligence Pipeline")
    print(f"{'='*85}")
    print(f"{'Strategy':<25} {'PnL ($)':>10} {'Trades':>8} {'Sharpe':>8} {'MaxDD':>8} {'Fees':>8}")
    print("-" * 85)

    for name, result in results.items():
        d = result.to_dict()
        eps = d.get("episodes", [])
        agg = d.get("aggregate", {})

        total_pnl = agg.get("total_pnl_cents", {}).get("sum", 0) or 0
        total_trades = sum(ep.get("total_contracts_traded", 0) for ep in eps)
        total_fees = sum(ep.get("total_fees_cents", 0) for ep in eps)
        max_dd = agg.get("max_drawdown_pct", {}).get("max", 0) or 0
        avg_sharpe = agg.get("sharpe_ratio", {}).get("mean", 0) or 0

        print(
            f"{name:<25} ${total_pnl/100:>8.2f} {total_trades:>8} "
            f"{avg_sharpe:>7.2f} {max_dd*100:>6.1f}% ${total_fees/100:>6.2f}"
        )

    # Print signal diagnostics
    if signal_diagnostics:
        print(f"\n{'='*70}")
        print("  SIGNAL DIAGNOSTICS")
        print(f"{'='*70}")
        for name, diag in signal_diagnostics.items():
            print(f"\n  {name}:")
            print(f"    Total signals: {diag['total_signals']}")
            print(f"    By type: {diag['by_type']}")
            print(f"    By direction: {diag['by_direction']}")

    # Save diagnostics
    diag_path = Path("results_signals/diagnostics.json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump({
            "run_time": datetime.utcnow().isoformat(),
            "episodes": episodes,
            "signal_diagnostics": signal_diagnostics,
        }, f, indent=2, default=str)
    print(f"\nDiagnostics saved to {diag_path}")


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        val = item.get(key, "unknown")
        if isinstance(val, list):
            val = ",".join(sorted(val))
        counts[val] = counts.get(val, 0) + 1
    return counts


if __name__ == "__main__":
    main()
