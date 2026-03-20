#!/usr/bin/env python3
"""
Backtest: Signal-as-adjustment vs old signal-as-trigger architecture.

Validates that retiring signal triggers and wiring signals as probability
inputs to PremiumHarvester is at least as good as PH standalone, and
strictly better than the old signal fusion approach.

Expected results:
- PHWithSignalAdjustment >= PH V2 standalone (same trades, better prob estimates)
- PHWithSignalAdjustment >> FullStack / SignalFusion (no extra fee bleed)
- PHWithSignalAdjustment >> Passive baseline

Usage:
    cd backtest-framework
    pip install -e .
    python run_adjustment_backtest.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import BenchmarkHarness, SimulatorConfig
from overhauled_agents import PremiumHarvesterV2
from signal_agents import SignalFusionAgent, FullStackAgent
from signal_adjustment_agents import PHWithSignalAdjustment
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
        "Passive (baseline)":       PassiveAgent(),
        "PH-V2 (standalone)":       PremiumHarvesterV2(max_total_positions=20),
        "PH+SignalAdjust (NEW)":    PHWithSignalAdjustment(
                                        signal_adjustment_max=0.03,
                                        scorer_min_composite=0.5,
                                    ),
        "SignalFusion (old)":       SignalFusionAgent(),
        "FullStack (old)":          FullStackAgent(),
    }

    results = {}
    diagnostics = {}

    for name, agent in agents.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        result = harness.run(agent)
        result.print_summary()
        results[name] = result

        # Save per-agent results
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        output_dir = Path(f"results_adjustment/{safe_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.save(output_dir / "summary.json")
        result.save_trades(output_dir / "trades.json")
        result.save_equity_csv(output_dir / "equity_curve.csv")
        try:
            result.save_equity_curve(output_dir / "equity_curve.png")
        except (ImportError, Exception) as e:
            print(f"  (Skipping plot: {e})")

        # Extract adjustment diagnostics
        if hasattr(agent, "adjustment_log"):
            adj_log = agent.adjustment_log
            bullish = sum(1 for a in adj_log if a["delta"] > 0)
            bearish = sum(1 for a in adj_log if a["delta"] < 0)
            diagnostics[name] = {
                "total_adjustments": len(adj_log),
                "bullish": bullish,
                "bearish": bearish,
                "balance_ratio": f"{bullish}/{bearish}" if bearish > 0 else f"{bullish}/0",
                "avg_delta": round(sum(a["delta"] for a in adj_log) / max(len(adj_log), 1), 4),
                "trades_placed": len(agent.trade_log),
                "trades_with_adjustment": sum(1 for t in agent.trade_log if t["adjustment"] != 0.0),
            }
        elif hasattr(agent, "fusion") and hasattr(agent.fusion, "signal_log"):
            diagnostics[name] = {
                "total_signals": len(agent.fusion.signal_log),
                "mode": "old (signal-as-trigger)",
            }

    # ---- Comparison table ----
    print(f"\n{'='*90}")
    print("  COMPARISON: Signal-as-Adjustment vs Signal-as-Trigger")
    print(f"{'='*90}")
    print(f"{'Strategy':<28} {'PnL ($)':>10} {'Trades':>8} {'Sharpe':>8} {'MaxDD':>8} {'Fees':>8}")
    print("-" * 90)

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
            f"{name:<28} ${total_pnl/100:>8.2f} {total_trades:>8} "
            f"{avg_sharpe:>7.2f} {max_dd*100:>6.1f}% ${total_fees/100:>6.2f}"
        )

    # ---- Adjustment diagnostics ----
    if diagnostics:
        print(f"\n{'='*70}")
        print("  SIGNAL ADJUSTMENT DIAGNOSTICS")
        print(f"{'='*70}")
        for name, diag in diagnostics.items():
            print(f"\n  {name}:")
            for k, v in diag.items():
                print(f"    {k}: {v}")

    # ---- Validation checks ----
    print(f"\n{'='*70}")
    print("  VALIDATION CHECKS")
    print(f"{'='*70}")

    def get_pnl(name):
        d = results[name].to_dict()
        return d.get("aggregate", {}).get("total_pnl_cents", {}).get("sum", 0) or 0

    ph_baseline_pnl = get_pnl("PH-V2 (standalone)")
    ph_adjusted_pnl = get_pnl("PH+SignalAdjust (NEW)")
    fusion_old_pnl = get_pnl("SignalFusion (old)")
    fullstack_old_pnl = get_pnl("FullStack (old)")

    checks = [
        ("PH+Adjust >= Passive",
         ph_adjusted_pnl >= 0,
         f"${ph_adjusted_pnl/100:.2f}"),
        ("PH+Adjust >= PH standalone",
         ph_adjusted_pnl >= ph_baseline_pnl - 50,  # within 50c tolerance
         f"${ph_adjusted_pnl/100:.2f} vs ${ph_baseline_pnl/100:.2f}"),
        ("PH+Adjust > old SignalFusion",
         ph_adjusted_pnl > fusion_old_pnl,
         f"${ph_adjusted_pnl/100:.2f} vs ${fusion_old_pnl/100:.2f}"),
        ("PH+Adjust > old FullStack",
         ph_adjusted_pnl > fullstack_old_pnl,
         f"${ph_adjusted_pnl/100:.2f} vs ${fullstack_old_pnl/100:.2f}"),
    ]

    all_pass = True
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}: {detail}")

    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # Save diagnostics
    diag_path = Path("results_adjustment/diagnostics.json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump({
            "run_time": datetime.utcnow().isoformat(),
            "episodes": episodes,
            "diagnostics": diagnostics,
            "validation": {label: {"passed": passed, "detail": detail} for label, passed, detail in checks},
        }, f, indent=2, default=str)
    print(f"\nDiagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
