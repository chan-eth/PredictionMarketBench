#!/usr/bin/env python3
"""
Backtest: Strategy expansion — Bundle Arb + LLM Ensemble + Fuzzy Cross-Market.

Validates that the three new strategies improve outcomes vs PH standalone:
- BundleArb: captures YES+NO < 100c gaps (additive profit, zero risk)
- PHWithEnsemble: better probability estimates → same/better PH trades
- PHWithFuzzyXMarket: more cross-market signals → better PH adjustments
- FullExpansion: all combined

Expected results:
- BundleArb captures some free arb (may be 0 if episodes lack gaps)
- PHWithEnsemble >= PH standalone (consensus gating reduces noise)
- PHWithFuzzyXMarket >= PH standalone (more signal coverage)
- FullExpansion >= PH standalone (combined benefits)

Usage:
    cd backtest-framework
    python run_expansion_backtest.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import BenchmarkHarness, SimulatorConfig
from overhauled_agents import PremiumHarvesterV2
from expansion_agents import (
    BundleArbAgent,
    PHWithEnsembleSignals,
    PHWithFuzzyXMarket,
    FullExpansionAgent,
)
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
        "PH-V2 (control)":         PremiumHarvesterV2(max_total_positions=20),
        "BundleArb (standalone)":   BundleArbAgent(min_edge_cents=3, max_contracts=25),
        "PH+Ensemble":              PHWithEnsembleSignals(
                                        n_models=2,
                                        model_noise_std=0.08,
                                        consensus_spread=0.15,
                                        adjustment_weight=0.3,
                                    ),
        "PH+FuzzyXMarket":         PHWithFuzzyXMarket(
                                        match_probability=0.6,
                                        min_discrepancy=0.08,
                                        signal_adjustment_max=0.03,
                                    ),
        "FullExpansion (all 3)":    FullExpansionAgent(),
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
        output_dir = Path(f"results_expansion/{safe_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.save(output_dir / "summary.json")
        result.save_trades(output_dir / "trades.json")
        result.save_equity_csv(output_dir / "equity_curve.csv")
        try:
            result.save_equity_curve(output_dir / "equity_curve.png")
        except (ImportError, Exception) as e:
            print(f"  (Skipping plot: {e})")

        # Diagnostics per agent type
        if hasattr(agent, "arb_log") and isinstance(agent, BundleArbAgent):
            diagnostics[name] = {
                "type": "bundle_arb",
                "opportunities_found": len(agent.arb_log),
                "total_arb_profit_cents": sum(a.get("profit_cents", 0) for a in agent.arb_log),
                "avg_gap": round(
                    sum(a["gap"] for a in agent.arb_log) / max(len(agent.arb_log), 1), 1
                ),
            }
        elif hasattr(agent, "ensemble_log"):
            consensus_count = sum(1 for e in agent.ensemble_log if e["consensus"])
            total = max(len(agent.ensemble_log), 1)
            diagnostics[name] = {
                "type": "ensemble",
                "total_evaluations": len(agent.ensemble_log),
                "consensus_rate": f"{consensus_count}/{total} ({consensus_count/total*100:.0f}%)",
                "trades_with_ensemble": sum(1 for t in agent.trade_log if t.get("ensemble_used")),
                "trades_total": len(agent.trade_log),
            }
        elif hasattr(agent, "xmarket_log"):
            diagnostics[name] = {
                "type": "cross_market_fuzzy",
                "xmarket_signals": len(agent.xmarket_log),
                "trades_with_xmarket": sum(1 for t in agent.trade_log if t.get("xmarket_signal")),
                "trades_total": len(agent.trade_log),
            }
        elif isinstance(agent, FullExpansionAgent):
            diagnostics[name] = {
                "type": "full_expansion",
                "arb_opportunities": len(agent.arb_log),
                "ph_trades": len(agent.trade_log),
                "trades_with_ensemble": sum(1 for t in agent.trade_log if t.get("ensemble_used")),
                "trades_with_xmarket": sum(1 for t in agent.trade_log if t.get("xmarket_used")),
            }

    # ---- Comparison table ----
    print(f"\n{'='*90}")
    print("  EXPANSION STRATEGY COMPARISON")
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

    # ---- Diagnostics ----
    if diagnostics:
        print(f"\n{'='*70}")
        print("  STRATEGY DIAGNOSTICS")
        print(f"{'='*70}")
        for name, diag in diagnostics.items():
            print(f"\n  {name}:")
            for k, v in diag.items():
                print(f"    {k}: {v}")

    # ---- Validation ----
    print(f"\n{'='*70}")
    print("  VALIDATION CHECKS")
    print(f"{'='*70}")

    def get_pnl(name):
        d = results[name].to_dict()
        return d.get("aggregate", {}).get("total_pnl_cents", {}).get("sum", 0) or 0

    ph_pnl = get_pnl("PH-V2 (control)")
    arb_pnl = get_pnl("BundleArb (standalone)")
    ensemble_pnl = get_pnl("PH+Ensemble")
    fuzzy_pnl = get_pnl("PH+FuzzyXMarket")
    full_pnl = get_pnl("FullExpansion (all 3)")

    checks = [
        ("BundleArb >= 0 (zero-risk)",
         arb_pnl >= 0,
         f"${arb_pnl/100:.2f}"),
        ("PH+Ensemble >= Passive",
         ensemble_pnl >= 0,
         f"${ensemble_pnl/100:.2f}"),
        ("PH+Ensemble ~= PH standalone",
         abs(ensemble_pnl - ph_pnl) < 500,  # within $5
         f"${ensemble_pnl/100:.2f} vs ${ph_pnl/100:.2f}"),
        ("PH+FuzzyXMarket >= Passive",
         fuzzy_pnl >= 0,
         f"${fuzzy_pnl/100:.2f}"),
        ("FullExpansion >= PH standalone",
         full_pnl >= ph_pnl - 100,  # within $1 tolerance
         f"${full_pnl/100:.2f} vs ${ph_pnl/100:.2f}"),
    ]

    all_pass = True
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}: {detail}")

    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # Save
    diag_path = Path("results_expansion/diagnostics.json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump({
            "run_time": datetime.now(timezone.utc).isoformat(),
            "episodes": episodes,
            "diagnostics": diagnostics,
            "validation": {label: {"passed": passed, "detail": detail} for label, passed, detail in checks},
        }, f, indent=2, default=str)
    print(f"\nDiagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
