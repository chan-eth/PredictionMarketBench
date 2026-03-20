#!/usr/bin/env python3
"""
Backtest: SignalFusion V2 ablation study.

Compares 7 agents to isolate the contribution of each improvement layer:
- Maker execution (fee savings)
- Hard gate (rule-based filtering)
- ML confidence gate (learned filtering)

Usage:
    cd backtest-framework
    python run_sfv2_backtest.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oddpool_bench import BenchmarkHarness, SimulatorConfig
from overhauled_agents import PremiumHarvesterV2
from signal_agents import SignalFusionAgent
from signal_adjustment_agents import PHWithSignalAdjustment
from signal_fusion_v2_agents import SignalFusionV2, SignalFusionV2HardOnly, SignalFusionV2MakerOnly
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
        "1. Passive":               PassiveAgent(),
        "2. PH-V2":                 PremiumHarvesterV2(max_total_positions=20),
        "3. PH+Adjust":             PHWithSignalAdjustment(signal_adjustment_max=0.03),
        "4. SF-Old":                SignalFusionAgent(),
        "5. SFv2-MakerOnly":        SignalFusionV2MakerOnly(),
        "6. SFv2-HardGate":         SignalFusionV2HardOnly(),
        "7. SFv2-Full":             SignalFusionV2(),
    }

    results = {}
    agent_stats = {}

    for name, agent in agents.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        result = harness.run(agent)
        result.print_summary()
        results[name] = result

        # Collect V2 diagnostics
        stats = {}
        if hasattr(agent, "signals_fired"):
            stats["signals_fired"] = agent.signals_fired
            stats["signals_blocked_hard"] = agent.signals_blocked_hard
            stats["signals_blocked_ml"] = agent.signals_blocked_ml
            stats["block_rate"] = (
                (agent.signals_blocked_hard + agent.signals_blocked_ml) / max(agent.signals_fired, 1)
            )
        if hasattr(agent, "hard_gate") and agent.hard_gate:
            stats["hard_gate_reasons"] = agent.hard_gate.block_stats
        agent_stats[name] = stats

        # Save per-agent results
        safe_name = name.split(". ", 1)[-1].lower().replace("+", "_").replace("-", "_").replace(" ", "_")
        output_dir = Path(f"results_sfv2/{safe_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.save(output_dir / "summary.json")
        result.save_trades(output_dir / "trades.json")
        result.save_equity_csv(output_dir / "equity_curve.csv")
        try:
            result.save_equity_curve(output_dir / "equity_curve.png")
        except (ImportError, Exception) as e:
            print(f"  (Skipping plot: {e})")

    # ---- Comparison table ----
    print(f"\n{'='*100}")
    print("  SIGNAL FUSION V2 — ABLATION RESULTS")
    print(f"{'='*100}")
    print(f"{'Strategy':<22} {'PnL ($)':>10} {'Trades':>8} {'Sharpe':>8} {'MaxDD':>8} {'Fees':>8} {'Blocked':>10}")
    print("-" * 100)

    for name, result in results.items():
        d = result.to_dict()
        eps = d.get("episodes", [])
        agg = d.get("aggregate", {})

        total_pnl = agg.get("total_pnl_cents", {}).get("sum", 0) or 0
        total_trades = sum(ep.get("total_contracts_traded", 0) for ep in eps)
        total_fees = sum(ep.get("total_fees_cents", 0) for ep in eps)
        max_dd = agg.get("max_drawdown_pct", {}).get("max", 0) or 0
        avg_sharpe = agg.get("sharpe_ratio", {}).get("mean", 0) or 0

        stats = agent_stats.get(name, {})
        blocked = stats.get("signals_blocked_hard", 0) + stats.get("signals_blocked_ml", 0)
        block_str = f"{blocked}" if blocked > 0 else "-"

        print(
            f"{name:<22} ${total_pnl/100:>8.2f} {total_trades:>8} "
            f"{avg_sharpe:>7.2f} {max_dd*100:>6.1f}% ${total_fees/100:>6.2f} {block_str:>10}"
        )

    # ---- V2 Diagnostics ----
    print(f"\n{'='*70}")
    print("  V2 GATE DIAGNOSTICS")
    print(f"{'='*70}")
    for name, stats in agent_stats.items():
        if not stats:
            continue
        print(f"\n  {name}:")
        for k, v in stats.items():
            if k == "hard_gate_reasons" and v:
                print(f"    {k}:")
                for reason, count in sorted(v.items(), key=lambda x: -x[1]):
                    print(f"      {reason}: {count}")
            else:
                print(f"    {k}: {v}")

    # ---- Ablation Analysis ----
    print(f"\n{'='*70}")
    print("  ABLATION ANALYSIS")
    print(f"{'='*70}")

    def get_pnl(name):
        d = results[name].to_dict()
        return d.get("aggregate", {}).get("total_pnl_cents", {}).get("sum", 0) or 0

    def get_fees(name):
        d = results[name].to_dict()
        return sum(ep.get("total_fees_cents", 0) for ep in d.get("episodes", []))

    sf_old_pnl = get_pnl("4. SF-Old")
    sf_old_fees = get_fees("4. SF-Old")
    maker_pnl = get_pnl("5. SFv2-MakerOnly")
    maker_fees = get_fees("5. SFv2-MakerOnly")
    hard_pnl = get_pnl("6. SFv2-HardGate")
    hard_fees = get_fees("6. SFv2-HardGate")
    full_pnl = get_pnl("7. SFv2-Full")
    full_fees = get_fees("7. SFv2-Full")
    ph_pnl = get_pnl("2. PH-V2")

    print(f"  Maker execution value:  ${(maker_pnl - sf_old_pnl)/100:+.2f} PnL, ${(sf_old_fees - maker_fees)/100:.2f} fee savings")
    print(f"  Hard gate value:        ${(hard_pnl - maker_pnl)/100:+.2f} PnL (on top of maker)")
    print(f"  ML gate value:          ${(full_pnl - hard_pnl)/100:+.2f} PnL (on top of hard gate)")
    print(f"  Total V2 improvement:   ${(full_pnl - sf_old_pnl)/100:+.2f} PnL vs old SignalFusion")
    print(f"  V2 vs PH standalone:    ${(full_pnl - ph_pnl)/100:+.2f} PnL")

    # ---- Validation checks ----
    print(f"\n{'='*70}")
    print("  VALIDATION CHECKS")
    print(f"{'='*70}")

    checks = [
        ("SFv2-Full >= Passive",
         full_pnl >= 0,
         f"${full_pnl/100:.2f}"),
        ("SFv2-Full > SF-Old",
         full_pnl > sf_old_pnl,
         f"${full_pnl/100:.2f} vs ${sf_old_pnl/100:.2f}"),
        ("SFv2-Full fees < $5",
         full_fees < 500,
         f"${full_fees/100:.2f}"),
        ("Maker saves fees vs old",
         maker_fees < sf_old_fees,
         f"${maker_fees/100:.2f} vs ${sf_old_fees/100:.2f}"),
    ]

    all_pass = True
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}: {detail}")

    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # Save diagnostics
    diag_path = Path("results_sfv2/diagnostics.json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump({
            "episodes": episodes,
            "agent_stats": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in agent_stats.items()},
            "ablation": {
                "maker_pnl_delta": (maker_pnl - sf_old_pnl) / 100,
                "hard_gate_pnl_delta": (hard_pnl - maker_pnl) / 100,
                "ml_gate_pnl_delta": (full_pnl - hard_pnl) / 100,
                "total_improvement": (full_pnl - sf_old_pnl) / 100,
            },
            "validation": {label: {"passed": passed, "detail": detail} for label, passed, detail in checks},
        }, f, indent=2, default=str)
    print(f"\nDiagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
