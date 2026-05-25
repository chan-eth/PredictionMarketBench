#!/usr/bin/env python3
"""
Comprehensive backtest runner — all strategies, full fee + spread reporting.

Usage:
    python run_backtest.py

Fee model: KalshiOct2025  — ceil(0.07 × C × P × (1-P)) taker / 1.75% maker
Spread:    Real orderbook walk from recorded trade tape (maker_taker mode)

Net PnL = gross PnL − fees − spread cost (all three rolled into simulator output)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from oddpool_bench import BenchmarkHarness, SimulatorConfig
from our_agents import PremiumHarvester
from scaled_harvester import ScaledHarvesterV2
from drawdown_harvester import DrawdownHarvester

# ── Config ────────────────────────────────────────────────────────────────────

EPISODES_DIR = Path(__file__).parent / "episodes"
RESULTS_DIR  = Path(__file__).parent / "results_backtest"
RESULTS_DIR.mkdir(exist_ok=True)

config = SimulatorConfig(
    agent_call_cadence_seconds=5.0,
    equity_sample_interval_seconds=30.0,
    verbose=False,
)

harness = BenchmarkHarness(EPISODES_DIR, config)

# ── Strategy definitions ──────────────────────────────────────────────────────

STRATEGIES = {
    "Baseline (mkt, cap=10)": PremiumHarvester(
        min_no_price=93, max_no_price=99, min_edge=0.005,
        max_position_per_market=10, max_total_positions=20,
    ),
    "Scaled (mkt, Kelly=0.25)": ScaledHarvesterV2(
        min_no_price=81, kelly_fraction=0.25, cash_reserve_pct=0.25,
        max_position_per_market=30,
    ),
    "DCA mkt cap=40": DrawdownHarvester(
        min_no_price=81, kelly_fraction=0.10, dca_kelly_fraction=0.08,
        max_position_per_market=40, max_total_positions=60,
        dip_step_cents=3, max_dca_tranches=3,
        use_limit_orders=False,
    ),
    "DCA mkt cap=80 *BEST*": DrawdownHarvester(
        min_no_price=81, kelly_fraction=0.10, dca_kelly_fraction=0.08,
        max_position_per_market=80, max_total_positions=60,
        dip_step_cents=3, max_dca_tranches=3,
        use_limit_orders=False,
    ),
    "DCA maker cap=40": DrawdownHarvester(
        min_no_price=81, kelly_fraction=0.10, dca_kelly_fraction=0.08,
        max_position_per_market=40, max_total_positions=60,
        dip_step_cents=3, max_dca_tranches=3,
        use_limit_orders=True,
    ),
}

# ── Run ───────────────────────────────────────────────────────────────────────

print()
print("=" * 95)
print("  KALSHI BACKTEST — ALL STRATEGIES (fee + spread modeled)")
print(f"  Fee model: KalshiOct2025  taker=7%×P×(1-P)  maker=1.75%×P×(1-P)")
print(f"  Spread: real orderbook walk via trade tape")
print(f"  Episodes: {', '.join(harness.list_episodes())}")
print("=" * 95)
print(f"{'Strategy':<26} {'Net$':>7} {'Ret%':>6} {'Gross$':>7} {'Fees$':>7} {'Fee%':>5} {'MaxDD':>6} {'Sharpe':>7} {'Fills':>6} {'FillR':>6}")
print("-" * 95)

all_results = {}

for name, agent in STRATEGIES.items():
    result = harness.run(agent)

    net_pnl     = sum(r.total_pnl_cents for r in result.episode_results) / 100
    fees        = sum(r.total_fees_cents for r in result.episode_results) / 100
    slippage    = sum(r.total_slippage_cents for r in result.episode_results) / 100
    gross_pnl   = net_pnl + fees  # before Kalshi takes their cut
    fee_pct     = (fees / gross_pnl * 100) if gross_pnl > 0 else 0.0
    contracts   = sum(r.total_contracts_traded for r in result.episode_results)
    max_dd      = max(r.max_drawdown_pct for r in result.episode_results) * 100

    sharpe_vals = [r.sharpe_ratio for r in result.episode_results if r.sharpe_ratio is not None]
    sharpe      = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0.0

    fill_ratios = [r.fill_ratio for r in result.episode_results]
    fill_ratio  = sum(fill_ratios) / len(fill_ratios) * 100

    initial     = result.episode_results[0].initial_equity_cents / 100
    ret_pct     = net_pnl / (initial * len(result.episode_results)) * 100

    all_results[name] = result

    marker = "  <--" if "*BEST*" in name else ""
    print(
        f"{name:<26} "
        f"${net_pnl:>6.2f} "
        f"{ret_pct:>5.1f}% "
        f"${gross_pnl:>6.2f} "
        f"${fees:>6.2f} "
        f"{fee_pct:>4.1f}% "
        f"{max_dd:>5.1f}% "
        f"{sharpe:>7.2f} "
        f"{contracts:>6} "
        f"{fill_ratio:>5.1f}%"
        f"{marker}"
    )

print("=" * 95)
print()

# ── Per-episode breakdown for the best strategy ───────────────────────────────

best_name = "DCA mkt cap=80 *BEST*"
best_result = all_results[best_name]

print(f"  Per-episode detail — {best_name}")
print(f"  {'Episode':<35} {'Net$':>7} {'Fees$':>6} {'Gross$':>7} {'MaxDD':>6} {'Sharpe':>7} {'Contracts':>10}")
print(f"  {'-'*80}")

for r in best_result.episode_results:
    net   = r.total_pnl_cents / 100
    fees  = r.total_fees_cents / 100
    gross = net + fees
    dd    = r.max_drawdown_pct * 100
    sh    = r.sharpe_ratio or 0.0
    print(
        f"  {r.episode_id:<35} "
        f"${net:>6.2f} "
        f"${fees:>5.2f} "
        f"${gross:>6.2f} "
        f"{dd:>5.1f}% "
        f"{sh:>7.2f} "
        f"{r.total_contracts_traded:>10}"
    )

print()

# ── Fee impact analysis ───────────────────────────────────────────────────────

print("  Fee impact analysis (across all strategies):")
print(f"  {'Strategy':<26}  {'Fee/contract¢':>13}  {'Fee drag':>10}")
print(f"  {'-'*55}")
for name, result in all_results.items():
    fees_c    = sum(r.total_fees_cents for r in result.episode_results)
    contracts = sum(r.total_contracts_traded for r in result.episode_results)
    net_pnl_c = sum(r.total_pnl_cents for r in result.episode_results)
    gross_c   = net_pnl_c + fees_c
    fpc       = fees_c / contracts if contracts else 0
    fdrag     = fees_c / gross_c * 100 if gross_c > 0 else 0
    print(f"  {name:<26}  {fpc:>12.2f}¢  {fdrag:>9.1f}%")

print()

# ── Save best result ──────────────────────────────────────────────────────────

best_result.save(RESULTS_DIR / "best_summary.json")
best_result.save_trades(RESULTS_DIR / "best_trades.json")
best_result.save_equity_csv(RESULTS_DIR / "best_equity.csv")
try:
    best_result.save_equity_curve(RESULTS_DIR / "best_equity.png")
except Exception as e:
    print(f"  (Plot skipped: {e})")

print(f"  Results saved → {RESULTS_DIR}/")
