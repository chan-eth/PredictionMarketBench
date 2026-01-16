"""
Benchmark harness for running agents across episodes.

Provides the main entry point for benchmark evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from .agent import Agent
from .simulator import Simulator, SimulatorConfig
from .types import EpisodeResult, OrderResult, EquitySnapshot
from .metrics import aggregate_metrics
from .data import list_episodes


@dataclass
class BenchmarkResult:
    """Results from running a benchmark across episodes."""
    
    episode_results: list[EpisodeResult]
    aggregate: dict
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_episodes": len(self.episode_results),
            "episodes": [
                {
                    "episode_id": r.episode_id,
                    "total_pnl_cents": r.total_pnl_cents,
                    "total_pnl_pct": r.total_pnl_pct,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "sharpe_ratio": r.sharpe_ratio,
                    "total_contracts_traded": r.total_contracts_traded,
                    "total_fees_cents": r.total_fees_cents,
                    "fill_ratio": r.fill_ratio,
                }
                for r in self.episode_results
            ],
            "aggregate": self.aggregate,
        }
    
    def save(self, path: Path) -> None:
        """Save summary results to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_trades(self, path: Path) -> None:
        """
        Save all trades to a JSON file.
        
        Each trade includes:
        - episode_id
        - timestamp
        - ticker
        - side (YES/NO)
        - action (BUY/SELL)
        - order_type
        - requested_count
        - filled_count
        - fills (price, size, fee per level)
        - total_cost_cents
        - total_fees_cents
        - average_fill_price
        - rejected
        - reject_reason
        """
        path = Path(path)
        
        all_trades = []
        for episode in self.episode_results:
            for order_result in episode.order_results:
                trade = self._order_result_to_dict(episode.episode_id, order_result)
                all_trades.append(trade)
        
        output = {
            "n_trades": len(all_trades),
            "total_contracts": sum(t["filled_count"] for t in all_trades),
            "total_fees_cents": sum(t["total_fees_cents"] for t in all_trades),
            "trades": all_trades,
        }
        
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved {len(all_trades)} trades to {path}")
    
    def _order_result_to_dict(self, episode_id: str, result: OrderResult) -> dict:
        """Convert OrderResult to serializable dict."""
        return {
            "episode_id": episode_id,
            "timestamp": result.ts.isoformat(),
            "ticker": result.order.ticker,
            "side": result.order.side.value,
            "action": result.order.action.value,
            "order_type": result.order.order_type.value,
            "requested_count": result.order.count,
            "limit_price_cents": result.order.limit_price_cents,
            "filled_count": result.filled_count,
            "fills": [
                {
                    "price_cents": f.price_cents,
                    "size": f.size,
                    "fee_cents": f.fee_cents,
                }
                for f in result.fills
            ],
            "total_cost_cents": result.total_cost_cents,
            "total_fees_cents": result.total_fees_cents,
            "average_fill_price": result.average_fill_price,
            "rejected": result.rejected,
            "reject_reason": result.reject_reason,
        }
    
    def save_equity_curve(
        self,
        path: Path,
        figsize: tuple[int, int] = (12, 8),
        show_episodes: bool = True,
    ) -> None:
        """
        Save equity curve plot to file.
        
        Args:
            path: Output path (supports .png, .pdf, .svg)
            figsize: Figure size in inches (width, height)
            show_episodes: Whether to show separate lines per episode
        
        Requires matplotlib: pip install matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )
        
        path = Path(path)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        ax_equity = axes[0]
        ax_drawdown = axes[1]
        
        colors = plt.cm.tab10.colors
        
        for i, episode in enumerate(self.episode_results):
            color = colors[i % len(colors)]
            
            # Extract data
            times = [snap.ts for snap in episode.equity_curve]
            equities = [snap.equity_cents / 100 for snap in episode.equity_curve]  # Convert to dollars
            
            # Calculate PnL (relative to initial)
            initial = episode.initial_equity_cents / 100
            pnls = [e - initial for e in equities]
            
            # Calculate drawdown
            peak = equities[0]
            drawdowns = []
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                drawdowns.append(dd)
            
            label = f"{episode.episode_id} (PnL: ${episode.total_pnl_cents/100:+.2f})"
            
            if show_episodes:
                ax_equity.plot(times, pnls, label=label, color=color, linewidth=1.5)
                ax_drawdown.fill_between(times, 0, drawdowns, alpha=0.3, color=color)
                ax_drawdown.plot(times, drawdowns, color=color, linewidth=1)
            else:
                ax_equity.plot(times, pnls, color=color, linewidth=1.5, alpha=0.7)
        
        # Equity plot formatting
        ax_equity.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax_equity.set_ylabel('PnL ($)', fontsize=11)
        ax_equity.set_title('Equity Curve - PnL Over Time', fontsize=13, fontweight='bold')
        ax_equity.grid(True, alpha=0.3)
        if show_episodes and len(self.episode_results) <= 10:
            ax_equity.legend(loc='best', fontsize=9)
        
        # Drawdown plot formatting
        ax_drawdown.set_ylabel('Drawdown (%)', fontsize=11)
        ax_drawdown.set_xlabel('Time', fontsize=11)
        ax_drawdown.grid(True, alpha=0.3)
        ax_drawdown.invert_yaxis()  # Drawdown shown as negative
        
        # Format x-axis dates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.tick_params(axis='x', rotation=45)
        
        # Add summary stats
        total_pnl = sum(r.total_pnl_cents for r in self.episode_results) / 100
        avg_dd = self.aggregate.get("max_drawdown_pct", {}).get("mean", 0) * 100
        sharpe = self.aggregate.get("sharpe_ratio", {}).get("mean")
        
        stats_text = f"Total PnL: ${total_pnl:+.2f} | Avg Max DD: {avg_dd:.1f}%"
        if sharpe is not None:
            stats_text += f" | Avg Sharpe: {sharpe:.2f}"
        
        fig.suptitle(stats_text, y=0.02, fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved equity curve to {path}")
    
    def save_equity_csv(self, path: Path) -> None:
        """
        Save equity curve data to CSV file.
        
        Columns: episode_id, timestamp, cash_cents, position_value_cents, equity_cents, pnl_cents
        """
        path = Path(path)
        
        rows = []
        for episode in self.episode_results:
            initial = episode.initial_equity_cents
            for snap in episode.equity_curve:
                rows.append({
                    "episode_id": episode.episode_id,
                    "timestamp": snap.ts.isoformat(),
                    "cash_cents": snap.cash_cents,
                    "position_value_cents": snap.position_value_cents,
                    "equity_cents": snap.equity_cents,
                    "pnl_cents": snap.equity_cents - initial,
                })
        
        # Write CSV
        if rows:
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"Saved {len(rows)} equity snapshots to {path}")
    
    def print_summary(self) -> None:
        """Print a summary of results to stdout."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"\nEpisodes: {len(self.episode_results)}")
        
        print("\nPer-Episode Results:")
        print("-" * 60)
        for r in self.episode_results:
            pnl_dollars = r.total_pnl_cents / 100
            print(f"  {r.episode_id}:")
            print(f"    PnL: ${pnl_dollars:+.2f} ({r.total_pnl_pct * 100:+.2f}%)")
            print(f"    Max Drawdown: {r.max_drawdown_pct * 100:.2f}%")
            if r.sharpe_ratio is not None:
                print(f"    Sharpe: {r.sharpe_ratio:.2f}")
            print(f"    Contracts: {r.total_contracts_traded}")
            print(f"    Fill Ratio: {r.fill_ratio * 100:.1f}%")
        
        print("\nAggregate Metrics:")
        print("-" * 60)
        agg = self.aggregate
        if agg:
            pnl = agg.get("total_pnl_cents", {})
            print(f"  Total PnL: ${pnl.get('sum', 0) / 100:.2f}")
            print(f"  Mean PnL: ${pnl.get('mean', 0) / 100:.2f}")
            
            pnl_pct = agg.get("total_pnl_pct", {})
            print(f"  Mean Return: {pnl_pct.get('mean', 0) * 100:.2f}%")
            
            dd = agg.get("max_drawdown_pct", {})
            print(f"  Mean Max DD: {dd.get('mean', 0) * 100:.2f}%")
            print(f"  Worst Max DD: {dd.get('max', 0) * 100:.2f}%")
            
            sharpe = agg.get("sharpe_ratio", {})
            if sharpe.get("mean") is not None:
                print(f"  Mean Sharpe: {sharpe['mean']:.2f}")
            
            fill = agg.get("fill_ratio", {})
            print(f"  Mean Fill Ratio: {fill.get('mean', 0) * 100:.1f}%")
        
        print("=" * 60 + "\n")


class BenchmarkHarness:
    """
    Main harness for running benchmark evaluations.
    
    Usage:
        harness = BenchmarkHarness(episodes_dir="./episodes")
        result = harness.run(my_agent)
        result.print_summary()
    """
    
    def __init__(
        self,
        episodes_dir: str | Path = "./episodes",
        config: Optional[SimulatorConfig] = None,
    ):
        """
        Initialize the benchmark harness.
        
        Args:
            episodes_dir: Directory containing episode folders
            config: Simulator configuration (shared across all episodes)
        """
        self.episodes_dir = Path(episodes_dir)
        self.config = config or SimulatorConfig()
        
        if not self.episodes_dir.exists():
            raise ValueError(f"Episodes directory not found: {self.episodes_dir}")
    
    def list_episodes(self) -> list[str]:
        """List available episode IDs."""
        return list_episodes(self.episodes_dir)
    
    def run_episode(
        self,
        agent: Agent,
        episode_id: str,
    ) -> EpisodeResult:
        """
        Run a single episode.
        
        Args:
            agent: Agent to evaluate
            episode_id: Episode ID to run
            
        Returns:
            EpisodeResult with performance metrics
        """
        episode_dir = self.episodes_dir / episode_id
        if not episode_dir.exists():
            raise ValueError(f"Episode not found: {episode_id}")
        
        simulator = Simulator(episode_dir, self.config)
        return simulator.run(agent)
    
    def run(
        self,
        agent: Agent,
        episode_ids: Optional[list[str]] = None,
    ) -> BenchmarkResult:
        """
        Run the benchmark across all (or specified) episodes.
        
        Args:
            agent: Agent to evaluate
            episode_ids: Optional list of specific episodes to run.
                        If None, runs all available episodes.
                        
        Returns:
            BenchmarkResult with per-episode and aggregate metrics
        """
        if episode_ids is None:
            episode_ids = self.list_episodes()
        
        if not episode_ids:
            raise ValueError("No episodes found to run")
        
        results = []
        for episode_id in episode_ids:
            if self.config.verbose:
                print(f"\nRunning episode: {episode_id}")
            
            result = self.run_episode(agent, episode_id)
            results.append(result)
        
        # Aggregate metrics
        aggregate = aggregate_metrics(results)
        
        return BenchmarkResult(
            episode_results=results,
            aggregate=aggregate,
        )


def run_benchmark(
    agent: Agent,
    episodes_dir: str | Path = "./episodes",
    episode_ids: Optional[list[str]] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Convenience function to run a benchmark.
    
    Args:
        agent: Agent to evaluate
        episodes_dir: Directory containing episode folders
        episode_ids: Optional list of specific episodes
        verbose: Whether to print progress
        
    Returns:
        BenchmarkResult
    """
    config = SimulatorConfig(verbose=verbose)
    harness = BenchmarkHarness(episodes_dir, config)
    return harness.run(agent, episode_ids)
