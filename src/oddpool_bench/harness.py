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

    def save_equity_gif(
        self,
        path: Path,
        episode_id: str = None,
        figsize: tuple[int, int] = (10, 6),
        fps: int = 15,
        duration_seconds: float = 8.0,
    ) -> None:
        """
        Save animated GIF of PnL over time for social media sharing.
        
        Creates an animation that shows the equity curve building up over time,
        perfect for sharing trading bot performance on social media.
        
        Args:
            path: Output path (should end in .gif). If episode_id is None and
                  multiple episodes exist, will append episode_id to filename.
            episode_id: Specific episode to animate. If None and only one episode,
                       uses that one. If None and multiple episodes, creates 
                       separate GIFs for each.
            figsize: Figure size in inches (width, height)
            fps: Frames per second for the animation
            duration_seconds: Total duration of the GIF
        
        Requires: pip install matplotlib pillow
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            raise ImportError(
                "matplotlib and pillow are required. Install with: pip install matplotlib pillow"
            )
        
        path = Path(path)
        
        # If path is a directory, construct filenames
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            base_dir = path
            use_directory = True
        else:
            # path is a file, extract directory
            base_dir = path.parent
            base_dir.mkdir(parents=True, exist_ok=True)
            use_directory = False
        
        # Determine which episodes to animate
        if episode_id is not None:
            # Find specific episode
            episodes_to_animate = [
                ep for ep in self.episode_results if ep.episode_id == episode_id
            ]
            if not episodes_to_animate:
                raise ValueError(f"Episode not found: {episode_id}")
        else:
            episodes_to_animate = self.episode_results
        
        # Generate GIF for each episode
        for ep_idx, episode in enumerate(episodes_to_animate):
            # Determine output path
            if use_directory:
                # Directory was passed - create filenames for each episode
                ep_path = base_dir / f"pnl_animation_{episode.episode_id}.gif"
            elif len(episodes_to_animate) > 1:
                # Multiple episodes - add episode ID to filename
                stem = path.stem
                suffix = path.suffix or ".gif"
                ep_path = path.parent / f"{stem}_{episode.episode_id}{suffix}"
            else:
                ep_path = path
            
            self._save_single_episode_gif(
                episode=episode,
                path=ep_path,
                figsize=figsize,
                fps=fps,
                duration_seconds=duration_seconds,
            )
    
    def _save_single_episode_gif(
        self,
        episode,
        path: Path,
        figsize: tuple[int, int],
        fps: int,
        duration_seconds: float,
    ) -> None:
        """Generate animated GIF for a single episode."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.animation import FuncAnimation, PillowWriter
        from datetime import timedelta
        
        # Extract equity data
        initial = episode.initial_equity_cents
        times = [snap.ts for snap in episode.equity_curve]
        pnls = [(snap.equity_cents - initial) / 100 for snap in episode.equity_curve]
        
        if not times:
            print(f"No equity data for episode {episode.episode_id}")
            return
        
        # Determine number of frames
        n_frames = int(fps * duration_seconds)
        n_points = len(times)
        
        # Convert times to numeric for interpolation
        start_time = times[0]
        end_time = times[-1]
        total_duration = (end_time - start_time).total_seconds()
        
        # Convert to seconds from start for interpolation
        times_numeric = [(t - start_time).total_seconds() for t in times]
        
        # Pre-compute interpolated data for each frame
        # Animation progresses smoothly through time, with line extending to current time
        frame_data = []
        for frame in range(n_frames):
            progress = (frame + 1) / n_frames
            current_time_sec = progress * total_duration
            
            # Find all data points up to current time, plus interpolate to current position
            x_vals = []
            y_vals = []
            
            for i, t_sec in enumerate(times_numeric):
                if t_sec <= current_time_sec:
                    x_vals.append(times[i])
                    y_vals.append(pnls[i])
                else:
                    break
            
            # If we're between data points, extend line to current time position
            # with the last known PnL value (step interpolation - PnL doesn't change
            # until there's a new trade)
            if x_vals and current_time_sec > times_numeric[len(x_vals) - 1]:
                # Add current position with last known PnL
                current_datetime = start_time + timedelta(seconds=current_time_sec)
                x_vals.append(current_datetime)
                y_vals.append(y_vals[-1] if y_vals else 0)
            
            frame_data.append((x_vals, y_vals))
        
        # Setup figure with dark theme for social media
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Style settings
        positive_color = '#00ff88'
        negative_color = '#ff4444'
        
        # Determine line color based on final PnL
        final_pnl = pnls[-1] if pnls else 0
        line_color = positive_color if final_pnl >= 0 else negative_color
        
        # Initialize plot elements
        line, = ax.plot([], [], color=line_color, linewidth=2.5, alpha=0.9)
        fill_collection = None
        
        # Set axis limits with padding
        y_min = min(pnls) if pnls else -1
        y_max = max(pnls) if pnls else 1
        y_range = max(abs(y_min), abs(y_max), 0.5)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-y_range * 1.2, y_range * 1.2)
        
        # Zero line
        ax.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Formatting
        ax.set_ylabel('PnL ($)', fontsize=12, color='white')
        ax.set_xlabel('Time', fontsize=12, color='white')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        
        # Title with episode name
        episode_short = episode.episode_id[:20] + "..." if len(episode.episode_id) > 20 else episode.episode_id
        title = ax.set_title(f'{episode_short}\nPnL: $0.00', fontsize=14, fontweight='bold', 
                            color='white', pad=10)
        
        # Add watermarks - oddpool.com on left, PredictionMarketBench on right
        fig.text(0.02, 0.02, 'oddpool.com', fontsize=9, 
                color='white', alpha=0.5, ha='left', va='bottom')
        fig.text(0.98, 0.02, 'PredictionMarketBench', fontsize=9, 
                color='white', alpha=0.5, ha='right', va='bottom')
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(frame):
            nonlocal fill_collection
            
            # Get pre-computed data for this frame
            x_data, y_data = frame_data[frame]
            
            if not x_data:
                return line,
            
            # Update line
            line.set_data(x_data, y_data)
            
            # Update fill (green for profit, red for loss)
            if fill_collection is not None:
                fill_collection.remove()
            
            current_pnl = y_data[-1] if y_data else 0
            fill_color = positive_color if current_pnl >= 0 else negative_color
            fill_collection = ax.fill_between(x_data, 0, y_data, alpha=0.3, color=fill_color)
            
            # Update line color based on current PnL
            line.set_color(fill_color)
            
            # Update title with current PnL
            title.set_text(f'{episode_short}\nPnL: ${current_pnl:+.2f}')
            title.set_color(fill_color)
            
            return line, fill_collection
        
        # Create animation
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=1000/fps, blit=False
        )
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(path, writer=writer, dpi=100)
        plt.close()
        
        # Reset style
        plt.style.use('default')
        
        print(f"Saved animated PnL GIF to {path}")
        print(f"  Episode: {episode.episode_id}")
        print(f"  Duration: {duration_seconds}s at {fps} fps ({n_frames} frames)")
        print(f"  Final PnL: ${final_pnl:+.2f}")

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
