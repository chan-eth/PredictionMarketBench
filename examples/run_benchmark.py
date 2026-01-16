#!/usr/bin/env python3
"""
Example script showing how to run the benchmark.

Usage:
    python examples/run_benchmark.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oddpool_bench import BenchmarkHarness, SimulatorConfig
from example_agents import PassiveAgent, RandomAgent, MomentumAgent


def main():
    # Configure the simulator
    config = SimulatorConfig(
        agent_call_cadence_seconds=5.0,  # Call agent every 5 seconds
        equity_sample_interval_seconds=60.0,  # Sample equity every minute
        verbose=True,
    )
    
    # Initialize harness
    episodes_dir = Path(__file__).parent.parent / "episodes"
    harness = BenchmarkHarness(episodes_dir, config)
    
    # List available episodes
    episodes = harness.list_episodes()
    print(f"Available episodes: {episodes}")
    
    if not episodes:
        print("\nNo episodes found! Run the data conversion script first:")
        print("  python scripts/convert_raw_data.py orderbook_states_*.csv")
        return
    
    # Test different agents
    agents = {
        "Passive": PassiveAgent(),
        "Random": RandomAgent(trade_probability=0.05),
        "Momentum": MomentumAgent(lookback_steps=10, threshold_cents=2),
    }
    
    for name, agent in agents.items():
        print(f"\n{'='*60}")
        print(f"Running benchmark with {name} agent")
        print('='*60)
        
        result = harness.run(agent)
        result.print_summary()
        
        # Create output directory for this agent
        output_dir = Path(f"results_{name.lower()}")
        output_dir.mkdir(exist_ok=True)
        
        # Save summary results
        result.save(output_dir / "summary.json")
        
        # Save all trades
        result.save_trades(output_dir / "trades.json")
        
        # Save equity curve data as CSV
        result.save_equity_csv(output_dir / "equity_curve.csv")
        
        # Save equity curve plot (requires matplotlib)
        try:
            result.save_equity_curve(output_dir / "equity_curve.png")
        except ImportError:
            print("  (Skipping plot - install matplotlib: pip install matplotlib)")
        
        print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
