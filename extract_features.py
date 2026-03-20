#!/usr/bin/env python3
"""Extract signal features from all episodes for ML gate training.

Usage:
    cd backtest-framework
    python extract_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from oddpool_bench import BenchmarkHarness, SimulatorConfig
from feature_extractor import FeatureExtractorAgent


def main():
    config = SimulatorConfig(
        agent_call_cadence_seconds=5.0,
        equity_sample_interval_seconds=60.0,
        verbose=True,
    )

    episodes_dir = Path(__file__).parent / "episodes"
    harness = BenchmarkHarness(episodes_dir, config)

    episodes = harness.list_episodes()
    print(f"Available episodes: {episodes}")

    agent = FeatureExtractorAgent()
    result = harness.run(agent)

    # Convert to DataFrame
    df = pd.DataFrame(agent.feature_rows)

    # Summary
    print(f"\n{'='*70}")
    print("  FEATURE EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total signal events: {len(df)}")

    if len(df) == 0:
        print("No signals extracted! Check signal generation logic.")
        return

    # Filter out unlabelable rows
    labeled = df[df["signal_correct"].notna()].copy()
    print(f"Labeled signals: {len(labeled)}")
    print(f"Unlabeled (no settlement): {len(df) - len(labeled)}")

    if len(labeled) > 0:
        print(f"\nClass balance:")
        print(f"  Correct: {labeled['signal_correct'].sum()} ({labeled['signal_correct'].mean()*100:.1f}%)")
        print(f"  Incorrect: {(~labeled['signal_correct']).sum()} ({(~labeled['signal_correct']).mean()*100:.1f}%)")

        print(f"\nPer-episode breakdown:")
        for ep_id, group in labeled.groupby("episode_id"):
            n = len(group)
            wr = group["signal_correct"].mean() * 100
            print(f"  {ep_id}: {n} signals, {wr:.1f}% correct")

        print(f"\nPer-category breakdown:")
        for cat, group in labeled.groupby("category"):
            n = len(group)
            wr = group["signal_correct"].mean() * 100
            print(f"  {cat}: {n} signals, {wr:.1f}% correct")

        print(f"\nPer-direction breakdown:")
        for d, group in labeled.groupby("direction"):
            n = len(group)
            wr = group["signal_correct"].mean() * 100
            print(f"  {d}: {n} signals, {wr:.1f}% correct")

        print(f"\nComposite score bins:")
        labeled["composite_bin"] = pd.cut(
            labeled["composite_score"],
            bins=[0, 0.3, 0.5, 0.7, 1.0, float("inf")],
            labels=["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0", "1.0+"],
        )
        for bin_label, group in labeled.groupby("composite_bin", observed=True):
            n = len(group)
            wr = group["signal_correct"].mean() * 100
            print(f"  {bin_label}: {n} signals, {wr:.1f}% correct")

        print(f"\nSpread bins:")
        labeled["spread_bin"] = pd.cut(
            labeled["spread_cents"],
            bins=[0, 2, 5, 10, 100],
            labels=["1-2c", "3-5c", "6-10c", "10c+"],
        )
        for bin_label, group in labeled.groupby("spread_bin", observed=True):
            n = len(group)
            wr = group["signal_correct"].mean() * 100
            print(f"  {bin_label}: {n} signals, {wr:.1f}% correct")

        print(f"\nSource combination win rates:")
        for (ph, mom, tape), group in labeled.groupby(["source_ph", "source_momentum", "source_tape"]):
            sources = []
            if ph: sources.append("PH")
            if mom: sources.append("Mom")
            if tape: sources.append("Tape")
            label = "+".join(sources) if sources else "none"
            n = len(group)
            wr = group["signal_correct"].mean() * 100
            print(f"  {label}: {n} signals, {wr:.1f}% correct")

    # Save
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "signal_features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    # Also save CSV for easy inspection
    csv_path = output_dir / "signal_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
