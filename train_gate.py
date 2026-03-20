#!/usr/bin/env python3
"""Train the ML confidence gate from extracted signal features.

1. Loads signal_features.parquet
2. Prints hard gate analysis (populates gate config)
3. Runs Leave-One-Episode-Out cross-validation
4. Trains final model if AUC > 0.55
5. Saves model or disables ML gate

Usage:
    cd backtest-framework
    python train_gate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from gates.confidence_gate import ConfidenceGate


FEATURE_COLS = [
    "composite_score",
    "n_sources",
    "spread_cents",
    "hours_to_close_log",
    "book_imbalance",
    "yes_mid",
    "source_ph_int",
]

DATA_DIR = Path(__file__).parent / "data"


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns needed for training."""
    df = df.copy()
    df["hours_to_close_log"] = np.log1p(df["hours_to_close"].clip(lower=0.01))
    df["source_ph_int"] = df["source_ph"].astype(int)
    return df


def main():
    features_path = DATA_DIR / "signal_features.parquet"
    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run extract_features.py first.")
        return

    df = pd.read_parquet(features_path)
    df = df[df["signal_correct"].notna()].copy()
    df["signal_correct"] = df["signal_correct"].astype(bool)
    df = prepare_df(df)

    print(f"\n{'='*70}")
    print("  ML GATE TRAINING")
    print(f"{'='*70}")
    print(f"Total labeled signals: {len(df)}")
    print(f"Correct: {df['signal_correct'].sum()} ({df['signal_correct'].mean()*100:.1f}%)")
    print(f"Incorrect: {(~df['signal_correct']).sum()} ({(~df['signal_correct']).mean()*100:.1f}%)")

    # --- Hard gate analysis ---
    print(f"\n{'='*70}")
    print("  HARD GATE ANALYSIS")
    print(f"{'='*70}")

    print("\nWin rate by category:")
    for cat, g in df.groupby("category"):
        print(f"  {cat}: {len(g)} signals, {g['signal_correct'].mean()*100:.1f}% correct")

    print("\nWin rate by direction:")
    for d, g in df.groupby("direction"):
        print(f"  {d}: {len(g)} signals, {g['signal_correct'].mean()*100:.1f}% correct")

    print("\nWin rate by source combination:")
    for (ph, mom, tape), g in df.groupby(["source_ph", "source_momentum", "source_tape"]):
        sources = []
        if ph: sources.append("PH")
        if mom: sources.append("Mom")
        if tape: sources.append("Tape")
        label = "+".join(sources) if sources else "none"
        print(f"  {label}: {len(g)} signals, {g['signal_correct'].mean()*100:.1f}% correct")

    print("\nWin rate by spread:")
    bins = pd.cut(df["spread_cents"], bins=[0, 2, 5, 10, 100], labels=["1-2c", "3-5c", "6-10c", "10c+"])
    for b, g in df.groupby(bins, observed=True):
        print(f"  {b}: {len(g)} signals, {g['signal_correct'].mean()*100:.1f}% correct")

    # --- Apply hard gate filter for ML training ---
    # The ML gate trains on signals that PASS the hard gate
    # (no point teaching it about signals we'll block anyway)
    from gates.hard_gate import HardGate
    hard_gate = HardGate()

    mask = []
    for _, row in df.iterrows():
        blocked, _ = hard_gate.should_block(row.to_dict())
        mask.append(not blocked)
    df_filtered = df[mask].copy()

    print(f"\nAfter hard gate: {len(df_filtered)} signals remain ({len(df)-len(df_filtered)} blocked)")
    print(f"  Correct: {df_filtered['signal_correct'].mean()*100:.1f}%")

    if len(df_filtered) < 100:
        print("\nToo few signals after hard gate for ML training. ML gate DISABLED.")
        (DATA_DIR / "ml_gate_disabled.txt").write_text("Insufficient data after hard gate filtering")
        return

    # --- Leave-One-Episode-Out cross-validation ---
    print(f"\n{'='*70}")
    print("  LEAVE-ONE-EPISODE-OUT CROSS-VALIDATION")
    print(f"{'='*70}")

    episodes = df_filtered["episode_id"].unique()
    episodes = [e for e in episodes if len(df_filtered[df_filtered["episode_id"] == e]) >= 10]
    print(f"Usable episodes: {episodes}")

    if len(episodes) < 2:
        print("Need at least 2 episodes with sufficient data. ML gate DISABLED.")
        (DATA_DIR / "ml_gate_disabled.txt").write_text("Insufficient episodes for LOEO")
        return

    X_all = df_filtered[FEATURE_COLS].values
    y_all = df_filtered["signal_correct"].values.astype(int)

    loeo_results = []
    for held_out in episodes:
        train_mask = df_filtered["episode_id"] != held_out
        test_mask = df_filtered["episode_id"] == held_out

        X_train = df_filtered.loc[train_mask, FEATURE_COLS].values
        y_train = df_filtered.loc[train_mask, "signal_correct"].values.astype(int)
        X_test = df_filtered.loc[test_mask, FEATURE_COLS].values
        y_test = df_filtered.loc[test_mask, "signal_correct"].values.astype(int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  {held_out}: SKIPPED (single class in train or test)")
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42)
        model.fit(X_train_s, y_train)

        y_prob = model.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_prob)

        # Find threshold where precision >= 50%
        valid = precision_arr >= 0.50
        if valid.any():
            idx = np.where(valid)[0][-1]  # highest recall with precision >= 50%
            best_thresh = thresholds[idx] if idx < len(thresholds) else 0.5
            best_prec = precision_arr[idx]
            best_recall = recall_arr[idx]
        else:
            best_thresh = 0.5
            best_prec = 0.0
            best_recall = 0.0

        print(f"  {held_out}: AUC={auc:.3f}, prec@50%={best_prec:.2f}, recall={best_recall:.2f}, thresh={best_thresh:.3f}, n_test={len(y_test)}")
        loeo_results.append({
            "episode": held_out,
            "auc": auc,
            "precision": best_prec,
            "recall": best_recall,
            "threshold": best_thresh,
            "n_test": len(y_test),
        })

    if not loeo_results:
        print("\nNo valid LOEO folds. ML gate DISABLED.")
        (DATA_DIR / "ml_gate_disabled.txt").write_text("No valid LOEO folds")
        return

    mean_auc = np.mean([r["auc"] for r in loeo_results])
    print(f"\n  Mean AUC: {mean_auc:.3f}")

    if mean_auc < 0.55:
        print(f"\n  Mean AUC {mean_auc:.3f} < 0.55 — ML gate DISABLED")
        print("  Hard gate + composite threshold will be used instead.")
        (DATA_DIR / "ml_gate_disabled.txt").write_text(f"Mean AUC {mean_auc:.3f} below threshold 0.55")
        return

    # --- Train final model on all data ---
    print(f"\n{'='*70}")
    print("  TRAINING FINAL MODEL")
    print(f"{'='*70}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    model = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42)
    model.fit(X_scaled, y_all)

    # Calibrate threshold from LOEO results
    mean_threshold = np.mean([r["threshold"] for r in loeo_results])
    # Be conservative: use the higher of mean threshold and 0.55
    final_threshold = max(mean_threshold, 0.55)

    print(f"  Model coefficients:")
    for name, coef in zip(FEATURE_COLS, model.coef_[0]):
        print(f"    {name}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
    print(f"  Calibrated threshold: {final_threshold:.3f}")

    # In-sample check
    y_prob_all = model.predict_proba(X_scaled)[:, 1]
    passes = y_prob_all >= final_threshold
    if passes.any():
        pass_wr = y_all[passes].mean() * 100
        print(f"  In-sample: {passes.sum()} signals pass ({passes.mean()*100:.1f}%), {pass_wr:.1f}% correct")
    else:
        print(f"  WARNING: No signals pass at threshold {final_threshold:.3f}")

    # Save model
    gate = ConfidenceGate()
    gate.model = model
    gate.scaler = scaler
    gate.threshold = final_threshold
    gate.enabled = True

    model_path = DATA_DIR / "confidence_gate_model.pkl"
    gate.save(str(model_path))
    print(f"\n  Model saved to {model_path}")

    # Remove disabled sentinel if it exists
    disabled_path = DATA_DIR / "ml_gate_disabled.txt"
    if disabled_path.exists():
        disabled_path.unlink()

    print(f"\n  ML gate ENABLED (AUC={mean_auc:.3f}, threshold={final_threshold:.3f})")


if __name__ == "__main__":
    main()
